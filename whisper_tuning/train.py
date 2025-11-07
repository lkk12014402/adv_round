import torch
import torch.nn as nn
from transformers import WhisperModel
from typing import List
from quant_layer import QuantizedLinearActMXFP4

MODEL_NAME = "openai/whisper-tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 训练配置
BLOCK_SIZE_W = 32
BLOCK_SIZE_A = 32
LR_WARMUP = 5e-3
LR_MAIN = 1e-3
DELTA_REG_W = 1e-4
DELTA_REG_A = 1e-4
EPOCHS_WARMUP = 2
EPOCHS_MAIN = 4
BATCHES_PER_EPOCH = 40
BATCH_SIZE = 4
FRAMES = 3000
PRINT_INTERVAL = 10
print("??????????????????????????????????????????????????????????????????????")

def gen_mel(batch, frames, device):
    return torch.randn(batch, 80, frames, device=device)

def replace_linears_with_quant_act(model: WhisperModel) -> List[QuantizedLinearActMXFP4]:
    quant_layers = []
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            qlayer = QuantizedLinearActMXFP4(
                module,
                weight_block_size=BLOCK_SIZE_W,
                act_block_size=BLOCK_SIZE_A,
                weight_beta=3.0,
                act_beta=4.0,
                range_scale=0.5,
                weight_mode="soft",
                act_mode="soft-ste",
                enable_weight_perturb=True,
                quant_activation_input=True,
                quant_activation_output=False,
                enable_activation_perturb=False,
                shared_act_rounding=False
            )
            setattr(model, name, qlayer)
            quant_layers.append(qlayer)
        else:
            quant_layers.extend(replace_linears_with_quant_act(module))
    return quant_layers

def collect_all_delta_params(layers: List[QuantizedLinearActMXFP4]):
    params = []
    for l in layers:
        if l.weight_rounding.delta_raw is not None and l.enable_weight_perturb:
            params.append(l.weight_rounding.delta_raw)
        if l.act_in_quant and l.act_in_quant.rounding_module.delta_raw is not None and l.enable_activation_perturb:
            params.append(l.act_in_quant.rounding_module.delta_raw)
        if l.act_out_quant and l.act_out_quant.rounding_module.delta_raw is not None and l.enable_activation_perturb:
            params.append(l.act_out_quant.rounding_module.delta_raw)
    return params

def run_epoch(stage, quant_model, ref_model, layers, optimizer):
    mse_fn = nn.MSELoss()
    total_loss = 0; total_mse = 0; total_reg = 0
    for step in range(1, BATCHES_PER_EPOCH + 1):
        mel = gen_mel(BATCH_SIZE, FRAMES, DEVICE)
        with torch.no_grad():
            ref_enc = ref_model.encoder(mel).last_hidden_state
        pred_enc = quant_model.encoder(mel).last_hidden_state

        mse = mse_fn(pred_enc, ref_enc)
        # 正则：权重 + 激活 δ
        reg_terms_w = []
        reg_terms_a = []
        for l in layers:
            ds_w = l.weight_rounding.constrained_delta()
            if ds_w is not None and l.enable_weight_perturb:
                reg_terms_w.append(ds_w.pow(2).mean())

            if l.act_in_quant and l.enable_activation_perturb:
                ds_in = l.act_in_quant.rounding_module.constrained_delta()
                if ds_in is not None:
                    reg_terms_a.append(ds_in.pow(2).mean())
            if l.act_out_quant and l.enable_activation_perturb:
                ds_out = l.act_out_quant.rounding_module.constrained_delta()
                if ds_out is not None:
                    reg_terms_a.append(ds_out.pow(2).mean())
        reg_w = DELTA_REG_W * (torch.stack(reg_terms_w).mean() if reg_terms_w else torch.zeros((), device=DEVICE))
        reg_a = DELTA_REG_A * (torch.stack(reg_terms_a).mean() if reg_terms_a else torch.zeros((), device=DEVICE))
        loss = mse + reg_w + reg_a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_reg += (reg_w + reg_a).item()

        for l in layers:
            print(l.weight_rounding.delta_raw)
            print(l.weight_rounding.delta_raw.grad)

        exit()
        if step % PRINT_INTERVAL == 0 or step == 1:
            avg_w_delta = torch.stack([
                l.weight_rounding.constrained_delta().mean()
                for l in layers if l.weight_rounding.delta_raw is not None
            ]).mean().item()
            if any(l.act_in_quant for l in layers):
                avg_a_delta = torch.stack([
                    l.act_in_quant.rounding_module.constrained_delta().mean()
                    for l in layers if l.act_in_quant and l.act_in_quant.rounding_module.delta_raw is not None
                ]).mean().item()
            else:
                avg_a_delta = 0.0
            print(f"[{stage} step {step}/{BATCHES_PER_EPOCH}] "
                  f"loss={loss.item():.6f} mse={mse.item():.6f} reg={reg_w.item()+reg_a.item():.6f} "
                  f"w_delta_mean={avg_w_delta:.4f} a_delta_mean={avg_a_delta:.4f}")

    return {
        "stage": stage,
        "loss": total_loss / BATCHES_PER_EPOCH,
        "mse": total_mse / BATCHES_PER_EPOCH,
        "reg": total_reg / BATCHES_PER_EPOCH
    }

def main():
    print("===================================================================")
    torch.manual_seed(1234)
    ref_model = WhisperModel.from_pretrained(MODEL_NAME).to(DEVICE)
    ref_model.eval()
    for p in ref_model.parameters(): p.requires_grad = False

    quant_model = WhisperModel.from_pretrained(MODEL_NAME).to(DEVICE)
    quant_model.eval()
    for p in quant_model.parameters(): p.requires_grad = False

    print("Replacing linear layers with joint weight+activation quantizers...")
    layers = replace_linears_with_quant_act(quant_model)
    print(f"Total quantized layers: {len(layers)}")
    print(layers)

    params = collect_all_delta_params(layers)
    print(f"Total trainable delta parameter tensors: {len(params)} "
          f" | elements={sum(p.numel() for p in params)}")

    optimizer = torch.optim.Adam(params, lr=LR_WARMUP)

    # Warmup: 权重用 soft, 激活用 soft-ste（或都 soft）
    print("\n=== Warmup Phase ===")
    for l in layers:
        l.set_weight_mode("soft", beta=3.0, enable_perturb=True)
        l.set_activation_mode("soft", beta=4.0, enable_perturb=True)
    for ep in range(1, EPOCHS_WARMUP + 1):
        stats = run_epoch(f"warmup-ep{ep}", quant_model, ref_model, layers, optimizer)
        print(f"[Warmup {ep}] avg_loss={stats['loss']:.6f} avg_mse={stats['mse']:.6f}")

    # Main Phase: 切换到 soft-ste
    print("\n=== Main Phase (STE) ===")
    for g in optimizer.param_groups: g['lr'] = LR_MAIN
    for l in layers:
        l.set_weight_mode("soft-ste", beta=7.0, enable_perturb=True)
        l.set_activation_mode("soft-ste", beta=6.0, enable_perturb=True)

    for ep in range(1, EPOCHS_MAIN + 1):
        stats = run_epoch(f"main-ep{ep}", quant_model, ref_model, layers, optimizer)
        print(f"[Main {ep}] avg_loss={stats['loss']:.6f} avg_mse={stats['mse']:.6f}")

    # Evaluation: Hard vs Hard(no perturb)
    print("\n=== Evaluation (hard) ===")
    mel = gen_mel(BATCH_SIZE, FRAMES, DEVICE)
    with torch.no_grad():
        ref_out = ref_model.encoder(mel).last_hidden_state

    for l in layers:
        l.set_weight_mode("hard", beta=7.0, enable_perturb=True)
        l.set_activation_mode("hard", beta=6.0, enable_perturb=True)

    with torch.no_grad():
        pred_hard = quant_model.encoder(mel).last_hidden_state
    mse_hard = (pred_hard - ref_out).pow(2).mean().item()
    print(f"[Hard Perturb Enabled] MSE={mse_hard:.6f}")

    for l in layers:
        l.set_weight_mode("hard", enable_perturb=False)
        l.set_activation_mode("hard", enable_perturb=False)

    with torch.no_grad():
        pred_no_pert = quant_model.encoder(mel).last_hidden_state
    mse_no_pert = (pred_no_pert - ref_out).pow(2).mean().item()
    print(f"[Hard Perturb Disabled] MSE={mse_no_pert:.6f}")
    print(f"ΔMSE (disabled - enabled) = {mse_no_pert - mse_hard:.6f}")

    # 打印部分统计
    print("\nSample Layer Stats (first 2 layers):")
    for i, l in enumerate(layers[:2]):
        print(f"Layer {i} weight_delta={l.weight_delta_stats()} act_delta={l.activation_delta_stats()}")

    print("Done.")

main()
