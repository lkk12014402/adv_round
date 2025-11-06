import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import importlib
from modules.quantized_linear import QuantizedLinear

def build_dummy_model(cfg):
    layers = []
    for _ in range(cfg.depth):
        lin = nn.Linear(cfg.in_dim, cfg.out_dim)
        qlin = QuantizedLinear(
            lin,
            group_size_weight=cfg.group_size_weight,
            group_size_act=cfg.group_size_act,
            act_quant=cfg.act_quant,
            act_use_ema=cfg.act_use_ema,
            act_ema_momentum=cfg.act_ema_momentum,
            act_recompute_interval=cfg.act_recompute_interval
        )
        layers.append(qlin)
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def gather_weight_eps(model):
    eps_params = []
    for m in model.modules():
        if isinstance(m, QuantizedLinear):
            eps_params.append(m.weight_quantizer.eps_param)
    return eps_params

def forward_model(model, x, adversarial_deltas):
    idx = 0
    for module in model:
        if isinstance(module, QuantizedLinear):
            delta = None
            if adversarial_deltas is not None:
                delta = adversarial_deltas[idx]
            x, _ = module(x, adversarial_delta=delta, quantize_activation=True)
            idx += 1
        else:
            x = module(x)
    return x

def generate_adversarial_deltas(model, x, y, loss_fn, cfg):
    eps_params = gather_weight_eps(model)
    deltas = [torch.zeros_like(p) for p in eps_params]

    for layer_i, p in enumerate(eps_params):
        delta = torch.zeros_like(p, requires_grad=True)
        for step in range(cfg.pgd_steps):
            eps = 0.5 * torch.tanh(p)
            total = torch.clamp(eps + delta, -0.5, 0.5)
            current_deltas = []
            counter = 0
            for m in model.modules():
                if isinstance(m, QuantizedLinear):
                    if counter == layer_i:
                        current_deltas.append(total.detach())
                    else:
                        current_deltas.append(None)
                    counter += 1
            logits = forward_model(model, x, current_deltas)
            loss = loss_fn(logits, y)
            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            with torch.no_grad():
                delta = delta + cfg.alpha * torch.sign(grad)
                delta.clamp_(-cfg.rho, cfg.rho)
                total_tmp = torch.clamp(eps + delta, -0.5, 0.5)
                delta = (total_tmp - eps).detach()
                delta.requires_grad_(True)
        deltas[layer_i] = delta.detach()

    return deltas

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    model = build_dummy_model(cfg).to(device)

    # Synthetic dataset
    data_x = torch.randn(cfg.num_samples, cfg.in_dim, device=device)
    data_y = torch.randint(0, cfg.num_classes, (cfg.num_samples,), device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data_x, data_y),
        batch_size=config.batch_size,
        shuffle=True
    )

    eps_params = gather_weight_eps(model)
    optimizer = optim.Adam(
        [p for p in eps_params] + [p for p in model.parameters() if p.requires_grad and p not in eps_params],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    teacher = None
    if cfg.use_distill:
        teacher = nn.Sequential(*[
            nn.Sequential(nn.Linear(cfg.in_dim, cfg.out_dim), nn.ReLU())
            for _ in range(cfg.depth)
        ]).to(device)
        teacher.eval()

    total_steps = cfg.epochs * len(loader)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    global_step = 0

    for epoch in range(cfg.epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)

            if global_step < warmup_steps:
                adversarial_deltas = None
            else:
                adversarial_deltas = generate_adversarial_deltas(model, bx, by, loss_fn, cfg)

            logits = forward_model(model, bx, adversarial_deltas)
            loss_task = loss_fn(logits, by)
            loss = loss_task

            if teacher is not None and cfg.use_distill:
                with torch.no_grad():
                    t_logits = forward_model(teacher, bx, None)
                distill = nn.KLDivLoss(reduction="batchmean")(
                    torch.log_softmax(logits / cfg.distill_temp, dim=-1),
                    torch.softmax(t_logits / cfg.distill_temp, dim=-1)
                ) * (cfg.distill_temp ** 2)
                loss = loss + cfg.lambda_distill * distill

            if cfg.epsilon_reg > 0:
                reg = 0.0
                for p in eps_params:
                    eps = 0.5 * torch.tanh(p)
                    reg = reg + eps.pow(2).mean()
                loss = loss + cfg.epsilon_reg * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % cfg.log_interval == 0:
                eps_mean_abs = sum([(0.5 * torch.tanh(p)).abs().mean().item() for p in eps_params]) / len(eps_params)
                print(f"[{global_step}/{total_steps}] "
                      f"loss={loss.item():.4f} task={loss_task.item():.4f} "
                      f"eps_mean_abs={eps_mean_abs:.4f}")

            global_step += 1

    ckpt = {
        "model_state": model.state_dict(),
        "config": vars(cfg)
    }
    torch.save(ckpt, cfg.output_path)
    print(f"Saved checkpoint to {cfg.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("cfg_module", args.config)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    config = cfg_module.get_config()
    train(config)
