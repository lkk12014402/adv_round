import torch
import torch.nn as nn
import torch.optim as optim
import argparse, json
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from quantize_whisper import apply_quantization_whisper
from epsilon_stats import epsilon_statistics

def gather_eps_params_old(model):
    eps = []
    for m in model.modules():
        if hasattr(m, "wq") and hasattr(m.wq, "eps_param") and m.wq.eps_param is not None:
            eps.append(m.wq.eps_param)
    return eps


def _ensure_eps_initialized_for_module(mod):
    """
    如果模块含有自适应权重量化器 (wq)，并且尚未初始化 eps_param，
    则根据其权重形状调用 _init_eps。
    支持：
      - QuantizedLinearWhisper:   weight_fp shape [out, in]
      - QuantizedConv1dWhisper:  weight_fp shape [out, in, k] -> flatten [out, in*k]
    """
    if not hasattr(mod, "wq"):
        return
    wq = getattr(mod, "wq", None)
    if wq is None:
        return
    # 只有 AdaptiveE2M1WeightQuantizer 才有 _init_eps
    if not hasattr(wq, "_init_eps"):
        return
    # 已经有 eps_param 且 _initialized = True 就不用再初始化
    if getattr(wq, "_initialized", False) and getattr(wq, "eps_param", None) is not None:
        return

    if not hasattr(mod, "weight_fp"):
        # 若包装类里未来改名，可自行扩展
        return

    weight = mod.weight_fp
    if weight is None:
        return

    # 线性层: 2D
    if weight.dim() == 2:
        # 直接初始化
        wq._init_eps(weight)
    # 卷积层（假定 Conv1d 3D： [C_out, C_in, K]）
    elif weight.dim() == 3:
        flattened = weight.view(weight.shape[0], -1)
        wq._init_eps(flattened)
    else:
        # 其它情况按最靠近线性展平
        flattened = weight.view(weight.shape[0], -1)
        wq._init_eps(flattened)



def gather_eps_params(model):
    """
    返回所有量化层可训练的 eps_param 列表。
    在收集前自动确保每个 AdaptiveE2M1WeightQuantizer 已初始化。
    """
    eps_params = []
    for m in model.modules():
        _ensure_eps_initialized_for_module(m)
        if hasattr(m, "wq"):
            wq = getattr(m, "wq")
            if hasattr(wq, "eps_param") and wq.eps_param is not None:
                # 如果 eps_param 未被 requires_grad（意外情况），可强制
                if not wq.eps_param.requires_grad and getattr(wq, "requires_grad", True):
                    wq.eps_param.requires_grad_(True)
                eps_params.append(wq.eps_param)
    return eps_params


def forward_tokens(model, processor, batch, device):
    audio = batch["audio"]["array"]
    sr = batch["audio"]["sampling_rate"]
    text_list = batch["text"]
    inputs = processor(audio, sampling_rate=sr, text=text_list, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return outputs.logits, inputs["labels"].clone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3")
    parser.add_argument("--dataset", type=str, default="MLCommons/peoples_speech")
    parser.add_argument("--adaptive_max", type=str, default="adaptive_max.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--init_mode", type=str, default="mixed")
    parser.add_argument("--output", type=str, default="whisper_quant_epsilon.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model).to(device)

    with open(args.adaptive_max) as f:
        adaptive_conf = json.load(f)

    model = apply_quantization_whisper(model,
                                       adaptive_conf,
                                       block_size_w=32,
                                       block_size_act=32,
                                       act_quant=True,
                                       init_mode=args.init_mode)

    print(model)

    #dataset = load_dataset(args.dataset, "clean", split="train")
    DATASET_SUBSET = "test"
    DATASET_SPLIT = "test"
    # Select number of samples. 512 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048

    # Load dataset and preprocess.
    dataset = load_dataset(
        args.dataset,
        DATASET_SUBSET,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    )
    print(dataset)
    print(dataset[0])

    loss_fn = nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id)

    print(f"============================================gather_eps_params")
    eps_params = gather_eps_params(model)
    print(eps_params[0])
    # 只训练 ε + bias
    train_params = list(eps_params) + [
        p for n, p in model.named_parameters()
        if "bias" in n and p.requires_grad
    ]
    optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        for i, batch in enumerate(dataset):
            model.train()
            logits, labels = forward_tokens(model, processor, batch, device)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            stats = epsilon_statistics(model)
            optimizer.step()

            if i % 10 == 0:
                stats = epsilon_statistics(model)
                mean_abs = sum(s["abs_mean"] for s in stats) / len(stats) if stats else 0.0
                print(f"[Epoch {epoch} Step {i}] loss={loss.item():.4f} mean|ε|={mean_abs:.4f}")

    torch.save({"model_state": model.state_dict(),
                "adaptive_max": adaptive_conf}, args.output)
    print(f"Saved quantized checkpoint to {args.output}")
