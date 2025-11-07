import torch
import pytest

# 假设以下导入路径与工程结构保持一致
from mxfp4_quantizer import E2M1WeightQuantizer  # 若你文件名不同请调整
from quantize_whisper import QuantizedLinearWhisper

def build_quantized_linear(in_features=64, out_features=32, init_mode="mixed", device="cuda"):
    base_linear = torch.nn.Linear(in_features, out_features, bias=True)
    qlayer = QuantizedLinearWhisper(
        linear=base_linear,
        layer_name="test_layer",
        layer_max_w=6.0,
        layer_max_act=6.0,
        block_size_w=32,
        block_size_act=32,
        act_quant=False,      # 单测先关闭激活动量化，聚焦权重 ε
        init_mode=init_mode
    ).to(device)
    return qlayer

def _forward_once(layer, x):
    out, meta = layer(x)  # QuantizedLinearWhisper forward 返回 (out, {"weight": meta_w})
    return out, meta["weight"]

def test_quantized_linear_gradient(device):
    torch.manual_seed(42)
    layer = build_quantized_linear(device=device)

    # 输入需要梯度以检查是否能回传
    x = torch.randn(8, layer.weight_fp.shape[1], device=device, requires_grad=True)
    # 构造一个随机回归目标
    target = torch.randn(8, layer.weight_fp.shape[0], device=device)

    out, meta_w = _forward_once(layer, x)
    loss = torch.nn.functional.mse_loss(out, target)

    # 反向传播
    loss.backward()

    print("requires_grad:", layer.wq.eps_param.requires_grad)
    print("is leaf:", layer.wq.eps_param.is_leaf)
    print("grad is None:", layer.wq.eps_param.grad is None)
    print("storage addr:", layer.wq.eps_param.data_ptr())
    exit()

    # 诊断信息
    eps_param = layer.wq.eps_param
    eps_eff = 0.5 * torch.tanh(eps_param.detach())
    grad_eps = eps_param.grad
    grad_bias = layer.bias.grad if layer.bias is not None else None
    grad_x = x.grad

    print(f"[DIAG] loss={loss.item():.6f}")
    print(f"[DIAG] eps mean|ε|={eps_eff.abs().mean().item():.6f} max|ε|={eps_eff.abs().max().item():.6f}")
    print(f"[DIAG] eps_param.grad is None? {grad_eps is None}")
    if grad_eps is not None:
        print(f"[DIAG] eps_param.grad norm={grad_eps.norm().item():.6e} max={grad_eps.abs().max().item():.6e}")
    print(f"[DIAG] bias.grad is None? {grad_bias is None}")
    if grad_bias is not None:
        print(f"[DIAG] bias.grad norm={grad_bias.norm().item():.6e}")
    print(f"[DIAG] x.grad is None? {grad_x is None}")
    if grad_x is not None:
        print(f"[DIAG] x.grad norm={grad_x.norm().item():.6e}")

    # 断言：需要梯度的三个对象都应该拿到非 None 梯度
    assert grad_eps is not None, "eps_param.grad 为 None，可能未加入可微代理或未被 optimizer 管理。"
    assert grad_bias is not None, "bias.grad 为 None，检查 bias 是否 requires_grad 或被错误替换。"
    assert grad_x is not None, "输入梯度缺失，检查 forward 是否使用了输入或被 detach。"

    # 如果是硬离散未引入 STE，这里梯度可能全 0；给出明确失败提示
    if grad_eps.norm().item() == 0.0:
        pytest.fail("eps_param.grad 为 0。当前量化可能是纯硬离散映射，需要引入 STE 或连续 proxy。")

def run_smoke_test():
    """
    不依赖 pytest 的简单运行入口：
    python tests/test_quantized_linear_whisper.py
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = build_quantized_linear(device=device)
    x = torch.randn(4, layer.weight_fp.shape[1], device=device, requires_grad=True)
    target = torch.randn(4, layer.weight_fp.shape[0], device=device)
    out, meta_w = _forward_once(layer, x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    eps_param = layer.wq.eps_param
    print("[Smoke] loss:", loss.item())
    print("[Smoke] eps grad norm:", None if eps_param.grad is None else eps_param.grad.norm().item())

if __name__ == "__main__":
    run_smoke_test()
    test_quantized_linear_gradient("cuda")
