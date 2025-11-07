import torch
import torch.nn as nn
from fp4_impl import E2M1_bounds, E2M1_values

class FP4RoundingPerturb(nn.Module):
    def __init__(self, block_size=32, tau=0.1, init_delta=0.0, per_boundary=False):
        super().__init__()
        self.block_size = block_size
        self.tau = tau
        self.per_boundary = per_boundary

        if per_boundary:
            raw = torch.zeros(len(E2M1_bounds))
            self.delta_raw = nn.Parameter(raw)
        else:
            self.delta_raw = nn.Parameter(torch.tensor(float(init_delta)))

        self.register_buffer("bounds_base", E2M1_bounds.clone().float())
        self.register_buffer("values_table", torch.tensor(E2M1_values).float())

    def _compute_bounds(self):
        if self.per_boundary:
            delta = 0.5 * torch.tanh(self.delta_raw)          # 每个边界一个扰动
            bounds = self.bounds_base + delta
        else:
            delta = 0.5 * torch.tanh(self.delta_raw)          # 单一全局扰动
            bounds = self.bounds_base + delta
        return bounds

    def _hard_ordinal(self, x_abs, shifted_bounds):
        cmp = (x_abs.unsqueeze(-1) > shifted_bounds)         # (..., B)
        ord_ = cmp.sum(dim=-1)
        return ord_.to(torch.long)

    def _soft_abs_value(self, x_abs, shifted_bounds):
        tau = self.tau
        p = torch.sigmoid((x_abs.unsqueeze(-1) - shifted_bounds) / tau)  # (..., B)
        w0 = 1 - p[..., 0]
        mids = p[..., :-1] - p[..., 1:]
        w_last = p[..., -1]
        weights = torch.cat([w0.unsqueeze(-1), mids, w_last.unsqueeze(-1)], dim=-1)  # (..., 8)
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        values = self.values_table.to(x_abs.device)
        abs_soft = (weights * values).sum(dim=-1)
        return abs_soft, weights

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        if original_shape[-1] % self.block_size != 0:
            raise ValueError(f"Last dim {original_shape[-1]} not divisible by block_size={self.block_size}")

        x_blocks = x.view(-1, self.block_size)
        amax = x_blocks.float().abs().max(dim=-1, keepdim=True).values
        descale = amax / 6.0
        min_value = torch.tensor(-127.0, device=x.device)
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))
        scale_factor = torch.exp2(e8m0_scale)

        x_scaled = (x_blocks / scale_factor).view(original_shape)

        shifted_bounds = self._compute_bounds().to(x.device)

        x_abs = x_scaled.abs()
        ord_hard = self._hard_ordinal(x_abs, shifted_bounds)
        sign_bit = (x_scaled < 0).to(torch.uint8)
        codes = ((sign_bit << 3) | ord_hard.to(torch.uint8))

        abs_soft, weights = self._soft_abs_value(x_abs, shifted_bounds)
        representable = self.values_table.to(x.device)[ord_hard]
        abs_hard = representable

        # 修正：使用软路径传梯度
        # abs_mix = abs_hard + (abs_soft - abs_hard).detach()  # 原错误写法（不传递 delta 的梯度）
        abs_mix = abs_soft + (abs_hard - abs_soft).detach()    # 正确写法
        y_scaled = torch.where(x_scaled >= 0, abs_mix, -abs_mix)

        y_blocks = y_scaled.view(-1, self.block_size)
        y = (y_blocks * scale_factor).view(original_shape)

        e8m0_scale_uint8 = (e8m0_scale + 127).to(torch.uint8).squeeze(-1)
        leading = original_shape[:-1]
        blocks_per_slice = original_shape[-1] // self.block_size
        scales = e8m0_scale_uint8.view(*leading, blocks_per_slice)

        return y, codes, scales, shifted_bounds, weights


def example_usage(device="cuda" if torch.cuda.is_available() else "cpu"):
    torch.manual_seed(0)
    quant = FP4RoundingPerturb(block_size=8, tau=0.15, init_delta=0.0).to(device)
    x = torch.randn(4, 8, device=device, requires_grad=True)
    y, codes, scales, shifted_bounds, weights = quant(x)
    # 设一个简单损失：让量化值逼近原值（只是演示）
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("Loss:", loss.item())
    print("delta_raw grad:", quant.delta_raw.grad)        # 现在应非 None
    print("x.grad mean:", x.grad.abs().mean().item())
    print("Shifted bounds:", shifted_bounds)
    print("Codes sample:", codes[0])

if __name__ == "__main__":
    example_usage()
