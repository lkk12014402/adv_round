import torch
import torch.nn as nn
from typing import Tuple, Optional

# =========================
# 常量与原始集合保持一致
# =========================
E2M1_MAX = 6.0
E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

# =========================
# 基础函数：与原实现保持一致
# =========================
def _cast_fp4(x_norm: torch.Tensor) -> torch.Tensor:
    """
    输入为已经做过块尺度归一化后的张量 x_norm。
    输出 uint8 编码：bit3 为符号 (1=负)，bit0-2 为 ord (0..7)。
    """
    sign = torch.sign(x_norm)
    sign_bit = (2 - sign) // 2                 # +1 -> 0, -1 -> 1
    # ord 分类：依据提供的阈值
    ord_ = torch.sum(
        (x_norm.abs().unsqueeze(-1) - E2M1_BOUNDS.to(x_norm.device)) > 0,
        dim=-1
    )
    fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
    return fp4_val

def _fuse_uint4_to_uint8(x_uint4: torch.Tensor) -> torch.Tensor:
    """
    将最后一维的 uint4 两两打包成 uint8（按您原逻辑）。
    不强制使用；训练阶段一般保留原始 uint4 方便调试。
    """
    left_side = x_uint4[..., 0::2]
    right_side = x_uint4[..., 1::2]
    new_data = right_side.clone() << 4
    new_data[..., : left_side.shape[-1]] += left_side
    return new_data

def _unfuse_uint8_to_uint4(x_uint8: torch.Tensor) -> torch.Tensor:
    left_side = x_uint8 & 0x0F
    right_side = (x_uint8 >> 4) & 0x0F
    shape = list(x_uint8.shape)
    shape[-1] = shape[-1] * 2
    out = torch.zeros(shape, dtype=torch.uint8, device=x_uint8.device)
    out[..., 0::2] = left_side
    out[..., 1::2] = right_side
    return out

# =========================
# 对原始函数的封装（无 ε）
# =========================
def quantize_mxfp4(x: torch.Tensor,
                   block_size: int = 32,
                   fuse_nibbles: bool = False,
                   pad_last_dim: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    与原实现保持结构：按最后一维分块计算尺度 + 编码。
    返回 (encoded_uint8, e8m0_scale_uint8)
    如果 fuse_nibbles=True，将每两个 uint4 打包为一个 uint8（输出形状最后一维减半）。
    """
    original_shape = x.shape
    last_dim = original_shape[-1]

    # 可选 padding（使最后一维可整除 block_size）
    if pad_last_dim and (last_dim % block_size != 0):
        pad = block_size - (last_dim % block_size)
        x = torch.nn.functional.pad(x, (0, pad))
        last_dim = last_dim + pad

    x_blocks = x.view(-1, block_size)
    input_amax = x_blocks.float().abs().max(dim=-1, keepdim=True).values
    descale = input_amax / E2M1_MAX
    min_value = torch.tensor(-127.0, device=x.device)
    e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))
    x_norm = x_blocks / torch.exp2(e8m0_scale)  # 归一化

    encoded = _cast_fp4(x_norm).view(x.shape)
    if fuse_nibbles:
        encoded = _fuse_uint4_to_uint8(encoded)

    e8m0_scale_uint8 = (e8m0_scale + 127).to(torch.uint8)
    # scale 展回块维度：num_blocks x 1
    return encoded.view(encoded.shape), e8m0_scale_uint8.view(-1, 1)

def dequantize_mxfp4(encoded: torch.Tensor,
                      e8m0_scale_uint8: torch.Tensor,
                      block_size: int = 32,
                      dtype: Optional[torch.dtype] = None,
                      fused_nibbles: bool = False,
                      original_last_dim: Optional[int] = None) -> torch.Tensor:
    """
    与原实现保持一致：encoded 为 uint4(或 fuse 后的 uint8)，e8m0_scale_uint8 为块尺度（加 127）。
    """
    if dtype is None:
        dtype = torch.float32

    if fused_nibbles:
        encoded = _unfuse_uint8_to_uint4(encoded)

    enc_shape = encoded.shape
    last_dim = enc_shape[-1]
    if last_dim % block_size != 0:
        raise ValueError(f"Encoded last dim {last_dim} not divisible by block_size {block_size}.")

    x_blocks = encoded.view(-1, block_size)

    sign = 1 - 2 * ((x_blocks & 0b1000) >> 3).to(torch.float32)         # +1/-1
    magnitude = (x_blocks & 0b0111).to(torch.long)
    values_table = torch.tensor(E2M1_VALUES, device=encoded.device)

    x_float = values_table[magnitude.view(-1)].view_as(magnitude).float()
    x_float = sign * x_float

    # 应用块尺度
    scale_factor = torch.exp2(e8m0_scale_uint8.float() - 127)  # num_blocks x 1
    x_float = x_float.view(-1, block_size) * scale_factor

    deq = x_float.view(enc_shape).to(dtype)

    # 如果之前做过 padding，这里可以裁剪回 original_last_dim
    if original_last_dim is not None and original_last_dim < enc_shape[-1]:
        deq = deq[..., :original_last_dim]

    return deq

# =========================
# 带 ε 的权重量化器（训练用）
# =========================
class E2M1WeightQuantizer(nn.Module):
    """
    与原量化逻辑保持一致，只在归一化后绝对值进入阈值比较前加入块偏移 ε。
    ε 映射：ε = 0.5 * tanh(p) ∈ (-0.5, 0.5)
    不加入 adversarial delta。
    """
    def __init__(self,
                 block_size: int = 32,
                 layer_max: float = 6.0,
                 init_mode: str = "mixed",        # "zero" | "uniform" | "threshold_center" | "mixed"
                 init_uniform_range: float = 0.05,
                 requires_grad: bool = True):
        super().__init__()
        self.block_size = block_size
        self.layer_max = layer_max
        self.init_mode = init_mode
        self.init_uniform_range = init_uniform_range
        self.requires_grad = requires_grad
        self.eps_param: Optional[nn.Parameter] = None
        self._initialized = False
        self.register_buffer("bounds", E2M1_BOUNDS.clone())
        self.register_buffer("values_table", torch.tensor(E2M1_VALUES))
        extended = torch.cat([
            torch.tensor([0.0], dtype=self.bounds.dtype),
            self.bounds,
            torch.tensor([self.layer_max], dtype=self.bounds.dtype)
        ])
        self.register_buffer("extended_bounds", extended)

    def _ensure_buffers_on(self, device: torch.device):
        # 同步所有 buffer 到指定设备
        if self.bounds.device != device:
            self.bounds = self.bounds.to(device)
        if self.values_table.device != device:
            self.values_table = self.values_table.to(device)
        if self.extended_bounds.device != device:
            self.extended_bounds = self.extended_bounds.to(device)

    def _compute_block_mean_norm(self, weight_flat):
        out_f, in_f = weight_flat.shape
        w_blocks = weight_flat.view(out_f, in_f // self.block_size, self.block_size)
        amax = w_blocks.abs().max(dim=-1, keepdim=True).values + 1e-12
        norm = w_blocks.abs() / amax
        mean_norm = norm.mean(dim=-1)
        return mean_norm.view(-1)

    def _threshold_center_init(self, mean_norm: torch.Tensor) -> torch.Tensor:
        # 确保 extended_bounds 与 mean_norm 在同一设备
        ext = self.extended_bounds
        if ext.device != mean_norm.device:
            ext = ext.to(mean_norm.device)
        # bucketize
        idx = torch.bucketize(mean_norm, ext, right=False) - 1
        idx.clamp_(0, ext.numel() - 2)
        left = ext[idx]
        right = ext[idx + 1]
        center = (left + right) * 0.5
        eps_block = center - mean_norm
        eps_block.clamp_(-0.3, 0.3)
        return eps_block

    def _mixed_init(self, mean_norm: torch.Tensor) -> torch.Tensor:
        base = self._threshold_center_init(mean_norm)
        noise = torch.empty_like(base).uniform_(-0.02, 0.02)
        base.add_(noise).clamp_(-0.3, 0.3)
        return base

    def _init_eps(self, weight_flat: torch.Tensor):
        out_f, in_f = weight_flat.shape
        assert in_f % self.block_size == 0
        num_blocks = out_f * (in_f // self.block_size)
        if self.init_mode == "zero":
            eps0 = torch.zeros(num_blocks, device=weight_flat.device)
        elif self.init_mode == "uniform":
            eps0 = (torch.rand(num_blocks, device=weight_flat.device) * 2 - 1) * self.init_uniform_range
        elif self.init_mode == "threshold_center":
            mean_norm = self._compute_block_mean_norm(weight_flat)
            eps0 = self._threshold_center_init(mean_norm)
        elif self.init_mode == "mixed":
            mean_norm = self._compute_block_mean_norm(weight_flat)
            eps0 = self._mixed_init(mean_norm)
        else:
            raise ValueError(f"Unknown init_mode {self.init_mode}")

        eps0 = torch.clamp(eps0, -0.49, 0.49)
        p0 = torch.atanh(eps0 / 0.5)
        p0.requires_grad_(self.requires_grad)
        self.eps_param = nn.Parameter(p0, requires_grad=self.requires_grad)
        self._initialized = True

    def forward(self, weight_fp: torch.Tensor,
                fuse_nibbles: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        返回 (dequant_weight, meta)
        meta: {
            'encoded': uint8编码,
            'scale': e8m0_scale_uint8 (num_blocks x 1 reshape),
            'epsilon': ε_eff (detach)
        }
        """
        out_f, in_f = weight_fp.shape
        if in_f % self.block_size != 0:
            raise ValueError(f"in_features={in_f} 不能被 block_size={self.block_size} 整除。可考虑 padding 或调整 block_size。")

        if not self._initialized:
            self._init_eps(weight_fp)

        # 计算块尺度（与原实现一致）
        w_blocks = weight_fp.view(out_f, in_f // self.block_size, self.block_size)
        amax = w_blocks.abs().max(dim=-1, keepdim=True).values
        descale = amax / self.layer_max
        min_value = torch.tensor(-127.0, device=weight_fp.device)
        e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale.float()), min_value))  # [out_f, num_block_per_out, 1]

        scale = torch.exp2(e8m0_scale)
        w_norm = w_blocks / scale

        # 生成 ε
        eps_base = 0.5 * torch.tanh(self.eps_param)  # [num_blocks]
        num_blocks = w_blocks.shape[0] * w_blocks.shape[1]
        eps_blocks = eps_base.view(num_blocks, 1)

        abs_norm = w_norm.abs().view(num_blocks, self.block_size)
        abs_shifted = torch.clamp(abs_norm + eps_blocks, min=0.0)

        # 使用与原 cast_fp4 相同的逻辑分类（构造一个临时张量组合 sign）
        sign = torch.sign(w_norm).view(num_blocks, self.block_size)
        sign_bit = (2 - sign) // 2
        ord_ = torch.sum(
            (abs_shifted.unsqueeze(-1) - self.bounds.to(abs_shifted.device)) > 0,
            dim=-1
        )
        encoded = (sign_bit * 0b1000 + ord_).to(torch.uint8)

        if fuse_nibbles:
            encoded = _fuse_uint4_to_uint8(encoded.view(out_f, in_f))

        # 解码成浮点以供训练（严格按照 dequant 流程）
        # magnitude -> values
        ord_clean = ord_.view(-1)  # [num_blocks * block_size]
        values_table = torch.tensor(E2M1_VALUES, device=weight_fp.device)
        mantissa = values_table[ord_clean].view(num_blocks, self.block_size).float()
        signed_mantissa = mantissa * (1 - 2 * sign_bit.float())
        deq_blocks = signed_mantissa * scale.view(num_blocks, 1)
        deq_weight = deq_blocks.view(out_f, in_f)

        e8m0_scale_uint8 = (e8m0_scale.view(num_blocks, 1) + 127).to(torch.uint8)

        meta = {
            "encoded": encoded.view(out_f, -1) if not fuse_nibbles else encoded,
            "scale": e8m0_scale_uint8.view(out_f, in_f // self.block_size, 1),
            "epsilon": eps_base.detach()
        }
        return deq_weight, meta

# =========================
# 激活动量化器（与原逻辑无 ε，可选）
# =========================
class E2M1ActivationQuantizer(nn.Module):
    """
    动态按块计算尺度并编码（与原 quantize_mxfp4 的流程保持一致）。
    不引入 ε。
    """
    def __init__(self,
                 block_size: int = 32,
                 layer_max: float = 6.0,
                 use_ema: bool = True,
                 ema_momentum: float = 0.95,
                 recompute_interval: int = 1):
        super().__init__()
        self.block_size = block_size
        self.layer_max = layer_max
        self.use_ema = use_ema
        self.ema_momentum = ema_momentum
        self.recompute_interval = recompute_interval
        self.register_buffer("running_scale", None)
        self._step = 0
        self._last_scale_uint8 = None
        self.use_ema = False

    def forward(self, x: torch.Tensor, quantize: bool = True, fuse_nibbles: bool = False):
        if not quantize:
            return x, None
        shape = x.shape
        last_dim = shape[-1]
        if last_dim % self.block_size != 0:
            raise ValueError(f"激活最后维 {last_dim} 不能被 block_size={self.block_size} 整除。")

        x_blocks = x.view(-1, self.block_size)
        recompute = (self._step % self.recompute_interval == 0) or (self._last_scale_uint8 is None)
        if recompute:
            amax = x_blocks.abs().max(dim=-1, keepdim=True).values
            descale = amax / self.layer_max
            min_value = torch.tensor(-127.0, device=x.device)
            e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale.float()), min_value))
            if self.use_ema:
                if self.running_scale is None:
                    self.running_scale = e8m0_scale
                else:
                    self.running_scale = self.ema_momentum * self.running_scale + (1 - self.ema_momentum) * e8m0_scale
                used_scale = self.running_scale
            else:
                used_scale = e8m0_scale
            scale_uint8 = (used_scale + 127).to(torch.uint8)
            self._last_scale_uint8 = scale_uint8
        else:
            scale_uint8 = self._last_scale_uint8

        scale = torch.exp2(scale_uint8.float() - 127).view(-1, 1)
        x_norm = x_blocks / scale
        encoded = _cast_fp4(x_norm).view(shape)
        if fuse_nibbles:
            encoded = _fuse_uint4_to_uint8(encoded)

        meta = {
            "encoded": encoded.detach(),
            "scale": scale_uint8.view(-1, 1)
        }
        self._step += 1
        # 返回解量化值直接继续计算
        # 解量化：复用与权重一致逻辑
        enc_for_deq = encoded
        if fuse_nibbles:
            enc_for_deq = _unfuse_uint8_to_uint4(encoded)
        enc_blocks = enc_for_deq.view(-1, self.block_size)
        sign = 1 - 2 * ((enc_blocks & 0b1000) >> 3).float()
        magnitude = (enc_blocks & 0b0111).long()
        values_table = torch.tensor(E2M1_VALUES, device=x.device)
        mantissa = values_table[magnitude.view(-1)].view_as(magnitude).float()
        x_float = (sign * mantissa).view(-1, self.block_size) * scale
        x_deq = x_float.view(shape)
        return x_deq, meta
