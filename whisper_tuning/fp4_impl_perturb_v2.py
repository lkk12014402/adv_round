import torch
import torch.nn as nn

# ================== 原始常量（保持不变） ==================
E2M1_max = 6.0
E2M1_values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
E2M1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])

# ================== 原始函数（未改动，保持兼容） ==================
def quantize_mxfp4(x: torch.Tensor, block_size: int | None = 32) -> tuple:
    """Original MXFP4 quantization (unchanged)."""
    def cast_fp4(x):
        sign = torch.sign(x)  # sign(0)=0 保持原逻辑
        sign_bit = (2 - sign) // 2  # 会产生“负零”编码的一种写法，保持原功能性
        ord_ = torch.sum(
            (x.abs().unsqueeze(-1) - E2M1_bounds.to(x.device)) > 0, dim=-1
        )
        fp4_val = (sign_bit * 0b1000 + ord_).to(torch.uint8)
        return fp4_val

    if block_size is None:
        block_size = 32

    original_shape = x.shape
    original_dtype = x.dtype
    x = x.view(-1, block_size)

    input_amax = x.float().abs().max(dim=-1, keepdim=True).values
    descale = input_amax / E2M1_max
    min_value = torch.tensor(-127.0, device=descale.device)
    e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))

    x = (x / torch.exp2(e8m0_scale)).view(original_shape)
    input_q = cast_fp4(x)
    e8m0_scale = (e8m0_scale + 127).to(torch.uint8)
    return input_q, e8m0_scale.view(input_q.shape[0], -1)


def dequantize_mxfp4(
    x: torch.Tensor,
    e8m0_scale: torch.Tensor,
    block_size: int | None = 32,
    dtype: torch.dtype = None,
    **kwarg
):
    """Original MXFP4 dequantization (unchanged)."""
    x_unfused = x
    sign = 1 - 2 * ((x_unfused & 0b1000) >> 3).to(torch.float32)
    magnitude = (x_unfused & 0b0111).to(torch.long)

    values = torch.tensor(E2M1_values, device=x.device)
    original_shape = magnitude.shape
    x_float = values[magnitude.reshape(-1)].reshape(original_shape)
    x_float = sign.float() * x_float

    x_float = x_float.reshape(-1, block_size)
    scale_factor = torch.exp2(e8m0_scale.float() - 127).reshape(-1, 1)
    x_float = x_float * scale_factor
    return x_float.reshape(original_shape).to(dtype)


# ================== 新增：按 block 共享扰动的模块 ==================
class BlockSharedRounding(nn.Module):
    """
    每个量化 block 一个可训练扰动 δ ∈ [-range_scale, range_scale]（可选）。
    作用：在比较前对 |x_scaled| 加 δ，改变其落入区间的倾向（影响向上/向下取值）。
    并使用区间内部的软概率混合 (sigmoid) + STE 以产生对 δ 的有效梯度。

    模式:
      - hard:      直接使用硬值（δ 只改变比较，但由于比较不可导 δ 无梯度）
      - soft:      使用软值概率混合（训练稳定，但前向与 FP4 硬值不同）
      - soft-ste:  前向硬、反向软（推荐）

    enable_perturbation=False 时退化为最近值逻辑，不分配/使用 δ。
    """
    def __init__(
        self,
        block_size: int = 32,
        beta: float = 10.0,
        range_scale: float = 0.5,
        mode: str = "soft-ste",
        enable_perturbation: bool = True
    ):
        super().__init__()
        self.block_size = block_size
        self.beta = beta
        self.range_scale = range_scale
        self.mode = mode
        self.enable_perturbation = enable_perturbation

        self.delta_raw = None  # 动态按输入块数分配

    def _allocate(self, num_blocks: int, device):
        if not self.enable_perturbation:
            return
        if self.delta_raw is None or self.delta_raw.numel() != num_blocks:
            self.delta_raw = nn.Parameter(torch.zeros(num_blocks, device=device))

    def constrained_delta(self):
        if (not self.enable_perturbation) or self.delta_raw is None:
            return None
        return self.range_scale * torch.tanh(self.delta_raw)  # [-range_scale, range_scale]

    def forward(self, x_scaled: torch.Tensor):
        """
        输入:
          x_scaled: 已完成缩放（除去 per-block exponent 之后）的张量，形状 (..., D)
        输出:
          abs_mix: 扰动后的（硬或软STE）绝对值
          ord_hard: 硬编码的 ordinal（0..7）
        """
        original_shape = x_scaled.shape
        D = original_shape[-1]
        if D % self.block_size != 0:
            raise ValueError(f"Last dim {D} must be divisible by block_size={self.block_size}")

        x_blocks = x_scaled.view(-1, self.block_size)        # (num_blocks, block_size)
        num_blocks = x_blocks.shape[0]
        self._allocate(num_blocks, x_blocks.device)

        abs_val = x_blocks.abs()                             # (num_blocks, block_size)
        bounds = E2M1_bounds.to(x_blocks.device)
        values = torch.tensor(E2M1_values, device=x_blocks.device)

        if not self.enable_perturbation:
            # 最近值：直接比较 abs_val
            cmp = (abs_val.unsqueeze(-1) > bounds)           # (num_blocks, block_size, 7)
            ord_hard = cmp.sum(dim=-1).clamp(max=7)
            abs_hard = values[ord_hard]
            return abs_hard.view(original_shape), ord_hard.view(original_shape)

        # 加扰动：abs_adj = abs_val + δ_block
        delta = self.constrained_delta().view(-1, 1)         # (num_blocks,1)
        abs_adj = abs_val + delta

        cmp = (abs_adj.unsqueeze(-1) > bounds)
        ord_hard = cmp.sum(dim=-1).clamp(max=7)
        abs_hard = values[ord_hard]

        if self.mode == "hard":
            # 纯硬前向（δ 不可导），仅用于最终部署或分析
            return abs_hard.view(original_shape), ord_hard.view(original_shape)

        # 软概率混合 (针对区间内部做向上/向下倾向调节)
        abs_soft = torch.empty_like(abs_hard)
        k = ord_hard
        mask_low = (k == 0)
        mask_high = (k == 7)
        abs_soft[mask_low] = values[0]
        abs_soft[mask_high] = values[-1]

        mid_mask = (~mask_low) & (~mask_high)
        if mid_mask.any():
            k_mid = k[mid_mask]                                  # (M,)
            v_dn = values[(k_mid - 1).clamp(min=0)]
            v_up = values[k_mid]
            # 区间中心 c_k = 0.5*(B_{k-1}+B_k)
            c_k = 0.5 * (bounds[k_mid - 1] + bounds[k_mid])
            # 使用扰动后的绝对值 abs_adj
            a_mid_adj = abs_adj.view(-1)[mid_mask.view(-1)]
            dist = a_mid_adj - c_k
            logits = self.beta * dist
            prob_up = torch.sigmoid(logits)
            abs_soft_mid = prob_up * v_up + (1 - prob_up) * v_dn
            abs_soft.view(-1)[mid_mask.view(-1)] = abs_soft_mid

        if self.mode == "soft":
            abs_mix = abs_soft
        else:  # soft-ste
            abs_mix = abs_soft + (abs_hard - abs_soft).detach()

        return abs_mix.view(original_shape), ord_hard.view(original_shape)


# ================== 新增：带可选扰动的量化函数 ==================
def quantize_mxfp4_perturb(
    x: torch.Tensor,
    block_size: int | None = 32,
    rounding_module: BlockSharedRounding | None = None,
    enable_perturbation: bool = True,
    mode: str = "soft-ste",
    beta: float = 10.0,
    range_scale: float = 0.5,
    return_pseudo: bool = True,
    detach_codes: bool = True
):
    """
    MXFP4 量化（保持原逻辑） + 可选每 block 共享扰动。

    参数:
      x:                输入张量
      block_size:       与原逻辑一致
      rounding_module:  传入已存在的模块可复用参数；None 则新建
      enable_perturbation: 是否启用扰动（False 时与原 quantize_mxfp4 等价）
      mode:             'hard' | 'soft' | 'soft-ste'
      beta:             软概率斜率（大→更接近硬）
      range_scale:      δ 范围限制（[-range_scale, range_scale]）
      return_pseudo:    True 返回伪量化浮点（训练使用）
      detach_codes:     是否对返回的 codes 做 .detach()（防止梯度链向下游误传播）

    返回:
      pseudo_float:     伪量化浮点（若 return_pseudo=True，否则返回 None）
      codes:            uint8 编码（与原格式一致）
      scales:           uint8 per-block exponent 缩放（与原函数形状一致）
      rounding_module:  包含可训练 δ 的模块（便于优化器绑定）
      ord_hard:         硬 ordinal（调试统计用）
    """
    if block_size is None:
        block_size = 32

    original_shape = x.shape
    x_dtype = x.dtype
    x_blocks = x.view(-1, block_size)

    # 1. 计算 per-block scale (与原保持一致)
    input_amax = x_blocks.float().abs().max(dim=-1, keepdim=True).values
    descale = input_amax / E2M1_max
    min_value = torch.tensor(-127.0, device=descale.device)
    e8m0_scale = torch.ceil(torch.maximum(torch.log2(descale), min_value))  # (num_blocks,1)
    scale_factor = torch.exp2(e8m0_scale)                                   # (num_blocks,1)

    # 2. 缩放
    x_scaled = (x_blocks / scale_factor).view(original_shape)

    # 3. 准备/应用扰动模块
    if rounding_module is None:
        rounding_module = BlockSharedRounding(
            block_size=block_size,
            beta=beta,
            range_scale=range_scale,
            mode=mode,
            enable_perturbation=enable_perturbation
        )
    else:
        # 同步配置（可以动态切换）
        rounding_module.block_size = block_size
        rounding_module.beta = beta
        rounding_module.range_scale = range_scale
        rounding_module.mode = mode
        rounding_module.enable_perturbation = enable_perturbation

    abs_mix, ord_hard = rounding_module(x_scaled)

    # 4. 组合符号 + 形成硬编码 (功能保持)
    sign = torch.sign(x_scaled)  # 与原保持 sign(0)=0
    sign_bit = (2 - sign) // 2
    # ord_hard 已是 0..7

    codes = (sign_bit.to(torch.uint8) * 0b1000 + ord_hard.to(torch.uint8))

    # 5. 反缩放构造伪量化浮点（仅训练使用，不破坏原硬编码）
    pseudo_float = None
    if return_pseudo:
        # 在 abs_mix 中已经是绝对值（硬或软融合）
        mix_val = torch.where(sign >= 0, abs_mix, -abs_mix)
        # 反缩放
        mix_blocks = mix_val.view(-1, block_size)
        restored = (mix_blocks * scale_factor).view(original_shape)
        pseudo_float = restored.to(x_dtype)

    # 6. 整理 scale 编码（保持与原 quantize_mxfp4 返回格式一致）
    e8m0_scale_uint8 = (e8m0_scale + 127).to(torch.uint8)  # (num_blocks,1)
    # 保持原 reshape 逻辑：view(input_q.shape[0], -1)
    # 注意：这里 codes 与原 input_q 形状一致（original_shape 展平后第一维）
    scales = e8m0_scale_uint8.view(codes.shape[0], -1)

    if detach_codes:
        codes = codes.detach()

    return pseudo_float, codes, scales, rounding_module, ord_hard




# ========= 分析函数：统计向上/向下取值差异 =========
def analyze_rounding_diff(
    x: torch.Tensor,
    block_size: int = 32,
    rounding_module: BlockSharedRounding | None = None,
    beta: float = 8.0,
    range_scale: float = 0.5,
    mode: str = "soft-ste",
    enable_perturbation: bool = True
):
    """
    比较同一张量在无扰动与有扰动情况下的 ordinal 差异，统计向上/向下取值的次数与比例。

    返回 dict:
      total: 元素总数
      unchanged / upward / downward: 计数与百分比
      by_magnitude: 基线 ordinal 分组后向上/向下/不变的明细
      by_block: 每 block 的 upward/downward/unchanged 数
      delta_raw_stats: (mean, std, min, max) （若启用扰动）
    """
    device = x.device
    # 基线（无扰动）
    _, base_codes, base_scales, base_module_tmp, ord_base = quantize_mxfp4_perturb(
        x, block_size=block_size,
        rounding_module=rounding_module,
        enable_perturbation=False,  # 禁用扰动，得到最近值
        mode="hard", return_pseudo=False
    )

    # 扰动（启用）
    pseudo, pert_codes, pert_scales, used_module, ord_pert = quantize_mxfp4_perturb(
        x, block_size=block_size,
        rounding_module=rounding_module,
        enable_perturbation=enable_perturbation,
        mode=mode, beta=beta, range_scale=range_scale,
        return_pseudo=True
    )

    ord_base = ord_base.to(torch.int32)
    ord_pert = ord_pert.to(torch.int32)
    diff = ord_pert - ord_base

    upward_mask = diff > 0
    downward_mask = diff < 0
    unchanged_mask = diff == 0

    total = diff.numel()
    upward = upward_mask.sum().item()
    downward = downward_mask.sum().item()
    unchanged = unchanged_mask.sum().item()

    def pct(v): return v * 100.0 / total

    # 按基线 ordinal 分组统计
    by_mag = {}
    for k in range(8):  # ord 0..7
        mask_k = (ord_base == k)
        count_k = mask_k.sum().item()
        if count_k == 0:
            by_mag[k] = {"count": 0, "up": 0, "down": 0, "unchanged": 0,
                         "up_pct": 0.0, "down_pct": 0.0, "unchanged_pct": 0.0}
            continue
        up_k = (upward_mask & mask_k).sum().item()
        down_k = (downward_mask & mask_k).sum().item()
        unchg_k = (unchanged_mask & mask_k).sum().item()
        by_mag[k] = {
            "count": count_k,
            "up": up_k,
            "down": down_k,
            "unchanged": unchg_k,
            "up_pct": up_k * 100.0 / count_k,
            "down_pct": down_k * 100.0 / count_k,
            "unchanged_pct": unchg_k * 100.0 / count_k
        }

    # 按 block 统计（把最后一维切块）
    D = x.shape[-1]
    if D % block_size != 0:
        raise ValueError(f"Last dim {D} must be divisible by block_size={block_size}")
    num_blocks_total = x.numel() // block_size
    ord_base_blocks = ord_base.view(-1, block_size)
    ord_pert_blocks = ord_pert.view(-1, block_size)
    diff_blocks = ord_pert_blocks - ord_base_blocks

    by_block = []
    for b in range(num_blocks_total):
        db = diff_blocks[b]
        up_b = (db > 0).sum().item()
        down_b = (db < 0).sum().item()
        unchg_b = (db == 0).sum().item()
        by_block.append({
            "block_index": b,
            "up": up_b,
            "down": down_b,
            "unchanged": unchg_b,
            "up_pct": up_b * 100.0 / block_size,
            "down_pct": down_b * 100.0 / block_size,
            "unchanged_pct": unchg_b * 100.0 / block_size
        })

    # δ 统计
    delta_stats = None
    if used_module is not None and used_module.delta_raw is not None and enable_perturbation:
        delta = used_module.constrained_delta().detach()
        delta_stats = {
            "mean": delta.mean().item(),
            "std": delta.std().item(),
            "min": delta.min().item(),
            "max": delta.max().item()
        }

    return {
        "total": total,
        "upward": upward,
        "downward": downward,
        "unchanged": unchanged,
        "upward_pct": pct(upward),
        "downward_pct": pct(downward),
        "unchanged_pct": pct(unchanged),
        "by_magnitude": by_mag,
        "by_block": by_block,
        "delta_raw_stats": delta_stats,
        "ord_base": ord_base,
        "ord_pert": ord_pert,
        "pseudo_float": pseudo
    }

# 构造“边界附近”数据
def synth_weights(num_rows=128, num_cols=256, block_size=32, noise_range=0.2):
    midpoints = torch.tensor([0.25,0.75,1.25,1.75,2.5,3.5,5.0])
    # 随机选中点
    choice = torch.randint(0, len(midpoints), (num_rows, num_cols))
    mids = midpoints[choice]
    noise = (torch.rand_like(mids) - 0.5) * (2*noise_range)
    base = mids + noise
    # 随机符号
    signs = torch.where(torch.rand_like(base) > 0.5, 1.0, -1.0)
    W = (base * signs).clamp(-5.5, 5.5)  # 避免超过最大中点太多
    # 再加少量远离边界的值模拟真实分布
    mask_far = torch.rand_like(W) < 0.1
    W[mask_far] = W[mask_far] * 2.0
    return W

def ordinal_to_values(ord_tensor: torch.Tensor):
    return E2M1_VALUES.to(ord_tensor.device)[ord_tensor]

def analyze_change(ord_before, ord_after):
    diff = ord_after - ord_before
    up = (diff>0).sum().item()
    down = (diff<0).sum().item()
    same = (diff==0).sum().item()
    total = diff.numel()
    return {
        "up": up, "down": down, "same": same,
        "up_pct": up*100/total, "down_pct": down*100/total, "same_pct": same*100/total
    }

def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = synth_weights().to(device)

    # 基线（无扰动、最近值）
    pseudo_base, ord_base, mod = quantize_mxfp4_perturb(
        W, block_size=32, enable_perturbation=False, mode="hard", return_pseudo=True
    )
    base_mse = (pseudo_base - W).pow(2).mean().item()
    print(f"[Baseline] MSE={base_mse:.6f}")

    # 预热阶段：soft (梯度稳定、beta 小)
    epochs_soft = 150
    optimizer = torch.optim.Adam([mod.delta_raw], lr=5e-3)
    for step in range(1, epochs_soft+1):
        pseudo, ord_soft, _ = quantize_mxfp4_perturb(
            W, block_size=32, rounding_module=mod,
            enable_perturbation=True, mode="soft",
            beta=4.0, range_scale=0.5, return_pseudo=True
        )
        loss = (pseudo - W).pow(2).mean() + 1e-4 * (mod.delta().pow(2).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 30 == 0 or step == 1:
            mse_now = (pseudo - W).pow(2).mean().item()
            print(f"[Soft {step}] MSE={mse_now:.6f} delta_mean={mod.delta().mean().item():.4f}")

    # 切换到 soft-ste（硬前向语义 + 软梯度），升高 beta
    epochs_ste = 150
    for step in range(1, epochs_ste+1):
        pseudo, ord_ste, _ = quantize_mxfp4_perturb(
            W, block_size=32, rounding_module=mod,
            enable_perturbation=True, mode="soft-ste",
            beta=7.0, range_scale=0.5, return_pseudo=True
        )
        loss = (pseudo - W).pow(2).mean() + 1e-4 * (mod.delta().pow(2).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 30 == 0 or step == 1:
            mse_now = (pseudo - W).pow(2).mean().item()
            print(f"[STE {step}] MSE={mse_now:.6f} delta_mean={mod.delta().mean().item():.4f}")

    # 最终统计
    final_pseudo, final_ord, _ = quantize_mxfp4_perturb(
        W, block_size=32, rounding_module=mod,
        enable_perturbation=True, mode="hard",
        beta=7.0, range_scale=0.5, return_pseudo=True
    )
    final_mse = (final_pseudo - W).pow(2).mean().item()
    change_stats = analyze_change(ord_base, final_ord)
    print(f"\n[Final] Baseline MSE={base_mse:.6f}  Final MSE={final_mse:.6f}  Δ={base_mse-final_mse:.6f}")
    print("Ordinal change:", change_stats)
    print("Delta stats: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
        mod.delta().mean().item(), mod.delta().std().item(),
        mod.delta().min().item(), mod.delta().max().item()
    ))

if __name__ == "__main__":
    main()
