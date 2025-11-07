import torch
import torch.nn as nn
from fp4_impl_perturb import quantize_mxfp4_perturb, BlockSharedRounding

class ActivationQuantMXFP4(nn.Module):
    """
    MXFP4 (E2M1) 量化激活张量，带可选 block 共享扰动 δ：
      - 针对输入张量的最后一维分块 (block_size)
      - 使用 quantize_mxfp4_perturb 获取伪量化输出 (pseudo)
      - 仅在训练需要梯度时使用 mode='soft-ste' 或 'soft'
      - 推理/评估可用 'hard'

    参数:
      block_size: 最后维度需要能整除（若不能可选择 pad 后再截断，此处默认报错）
      beta:       sigmoid 斜率
      range_scale: δ 范围 [-range_scale, range_scale]
      mode:       'soft' | 'soft-ste' | 'hard'
      enable_perturbation: 是否启用 δ 扰动
      detach_codes: 激活训练不需要 codes 梯度，保持 True
      share_rounding: 若外部传入 rounding_module，可复用（例如输入与输出共享）
    """
    def __init__(self,
                 block_size: int = 32,
                 beta: float = 4.0,
                 range_scale: float = 0.5,
                 mode: str = "soft-ste",
                 enable_perturbation: bool = True,
                 rounding_module: BlockSharedRounding | None = None,
                 detach_codes: bool = True):
        super().__init__()
        self.block_size = block_size
        self.beta = beta
        self.range_scale = range_scale
        self.mode = mode
        self.enable_perturbation = enable_perturbation
        self.detach_codes = detach_codes

        # 若外部没有提供共享 rounding 模块则自建
        if rounding_module is None:
            self.rounding_module = BlockSharedRounding(
                block_size=block_size,
                beta=beta,
                range_scale=range_scale,
                mode=mode,
                enable_perturbation=enable_perturbation
            )
        else:
            self.rounding_module = rounding_module

    def set_mode(self, mode: str, beta: float | None = None, enable_perturbation: bool | None = None):
        self.mode = mode
        if beta is not None:
            self.beta = beta
        if enable_perturbation is not None:
            self.enable_perturbation = enable_perturbation
        self.rounding_module.mode = self.mode
        self.rounding_module.beta = self.beta
        self.rounding_module.enable_perturbation = self.enable_perturbation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 最后一维必须整除 block_size
        if x.shape[-1] % self.block_size != 0:
            raise ValueError(f"Activation last dim {x.shape[-1]} not divisible by block_size={self.block_size}")
        pseudo, codes, scales, _, ord_hard = quantize_mxfp4_perturb(
            x,
            block_size=self.block_size,
            rounding_module=self.rounding_module,
            enable_perturbation=self.enable_perturbation,
            mode=self.mode,
            beta=self.beta,
            range_scale=self.range_scale,
            return_pseudo=True,
            detach_codes=self.detach_codes
        )
        return pseudo

    def delta_stats(self):
        rm = self.rounding_module
        if rm.delta_raw is None or not rm.enable_perturbation:
            return None
        d = rm.constrained_delta().detach()
        return {
            "mean": float(d.mean()),
            "std": float(d.std()),
            "min": float(d.min()),
            "max": float(d.max())
        }
