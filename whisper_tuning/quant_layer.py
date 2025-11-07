import torch
import torch.nn as nn
import torch.nn.functional as F
from fp4_impl_perturb import quantize_mxfp4_perturb, BlockSharedRounding
from activation_quant import ActivationQuantMXFP4

class QuantizedLinearMXFP4(nn.Module):
    """
    原始的权重量化线性层（保留你已有版本逻辑）。
    """
    def __init__(self,
                 linear: nn.Linear,
                 block_size: int = 32,
                 beta: float = 4.0,
                 range_scale: float = 0.5,
                 mode: str = "soft",
                 enable_perturbation: bool = True,
                 detach_codes: bool = True):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if self.in_features % block_size != 0:
            raise ValueError(f"in_features={self.in_features} not divisible by block_size={block_size}")

        self.register_buffer("weight_fp32", linear.weight.detach().clone())
        if linear.bias is not None:
            self.register_buffer("bias_fp32", linear.bias.detach().clone())
        else:
            self.bias_fp32 = None
        self.weight_fp32.requires_grad = False
        if self.bias_fp32 is not None:
            self.bias_fp32.requires_grad = False

        self.block_size = block_size
        self.beta = beta
        self.range_scale = range_scale
        self.mode = mode
        self.enable_perturbation = enable_perturbation
        self.detach_codes = detach_codes

        self.rounding_module = BlockSharedRounding(
            block_size=block_size,
            beta=beta,
            range_scale=range_scale,
            mode=mode,
            enable_perturbation=enable_perturbation
        )

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
        pseudo_float, codes, scales, _, ord_hard = quantize_mxfp4_perturb(
            self.weight_fp32,
            block_size=self.block_size,
            rounding_module=self.rounding_module,
            enable_perturbation=self.enable_perturbation,
            mode=self.mode,
            beta=self.beta,
            range_scale=self.range_scale,
            return_pseudo=True,
            detach_codes=self.detach_codes
        )
        return F.linear(x, pseudo_float, self.bias_fp32)

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


class QuantizedLinearActMXFP4(nn.Module):
    """
    同时支持：
      - 权重量化 (MXFP4 + δ 扰动)
      - 输入激活量化 (activation_pre)
      - 输出激活量化 (activation_post)

    可分别启/禁扰动与模式，降低耦合。

    参数新增：
      quant_activation_input: 是否量化输入激活
      quant_activation_output: 是否量化输出激活
      act_block_size: 激活量化块大小（需整除 hidden dim）
      shared_act_rounding: 输入与输出激活是否共享同一个 rounding module（减少参数）
    """
    def __init__(self,
                 linear: nn.Linear,
                 weight_block_size: int = 32,
                 act_block_size: int = 32,
                 weight_beta: float = 4.0,
                 act_beta: float = 4.0,
                 range_scale: float = 0.5,
                 weight_mode: str = "soft",
                 act_mode: str = "soft-ste",
                 enable_weight_perturb: bool = True,
                 quant_activation_input: bool = True,
                 quant_activation_output: bool = False,
                 enable_activation_perturb: bool = True,
                 shared_act_rounding: bool = False):
        super().__init__()
        # 权重准备
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if self.in_features % weight_block_size != 0:
            raise ValueError(f"in_features={self.in_features} not divisible by weight_block_size={weight_block_size}")

        self.register_buffer("weight_fp32", linear.weight.detach().clone())
        if linear.bias is not None:
            self.register_buffer("bias_fp32", linear.bias.detach().clone())
        else:
            self.bias_fp32 = None
        self.weight_fp32.requires_grad = False
        if self.bias_fp32 is not None:
            self.bias_fp32.requires_grad = False

        # 权重量化模块
        self.weight_rounding = BlockSharedRounding(
            block_size=weight_block_size,
            beta=weight_beta,
            range_scale=range_scale,
            mode=weight_mode,
            enable_perturbation=enable_weight_perturb,
            num_blocks=self.weight_fp32.shape[0] * (self.weight_fp32.shape[1]//weight_block_size)
        )
        self.weight_block_size = weight_block_size
        self.weight_mode = weight_mode
        self.weight_beta = weight_beta
        self.enable_weight_perturb = enable_weight_perturb

        # 激活量化设置
        self.quant_activation_input = quant_activation_input
        self.quant_activation_output = quant_activation_output
        self.enable_activation_perturb = enable_activation_perturb
        self.act_block_size = act_block_size
        self.act_mode = act_mode
        self.act_beta = act_beta
        self.range_scale = range_scale

        if shared_act_rounding:
            shared_rounding_module = BlockSharedRounding(
                block_size=act_block_size,
                beta=act_beta,
                range_scale=range_scale,
                mode=act_mode,
                enable_perturbation=enable_activation_perturb
            )
        else:
            shared_rounding_module = None

        # 输入激活量化器
        self.act_in_quant = None
        if quant_activation_input:
            self.act_in_quant = ActivationQuantMXFP4(
                block_size=act_block_size,
                beta=act_beta,
                range_scale=range_scale,
                mode=act_mode,
                enable_perturbation=enable_activation_perturb,
                rounding_module=shared_rounding_module
            )
        # 输出激活量化器
        self.act_out_quant = None
        if quant_activation_output:
            self.act_out_quant = ActivationQuantMXFP4(
                block_size=act_block_size,
                beta=act_beta,
                range_scale=range_scale,
                mode=act_mode,
                enable_perturbation=enable_activation_perturb,
                rounding_module=shared_rounding_module
            )

    def set_weight_mode(self, mode: str, beta: float | None = None, enable_perturb: bool | None = None):
        self.weight_mode = mode
        if beta is not None:
            self.weight_beta = beta
        if enable_perturb is not None:
            self.enable_weight_perturb = enable_perturb
        self.weight_rounding.mode = self.weight_mode
        self.weight_rounding.beta = self.weight_beta
        self.weight_rounding.enable_perturbation = self.enable_weight_perturb

    def set_activation_mode(self, mode: str, beta: float | None = None, enable_perturb: bool | None = None):
        self.act_mode = mode
        if beta is not None:
            self.act_beta = beta
        if enable_perturb is not None:
            self.enable_activation_perturb = enable_perturb
        if self.act_in_quant:
            self.act_in_quant.set_mode(mode, beta=self.act_beta, enable_perturbation=self.enable_activation_perturb)
        if self.act_out_quant:
            self.act_out_quant.set_mode(mode, beta=self.act_beta, enable_perturbation=self.enable_activation_perturb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入激活量化
        if self.act_in_quant is not None:
            x = self.act_in_quant(x)

        # 权重量化
        pseudo_w, _, _, _, _ = quantize_mxfp4_perturb(
            self.weight_fp32,
            block_size=self.weight_block_size,
            rounding_module=self.weight_rounding,
            enable_perturbation=self.enable_weight_perturb,
            mode=self.weight_mode,
            beta=self.weight_beta,
            range_scale=self.range_scale,
            return_pseudo=True,
            detach_codes=True
        )
        y = F.linear(x, pseudo_w, self.bias_fp32)

        # 输出激活量化
        if self.act_out_quant is not None:
            y = self.act_out_quant(y)
        return y

    def weight_delta_stats(self):
        rm = self.weight_rounding
        if rm.delta_raw is None or not rm.enable_perturbation:
            return None
        d = rm.constrained_delta().detach()
        return {
            "mean": float(d.mean()),
            "std": float(d.std()),
            "min": float(d.min()),
            "max": float(d.max())
        }

    def activation_delta_stats(self):
        stats_in = self.act_in_quant.delta_stats() if self.act_in_quant else None
        stats_out = self.act_out_quant.delta_stats() if self.act_out_quant else None
        return {"in": stats_in, "out": stats_out}
