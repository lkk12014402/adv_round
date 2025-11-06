import torch
import torch.nn as nn
from quantization.mxfp4 import MXFP4WeightQuantizer, MXFP4ActivationQuantizer

class QuantizedLinear(nn.Module):
    """
    Linear layer with:
      - Frozen FP weight (no gradient)
      - Learnable epsilon for weight MXFP4 quantization
      - Dynamic activation MXFP4 quantization (optional)
    """
    def __init__(self,
                 linear: nn.Linear,
                 group_size_weight: int = 32,
                 group_size_act: int = 32,
                 act_quant: bool = True,
                 act_use_ema: bool = True,
                 act_ema_momentum: float = 0.95,
                 act_recompute_interval: int = 1):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Freeze original weight
        self.weight_fp = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            # Preserving bias allows for training to absorb the output mean shift caused by quantization.
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=True)
        else:
            self.bias = None

        # Weight quantizer with learnable epsilon
        self.weight_quantizer = MXFP4WeightQuantizer(group_size=group_size_weight)

        # Activation quantizer (dynamic)
        self.act_quant = act_quant
        if act_quant:
            self.activation_quantizer = MXFP4ActivationQuantizer(
                group_size=group_size_act,
                use_act_ema=act_use_ema,
                act_ema_momentum=act_ema_momentum,
                recompute_interval=act_recompute_interval
            )
        else:
            self.activation_quantizer = None

    def forward(self,
                x: torch.Tensor,
                adversarial_delta: torch.Tensor = None,
                quantize_activation: bool = True):
        """
        x: input activation [..., in_features]
        adversarial_delta: optional per-group delta for weight epsilon
        quantize_activation: Should dynamic MXFP4 quantization be performed on the input x?
        """
        if self.act_quant and quantize_activation:
            x_q, act_exps = self.activation_quantizer(x, quantize=True)
        else:
            x_q, act_exps = x, None

        # 2. ε & adversarial_delta）
        w_q, eps_eff, w_exps = self.weight_quantizer(self.weight_fp, adversarial_delta)

        # 3. matmul
        out = torch.nn.functional.linear(x_q, w_q, self.bias)

        meta = {
            "weight_eps": eps_eff,
            "weight_exponents": w_exps,
            "act_exponents": act_exps
        }
        return out, meta
