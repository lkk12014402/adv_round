import torch
import math

class MXFP4Quantizer(torch.nn.Module):
    """
    Block-Floating FP4 (sign + shared exponent + 3-bit mantissa) quantizer with adversarial epsilon.
    Each group shares an exponent (power-of-two scale). Mantissa in [0,1], step=1/8.
    Representation: w ≈ sign(w) * 2^{e_base} * mantissa_q
    mantissa_q = round( clamp(|w|/2^{e_base} + epsilon + zeta, 0, 1) * 8 ) / 8
    epsilon in [-0.5,0.5] shifts mantissa before rounding.
    """
    def __init__(self, group_size: int = 32, eps_init: float = 0.0, requires_grad: bool = True):
        super().__init__()
        self.group_size = group_size
        self.eps_param = None
        self._initialized = False
        self.eps_init = eps_init
        self.requires_grad = requires_grad

    def initialize(self, weight: torch.Tensor):
        # Group along the last dimension (e.g., out_features for linear)
        numel = weight.numel()
        last_dim = weight.shape[-1]
        assert numel % last_dim == 0, "Unexpected weight shape for grouping."
        groups = math.ceil(last_dim / self.group_size)
        # One epsilon per group
        p = torch.zeros(groups, dtype=weight.dtype, device=weight.device)
        p += self.eps_init
        p.requires_grad_(self.requires_grad)
        self.eps_param = torch.nn.Parameter(p, requires_grad=self.requires_grad)
        self._initialized = True

    @staticmethod
    def ste_round(x: torch.Tensor):
        # Straight-Through Estimator for round
        return torch.round(x)

    def forward(self, weight: torch.Tensor, adversarial_delta: torch.Tensor = None):
        """
        Returns quantized weight and auxiliary info.
        adversarial_delta: same shape as eps_param, in [-rho,rho] after projection.
        """
        if not self._initialized:
            self.initialize(weight)
        # Build epsilon in [-0.5,0.5]
        eps = 0.5 * torch.tanh(self.eps_param)
        if adversarial_delta is not None:
            # Apply adversarial shift ensuring total still in [-0.5,0.5]
            total = eps + adversarial_delta
            total = torch.clamp(total, -0.5, 0.5)
            eps_eff = total
        else:
            eps_eff = eps

        # Reshape/group eps over last dim
        last_dim = weight.shape[-1]
        groups = math.ceil(last_dim / self.group_size)
        if groups != eps_eff.numel():
            raise ValueError("Mismatch in groups and epsilon length.")

        # Compute per-group exponents
        # Flatten last dim, process in chunks
        w = weight
        quant_w = torch.empty_like(w)
        exponents = []
        for g in range(groups):
            start = g * self.group_size
            end = min((g + 1) * self.group_size, last_dim)
            # Slice group along last dimension
            w_slice = w[..., start:end]
            # Absolute max for exponent
            max_abs = w_slice.abs().max()
            if max_abs == 0:
                e_base = 0  # All zeros, exponent immaterial
            else:
                e_base = math.floor(math.log2(max_abs.item()))
            exponents.append(e_base)
            scale = 2.0 ** e_base
            # Normalize mantissa to [0, ~max_abs/scale] which should be ≤ 2
            # We clamp ratio to [0,1] before quantizing; values > scale saturation.
            ratio = torch.clamp(w_slice.abs() / scale, 0.0, 1.0)

            # Apply epsilon shift
            shifted = ratio + eps_eff[g]

            shifted = torch.clamp(shifted, 0.0, 1.0)

            # Quantize mantissa to 3 bits => step=1/8
            mantissa_grid = self.ste_round(shifted * 8.0) / 8.0
            # Reconstruct sign
            sign = torch.sign(w_slice)
            quant_slice = sign * scale * mantissa_grid
            # Scatter back
            quant_w[..., start:end] = quant_slice

        exponents_tensor = torch.tensor(exponents, device=weight.device, dtype=weight.dtype)
        return quant_w, eps_eff.detach(), exponents_tensor


class MXFP4ActivationQuantizer(torch.nn.Module):
    """
    Dynamic MXFP4 quantizer for ACTIVATIONS.
    - Each forward can recompute per-group exponent based on current batch.
    - Optional EMA smoothing of exponents to stabilize.
    - Mantissa quantization: sign(a) * 2^{e} * Q_m(|a|/2^{e})
    - No learnable epsilon by default (可扩展).
    """
    def __init__(self,
                 group_size: int = 32,
                 use_act_ema: bool = True,
                 act_ema_momentum: float = 0.95,
                 recompute_interval: int = 1):
        super().__init__()
        self.group_size = group_size
        self.use_act_ema = use_act_ema
        self.act_ema_momentum = act_ema_momentum
        self.recompute_interval = recompute_interval
        self.register_buffer("running_exponents", None)
        self._step_counter = 0
        self._last_exponents = None  # cache last used exponents

    @staticmethod
    def ste_round(x: torch.Tensor):
        return torch.round(x)

    def _compute_exponents(self, act: torch.Tensor):
        last_dim = act.shape[-1]
        groups = math.ceil(last_dim / self.group_size)
        exps = []
        for g in range(groups):
            start = g * self.group_size
            end = min((g + 1) * self.group_size, last_dim)
            a_slice = act[..., start:end]
            max_abs = a_slice.abs().max()
            if max_abs == 0:
                e_base = 0
            else:
                e_base = math.floor(math.log2(max_abs.item()))
            exps.append(e_base)
        return torch.tensor(exps, device=act.device, dtype=act.dtype)

    def _update_running(self, new_exps: torch.Tensor):
        if self.running_exponents is None:
            self.running_exponents = new_exps
        else:
            self.running_exponents = (
                self.act_ema_momentum * self.running_exponents +
                (1 - self.act_ema_momentum) * new_exps
            )

    def forward(self, act: torch.Tensor, quantize: bool = True):
        """
        act: activation tensor [..., feature_dim]
        Returns quantized activation and used exponents.
        """
        if not quantize:
            return act, None

        last_dim = act.shape[-1]
        groups = math.ceil(last_dim / self.group_size)

        # Decide whether to recompute exponents
        recompute = (self._step_counter % self.recompute_interval == 0) or (self._last_exponents is None)
        if recompute:
            raw_exps = self._compute_exponents(act)
            if self.use_act_ema:
                self._update_running(raw_exps)
                exps_to_use = torch.round(self.running_exponents)  # optional rounding
            else:
                exps_to_use = raw_exps
            self._last_exponents = exps_to_use
        else:
            exps_to_use = self._last_exponents

        quant_act = torch.empty_like(act)
        for g in range(groups):
            start = g * self.group_size
            end = min((g + 1) * self.group_size, last_dim)
            a_slice = act[..., start:end]
            e_base = int(exps_to_use[g].item())
            scale = 2.0 ** e_base

            ratio = torch.clamp(a_slice.abs() / scale, 0.0, 1.0)
            mantissa = self.ste_round(ratio * 8.0) / 8.0
            quant_slice = torch.sign(a_slice) * scale * mantissa
            quant_act[..., start:end] = quant_slice

        self._step_counter += 1
        return quant_act, exps_to_use.detach()


@torch.no_grad()
def mxfp4_quantization_error(fp_tensor: torch.Tensor, q_tensor: torch.Tensor):
    return (fp_tensor - q_tensor).pow(2).mean().item()


def adversarial_perturb(eps_param: torch.Tensor,
                        loss_fn,
                        model_forward_fn,
                        weight: torch.Tensor,
                        labels: torch.Tensor,
                        rho: float,
                        pgd_steps: int = 1,
                        alpha: float = 0.05):
    """
    Generate adversarial delta for epsilon parameters (per-group).
    eps_param: raw p (before tanh).
    Returns delta tensor (same shape as eps_param) in [-rho, rho].
    """
    device = eps_param.device
    delta = torch.zeros_like(eps_param, device=device, requires_grad=True)

    for _ in range(pgd_steps):
        # Current effective epsilon
        eps = 0.5 * torch.tanh(eps_param)
        total = eps + delta
        total = torch.clamp(total, -0.5, 0.5)

        # Forward with temporary adversarial epsilon
        quant_w, _, _ = model_forward_fn(weight, total)
        logits = quant_w  # model_forward_fn should produce actual logits; simplified placeholder
        # If model_forward_fn returns logits, adapt accordingly.
        # Here assume it returns quantized weight; user must integrate in training loop.
        raise NotImplementedError("Integrate adversarial_perturb with actual model forward producing logits.")

    return delta.detach()
