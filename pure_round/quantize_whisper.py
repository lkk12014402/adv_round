import torch
import torch.nn as nn
from mxfp4_quantizer import E2M1WeightQuantizer, E2M1ActivationQuantizer

class QuantizedLinearWhisper(nn.Module):
    def __init__(self,
                 linear: nn.Linear,
                 layer_name: str,
                 layer_max_w: float,
                 layer_max_act: float,
                 block_size_w: int = 32,
                 block_size_act: int = 32,
                 act_quant: bool = True,
                 init_mode: str = "mixed"):
        super().__init__()
        self.layer_name = layer_name
        self.weight_fp = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=True) if linear.bias is not None else None
        self.wq = E2M1WeightQuantizer(block_size=block_size_w,
                                      layer_max=layer_max_w,
                                      init_mode=init_mode)
        self.act_quant = act_quant
        if act_quant:
            self.aq = E2M1ActivationQuantizer(block_size=block_size_act,
                                              layer_max=layer_max_act)
        else:
            self.aq = None

    def forward(self, x):
        if self.act_quant and self.aq is not None:
            x_q, _ = self.aq(x, quantize=True)
        else:
            x_q = x
        w_q, meta_w = self.wq(self.weight_fp)
        out = torch.nn.functional.linear(x_q, w_q, self.bias)
        return out, {"weight": meta_w}
        # return out

def _get_parent(root: nn.Module, path: str):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent

def apply_quantization_whisper(model,
                               adaptive_max_config: dict,
                               block_size_w: int = 32,
                               block_size_act: int = 32,
                               act_quant: bool = True,
                               init_mode: str = "mixed"):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in adaptive_max_config:
            if name == "proj_out":
                continue
            parent = _get_parent(model, name)
            attr = name.split(".")[-1]
            cfg = adaptive_max_config[name]
            qlayer = QuantizedLinearWhisper(
                linear=module,
                layer_name=name,
                layer_max_w=cfg["weight_max"],
                layer_max_act=cfg["act_max"],
                block_size_w=block_size_w,
                block_size_act=block_size_act,
                act_quant=act_quant,
                init_mode=init_mode
            )
            setattr(parent, attr, qlayer)
    return model
