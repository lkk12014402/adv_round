import torch

@torch.no_grad()
def quantization_error(fp_weights, quant_weights):
    return (fp_weights - quant_weights).pow(2).mean().item()

@torch.no_grad()
def epsilon_stats(eps_params):
    stats = []
    for p in eps_params:
        eps = 0.5 * torch.tanh(p)
        stats.append({
            "mean": eps.mean().item(),
            "abs_mean": eps.abs().mean().item(),
            "max_abs": eps.abs().max().item()
        })
    return stats
