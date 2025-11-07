import torch

@torch.no_grad()
def epsilon_statistics(model):
    stats = []
    for m in model.modules():
        if hasattr(m, "wq") and getattr(m.wq, "eps_param", None) is not None:

            # print(getattr(m, "layer_name", "?"))
            # print(m.wq.eps_param)
            # print(m.wq.eps_param.grad)


            eps = 0.5 * torch.tanh(m.wq.eps_param)
            stats.append({
                "layer": getattr(m, "layer_name", "?"),
                "mean": eps.mean().item(),
                "abs_mean": eps.abs().mean().item(),
                "max_abs": eps.abs().max().item(),
                "num_blocks": eps.numel()
            })

        if hasattr(m, "bias"):
            print(m.bias)
            print(m.bias.grad)
            exit()
    return stats
