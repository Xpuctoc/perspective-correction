import torch


def weighted_mse_loss(output, target, weight_penalty, device):
    loss = torch.nn.MSELoss()

    total_loss = torch.zeros([1], dtype=torch.float64, requires_grad=True, device=device)
    total_loss = total_loss + loss(output, target) * weight_penalty

    return total_loss
