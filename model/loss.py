import torch


def mse_loss(output, target):
    loss = torch.nn.MSELoss()
    return loss(output, target)
