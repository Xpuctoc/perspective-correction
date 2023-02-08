import torch


def mse_metric(output, target):
    with torch.no_grad():
        metric = torch.nn.MSELoss()
        return metric(output, target)
