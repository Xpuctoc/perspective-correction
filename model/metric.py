import torch


def rmse(output, target):
    with torch.no_grad():
        metric = torch.nn.MSELoss()
        return torch.sqrt(metric(output, target))
