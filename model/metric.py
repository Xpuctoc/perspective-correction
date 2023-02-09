import torch
import numpy as np


def rmse(output, target):
    with torch.no_grad():
        metric = torch.nn.MSELoss()
        return np.sqrt(metric(output, target))
