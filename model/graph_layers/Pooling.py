import torch


class MaxPool(torch.nn.Module):
    def forward(self, x):
        maximum, indices = torch.max(x, dim=1)
        return maximum

class SumPool(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, dim=1)

class AvgPool(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=1)