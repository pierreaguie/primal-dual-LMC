import torch
import torch.nn as nn


class ConvexQuadratic(nn.Module):

    def __init__(self, m : torch.Tensor):
        super(ConvexQuadratic, self).__init__()
        self.m = m

    def forward(self, x):
        return 1/2 * torch.dot(x - self.m, x - self.m).unsqueeze(0)
    


class EllipsoidConstraint(nn.Module):
    
    def __init__(self, m : torch.Tensor, C : torch.Tensor):
        super(EllipsoidConstraint, self).__init__()
        self.m = m
        self.C = C

    def forward(self, x):
        return torch.dot(torch.matmul(self.C, x - self.m), x - self.m).unsqueeze(0) - 1
    


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return torch.Tensor([0.])
        