import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class ConvexQuadratic(nn.Module):

    def __init__(self, m : torch.Tensor):
        super(ConvexQuadratic, self).__init__()
        self.m = m

    def forward(self, x):
        return 1/2 * torch.dot(x - self.m, x - self.m).unsqueeze(0)
    


class EllipsoidConstraint(nn.Module):
    
    def __init__(self, m : torch.Tensor, C : torch.Tensor, slack : float = 0.001):
        super(EllipsoidConstraint, self).__init__()
        self.m = m
        self.C = C
        self.slack = slack

    def forward(self, x):
        return F.relu(torch.dot(torch.matmul(self.C, x - self.m), x - self.m).unsqueeze(0) - 1) - self.slack
    


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return torch.Tensor([0.])
        


def truncated_gaussian_plotting(plot_path : str, 
                                samples_lmc : np.ndarray, samples_rejection : np.ndarray, 
                                C11 : float, C22 : float):
    """ 
    Reproduces Figure 2 in Chamon et al. "Constrained Sampling with Primal-Dual Langevin Monte Carlo"
    """
    
    lmc_mean = np.mean(samples_lmc, axis = 0)
    rejection_mean = np.mean(samples_rejection, axis = 0)

    fig, ax = plt.subplots(1, 2, figsize = (14,7))

    ellipse = Ellipse((0., 0.), 2 * C11, 2 * C22, angle = 0., facecolor = "blue", alpha = .2, edgecolor = "black")
    ax[0].add_patch(ellipse)
    ax[0].scatter(samples_lmc[:, 0], samples_lmc[:, 1], alpha=0.2)
    ax[0].scatter(lmc_mean[0], lmc_mean[1], marker = 'x', color = 'orange', label = "PD-LMC mean esimate")
    ax[0].scatter(rejection_mean[0], rejection_mean[1], marker = '+', color = 'red', label = "Rejection sampling mean estimate")
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_title('Generated Samples from PD-LMC')


    ellipse = Ellipse((0., 0.), 2 * C11, 2 * C22, angle = 0., facecolor = "blue", alpha = .2, edgecolor = "black")
    ax[1].add_patch(ellipse)
    ax[1].scatter(samples_rejection[:, 0], samples_rejection[:, 1], alpha=0.2)
    ax[1].scatter(lmc_mean[0], lmc_mean[1], marker = 'x', color = 'orange', label = "PD-LMC mean esimate")
    ax[1].scatter(rejection_mean[0], rejection_mean[1], marker = '+', color = 'red', label = "Rejection sampling mean estimate")
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].set_title('Generated Samples from rejection sampling')
    ax[1].legend(loc = 'lower left')

    plt.savefig(f"{plot_path}/generated_samples.png")