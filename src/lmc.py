import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from tqdm import tqdm
from typing import List, Tuple


def pd_lmc(f : nn.Module, g : nn.Module, h : nn.Module, 
           eta : float, n_samples : int, n_burnin : int, x_0 : torch.Tensor) -> List[np.ndarray]:
    """ 
    Runs the Primal-Dual Langevin Monte Carlo algorithm

    Inputs:
    --------
    - f (Module): function such that the target distribution pi verifies pi(x) = exp(-f(x)) / Z, where Z is a normalization constant
    - g (Module): defining function for the inequality constraints
    - h (Module): defining function for the equality constraints
    - eta (float): learning rate
    - n_samples (int): number of samples to generate from the solution mu*
    - n_burnin (int): number of iterations performed before sampling
    - x_0 (Tensor): initialization of x_k

    Outputs:
    --------
    - samples (List[ndarray]): generated samples from PD-LMC
    """
    
    x_k = torch.clone(x_0).requires_grad_(True)
    lambda_k = torch.zeros_like(g(x_0))
    nu_k = torch.zeros_like(h(x_0))

    samples_tensor = torch.empty((n_samples,) + x_0.shape)
    sample_idx = 0

    noise = sqrt(2 * eta) * torch.randn((n_samples + n_burnin,) + x_0.shape)

    for k in tqdm(range(n_samples + n_burnin), desc = "PD-LMC steps"):
        g_val = g(x_k)
        h_val = h(x_k)
        U_val = f(x_k) + torch.dot(lambda_k, g_val) + torch.dot(nu_k, h_val)
        grad = torch.autograd.grad(U_val, x_k, create_graph=False)[0]

        with torch.no_grad():
            x_next = x_k - eta * grad + noise[k]
            x_k = x_next.detach().requires_grad_(True)
            lambda_k = torch.max(lambda_k + eta * g_val, torch.zeros_like(lambda_k))
            nu_k = nu_k + eta * h_val

            if k >= n_burnin:
                samples_tensor[sample_idx] = x_k.detach()
                sample_idx += 1

    return samples_tensor.cpu().numpy()