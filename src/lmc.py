import torch
import torch.nn as nn
from math import sqrt


def pd_lmc(f : nn.Module, g : nn.Module, h : nn.Module, 
           eta : float, n_iter : int, x_0 : torch.Tensor) -> torch.Tensor:
    """ 
    Runs the Primal-Dual Langevin Monte Carlo algorithm

    Inputs:
    --------
    - f (torch.nn.Module): function such that the target distribution pi verifies pi(x) = exp(-f(x)) / Z, where Z is a normalization constant
    - g (torch.nn.Module): defining function for the inequality constraints
    - h (torch.nn.Module): defining function for the equality constraints
    - eta (float): learning rate
    - n_iter (int): number of iterations
    - x_0 (torch.Tensor): initialization of x_k

    Outputs:
    --------
    - x_K (torch.Tensor): generated sample from PD-LMC
    """
    
    x_k = torch.clone(x_0).requires_grad_(True)
    lambda_k = torch.zeros_like(g(x_0)).requires_grad_(False)
    nu_k = torch.zeros_like(h(x_0)).requires_grad_(False)

    for k in range(n_iter):
        x_k, lambda_k, nu_k = pd_lmc_iteration(f = f, g = g, h = h, x_k = x_k, lambda_k = lambda_k, nu_k = nu_k, eta = eta)

    return x_k.detach()


def pd_lmc_iteration(f : nn.Module, g : nn.Module, h : nn.Module,
                     x_k : torch.Tensor, lambda_k : torch.Tensor, nu_k : torch.Tensor,
                     eta : float):
    """ 
    Runs an iteration of the Primal-Dual Langevin Monte Carlo algorithm

    Inputs:
    --------
    - f (torch.nn.Module): function such that the target distribution pi verifies pi(x) = exp(-f(x)) / Z, where Z is a normalization constant
    - g (torch.nn.Module): defining function for the inequality constraints
    - h (torch.nn.Module): defining function for the equality constraints
    - x_k (torch.Tensor): value of x_k at the start of the iteration
    - lambda_k (torch.Tensor): value of lambda_k at the start of the iteration
    - eta (float): learning rate

    Outputs:
    --------
    - x_k (torch.Tensor): generated sample from PD-LMC
    """
    
    U_xk = f(x_k) + torch.dot(lambda_k, g(x_k)) + torch.dot(nu_k, h(x_k))
    U_xk.backward(retain_graph = True)
    x_next = x_k - eta * x_k.grad + sqrt(2 * eta) * torch.randn_like(x_k)

    lambda_k = torch.max(lambda_k + eta * g(x_k), torch.zeros_like(lambda_k))
    nu_k = nu_k + eta * h(x_k)
    x_k = x_next.detach().requires_grad_(True)

    return x_k, lambda_k, nu_k