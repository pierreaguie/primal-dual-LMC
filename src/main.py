import torch
from argparse import ArgumentParser
import numpy as np
from time import time

from lmc import pd_lmc
from utils import ConvexQuadratic, EllipsoidConstraint, Zero, truncated_gaussian_plotting
from rejection import rejection_sampling


parser = ArgumentParser(description="Script that samples from a 2D truncated standard Gaussian distribution using Primal-Dual Langevin Monte Carlo. \
                        The support of the truncated Gaussian is an ellipse aligned with the x,y axes.\n\
                        Note: the script only runs on cpu, cuda is not supported.")


parser.add_argument("--eta", type=float, default=5e-3, help= "Learning rate for the PD-LMC algorithm")
parser.add_argument("--n_burnin", type=int, default=10000, help= "Number of burn-in iterations for the PD-LMC algorithm")
parser.add_argument("--n_samples", type = int, default = 10000, help = "Number of samples to generate")
parser.add_argument("--m1", type = float, default=2, help= "First component of the mean vector of the base gaussian distribution")
parser.add_argument("--m2", type = float, default=2, help= "Second component of the mean vector of the base gaussian distribution")
parser.add_argument("--C11", type = float, default=0.5, help= "Length of the first semi-axis of the ellipsoid support")
parser.add_argument("--C22", type = float, default=2, help= "Length of the second semi-axis of the ellipsoid support")
parser.add_argument("--plot_path", type = str, default = "./figures/", help = "Directory where the plots are saved")

args = parser.parse_args()


if __name__ ==  "__main__":

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)    

    # Log-density of standard normal distribution of mean m
    m = torch.tensor([args.m1, args.m2])
    f = ConvexQuadratic(m)

    # g(x) <= 0 iff x is in the ellipse centered at 0 of covariance C
    C = torch.diag(torch.tensor([1/args.C11**2, 1/args.C22**2]))
    g = EllipsoidConstraint(torch.zeros_like(m), C)

    # h(x) = 0 (no equality constraint)
    h = Zero()

    # Generate samples with LMC
    start_time = time()
    samples_lmc = pd_lmc(f = f, g = g, h = h, eta = args.eta, n_samples = args.n_samples, n_burnin = args.n_burnin, x_0 = torch.zeros(2))
    print(f"Time to perform PD-LMC: {time() - start_time:.2f} seconds")

    # Generate samples via rejection sampling
    start_time = time()
    samples_rejection = rejection_sampling(n_samples = args.n_samples, m = m.detach().numpy(), C = C.detach().numpy())
    print(f"Time to perform rejection sampling: {time() - start_time:.2f} seconds")
    
    # Plotting
    truncated_gaussian_plotting(plot_path=args.plot_path, samples_lmc=samples_lmc, samples_rejection=samples_rejection, C11 = args.C11, C22 = args.C22)