import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from time import time

from lmc import pd_lmc
from utils import ConvexQuadratic, EllipsoidConstraint, Zero
from rejection import rejection_sampling


parser = ArgumentParser(description="Script that samples from a 2D truncated standard Gaussian distribution using Primal-Dual Langevin Monte Carlo. \
                        The support of the truncated Gaussian is an ellipse aligned with the x,y axes.")


parser.add_argument("--eta", type=float, default=1e-3, help= "Learning rate for the PD-LMC algorithm")
parser.add_argument("--n_iter", type=int, default=50, help= "Number of iterations for the PD-LMC algorithm")
parser.add_argument("--m1", type = float, default=2, help= "First component of the mean vector of the base gaussian distribution")
parser.add_argument("--m2", type = float, default=2, help= "Second component of the mean vector of the base gaussian distribution")
parser.add_argument("--C11", type = float, default=0.5, help= "Length of the first semi-axis of the ellipsoid support")
parser.add_argument("--C22", type = float, default=2, help= "Length of the second semi-axis of the ellipsoid support")
parser.add_argument("--n_samples", type = int, default = 500, help = "Number of samples to generate")
parser.add_argument("--plot_path", type = str, default = "./figures/", help = "Path of directory where the plots are saved")

args = parser.parse_args()


if __name__ ==  "__main__":

    # Log-density of standard normal distribution of mean m
    m = torch.tensor([args.m1, args.m2])
    f = ConvexQuadratic(m)

    # g(x) <= 0 iff x is in the ellipsoid centered at 0 aligned with x and y, of semi axes of length C11 and C22
    C = torch.diag(torch.tensor([1/args.C11**2, 1/args.C22**2]))
    g = EllipsoidConstraint(torch.zeros_like(m), C)

    # h(x) = 0 (no equality constraint)
    h = Zero()


    # Generate samples with LMC
    samples_lmc = np.zeros((args.n_samples, 2))
    for i in tqdm(range(args.n_samples)):
        samples_lmc[i,:] = pd_lmc(f = f, g = g, h = h, eta = args.eta, n_iter = args.n_iter, x_0 = torch.zeros(2)).numpy()
    lmc_mean = np.mean(samples_lmc, axis = 0)

    
    # Generate samples via rejection sampling
    samples_rejection = rejection_sampling(n_samples = args.n_samples, m = m.detach().numpy(), C = C.detach().numpy())
    rejection_mean = np.mean(samples_rejection, axis = 0)
    


    # Plotting
    fig, ax = plt.subplots(1, 2, figsize = (14,7))

    ellipse = Ellipse((0., 0.), 2 * args.C11, 2 * args.C22, angle = 0., facecolor = "blue", alpha = .2, edgecolor = "black")

    ax[0].add_patch(ellipse)
    ax[0].scatter(samples_lmc[:, 0], samples_lmc[:, 1], alpha=0.5)
    ax[0].scatter(lmc_mean[0], lmc_mean[1], marker = 'x', color = 'orange', label = "PD-LMC mean esimate")
    ax[0].scatter(rejection_mean[0], rejection_mean[1], marker = '+', color = 'red', label = "Rejection sampling mean estimate")
    ax[0].set_xlabel('x1')
    ax[0].set_ylabel('x2')
    ax[0].set_title('Generated Samples from PD-LMC')


    ellipse = Ellipse((0., 0.), 2 * args.C11, 2 * args.C22, angle = 0., facecolor = "blue", alpha = .2, edgecolor = "black")

    ax[1].add_patch(ellipse)
    ax[1].scatter(samples_rejection[:, 0], samples_rejection[:, 1], alpha=0.5)
    ax[1].scatter(lmc_mean[0], lmc_mean[1], marker = 'x', color = 'orange', label = "PD-LMC mean esimate")
    ax[1].scatter(rejection_mean[0], rejection_mean[1], marker = '+', color = 'red', label = "Rejection sampling mean estimate")
    ax[1].set_xlabel('x1')
    ax[1].set_ylabel('x2')
    ax[1].set_title('Generated Samples from rejection sampling')
    ax[1].legend(loc = 'lower left')


    plt.savefig(f"{args.plot_path}/generated_samples.png")