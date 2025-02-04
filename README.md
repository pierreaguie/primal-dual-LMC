# Primal-Dual Langevin Monte Carlo

Implementation of the Primal-Dual Langevin Monte Carlo algorithm introduced in [Chamon et al. "Constrained Sampling with Primal-Dual Langevin Monte Carlo"](https://arxiv.org/abs/2411.00568). This repository aims to reproduce the 2D truncated Gaussian experiment presented in the original paper, in the case where the support of the distirbution is an axis-aligned ellipse.

## Running the experiment

To reproduce the experiment, run the following command

```bash
python src/main.py
```

## Repository structure


```bash
├── README.md                             # This file
├── figures
│   └── generated_samples.png             # Generated figure when running src/main.py
├── requirements.txt
└── src
    ├── lmc.py                            # Includes the code for running Primal-Dual Langevin Monte Carlo
    ├── main.py                           # Runs the 2D truncated Gaussian experiment
    ├── rejection.py                      # Code for rejection sampling 
    └── utils.py                          # Utility functions for the 2D truncated Gaussian experiment
``` 

## Citation

This implementation is based on this paper

```bash
@InProceedings{Chamon24c,
    author = "Chamon, L. F. O. and Jaghargh, M. R. K. and Korba, A.",
    title = "Constrained sampling with primal-dual {L}angevin {M}onte {C}arlo",
    booktitle = "Conference on Neural Information Processing Systems (NeurIPS)",
    year = "2024",
}
```

and on its accompanying [repository](https://github.com/lfochamon/pdlmc/tree/main).