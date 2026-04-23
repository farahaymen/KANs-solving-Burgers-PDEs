# KAN-PINN: Kolmogorov–Arnold Networks as Physics-Informed Neural Networks for Solving the 1D Viscous Burgers' Equation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![IEEE Access](https://img.shields.io/badge/Target-IEEE%20Access-red)](https://ieeeaccess.ieee.org/)

---

## Abstract

This repository provides the full source code and experimental framework accompanying the paper:

> **"Solving the 1D Viscous Burgers' Equation via Kolmogorov–Arnold Networks: A Comparative Study of Polynomial Basis Functions and Viscosity Regimes"**

We systematically investigate four Kolmogorov–Arnold Network (KAN) architectures — B-Spline, Chebyshev, Hermite, and Legendre — as physics-informed solvers for the 1D viscous Burgers' equation across two viscosity regimes (ν = 1.0 and ν = 0.005) and two polynomial degrees (k = 3 and k = 5). All KAN variants are benchmarked against a standard Physics-Informed Neural Network (PINN) baseline and a high-resolution Finite Difference (FD) reference solver.

---

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Model Architectures](#model-architectures)
- [Experimental Design](#experimental-design)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Background

### Governing Equation

We solve the **1D viscous Burgers' equation** on the spatio-temporal domain (x, t) ∈ [−3, 3] × [0, 1]:

```
∂u/∂t + u ∂u/∂x − ν ∂²u/∂x² = 0
```

with initial condition:

```
u(x, 0) = exp(−x²)
```

and homogeneous Dirichlet boundary conditions:

```
u(−3, t) = u(3, t) = 0
```

The parameter ν denotes the kinematic viscosity. Two physically distinct regimes are studied:
- **ν = 1.0**: smooth diffusion-dominated regime
- **ν = 0.005**: advection-dominated regime with shock formation

### Kolmogorov–Arnold Networks (KANs)

KANs [Liu et al., 2024] replace fixed activation functions in standard MLPs with learnable univariate functions parameterized as polynomial or spline expansions on each edge. This confers improved function approximation properties, interpretability, and smoothness — qualities of direct relevance to solving PDEs.

### Physics-Informed Neural Networks (PINNs)

PINNs [Raissi et al., 2019] embed PDE constraints directly into the training loss, combining data-driven fitting with physical consistency. The composite loss is:

```
L = L_IC + L_BC + L_PDE
```

where each term is a mean squared error over sampled collocation points.

### Finite Difference Reference Solver

A high-resolution explicit Lax–Wendroff-type FD solver (Δx = 6/400, Δt = 10⁻⁴) provides ground-truth reference solutions against which all neural models are compared.

---

## Features

| Feature | Details |
|---|---|
| **KAN Architectures** | B-Spline, Chebyshev, Hermite, Legendre |
| **Polynomial Degrees** | k = 3 (cubic), k = 5 (quintic) |
| **Viscosity Regimes** | ν = 1.0 (smooth), ν = 0.005 (shock) |
| **Baseline Model** | Standard MLP PINN |
| **Reference Solver** | Explicit Finite Difference |
| **Loss Components** | IC loss, BC loss, PDE residual loss |
| **Loss Smoothing** | Exponential Moving Average (EMA, α = 0.999) |
| **Hardware** | CUDA GPU acceleration (auto-detected) |
| **Visualization** | Automated comparison plots at t ∈ {0.0, 0.3, 0.6, 1.0} |

---

## Repository Structure

```
.
├── config.py                    # Global hyperparameters and domain settings
├── main.py                      # Single-model training entry point
├── run_all_experiments.py       # Full factorial experiment runner
│
├── data/
│   ├── __init__.py
│   ├── dataset.py               # Training data generation (IC, BC, collocation)
│   └── fd_solver.py             # Finite difference reference solver
│
├── models/
│   ├── __init__.py
│   ├── kan_bspline.py           # B-Spline KAN layer and network
│   ├── kan_chebyshev.py         # Chebyshev polynomial KAN
│   ├── kan_hermite.py           # Hermite polynomial KAN
│   ├── kan_legendre.py          # Legendre polynomial KAN
│   └── pinn.py                  # MLP PINN baseline
│
├── utils/
│   ├── __init__.py
│   ├── losses.py                # PDE residual and composite loss function
│   ├── predict.py               # Inference utility
│   ├── smoothing.py             # EMA loss smoothing
│   ├── plot.py                  # Plot and save FD vs model comparison
│   └── plotting.py              # Training loss visualization
│
└── results/                     # Auto-generated output directory
    └── {model}/degree_{k}/nu_{ν}/
        └── compare_t_{t}.png
```

---

## Installation

### Prerequisites

- Python ≥ 3.8
- PyTorch ≥ 2.0 (with CUDA support recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/kan-pinn-burgers.git
cd kan-pinn-burgers

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch numpy matplotlib
```

---

## Quick Start

### Run All Experiments (Full Factorial)

Trains all 4 KAN variants × 2 degrees × 2 viscosities = **16 configurations** and saves comparison plots automatically:

```bash
python run_all_experiments.py
```

Results are saved to:

```
results/{model_name}/degree_{k}/nu_{nu}/compare_t_{t_val}.png
```

### Train a Single Configuration

```bash
python main.py --model legendre --degree 5 --nu 0.005
```

Supported `--model` values: `bspline`, `chebyshev`, `hermite`, `legendre`, `pinn`

---

## Configuration

All global hyperparameters are centralized in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `x_lower`, `x_upper` | −3.0, 3.0 | Spatial domain bounds |
| `t_lower`, `t_upper` | 0.0, 1.0 | Temporal domain bounds |
| `N_ic` | 56 | Initial condition sample points |
| `N_bc` | 56 | Boundary condition sample points |
| `N_coll` | 10,000 | PDE collocation points |
| `layer_sizes` | [2, 64, 64, 64, 64, 1] | Network layer widths |
| `degrees` | [3, 5] | Polynomial degrees to sweep |
| `viscosities` | [1.0, 0.005] | Viscosity values to sweep |
| `epochs` | 10,000 | Training epochs |
| `lr` | 1×10⁻³ | Adam optimizer learning rate |
| `alpha` | 0.999 | EMA smoothing coefficient |
| `nx_fd` | 401 | FD solver spatial grid points |
| `dt_fd` | 1×10⁻⁴ | FD solver time step |
| `time_slices` | [0.0, 0.3, 0.6, 1.0] | Evaluation time snapshots |

---

## Model Architectures

### B-Spline KAN (`kan_bspline.py`)

Employs a uniform B-spline basis of configurable order k and grid size g. Each KAN layer computes:

```
y = W_base · σ(x) + W_spline · B(x)
```

where B(x) are the B-spline basis functions evaluated on a learnable grid, and σ is the SiLU base activation.

### Chebyshev KAN (`kan_chebyshev.py`)

Uses Chebyshev polynomials of the first kind T_n(x) as the basis expansion. Input is normalized to [−1, 1] via linear scaling, and the recurrence T_{n+1}(x) = 2x T_n(x) − T_{n−1}(x) is used for efficient computation.

### Hermite KAN (`kan_hermite.py`)

Uses probabilist's Hermite polynomials via the recurrence H_{n+1}(x) = 2x H_n(x) − 2n H_{n−1}(x), normalized to the domain [−3, 3]. Well-suited to Gaussian-shaped initial conditions.

### Legendre KAN (`kan_legendre.py`)

Applies Legendre polynomials defined on [−1, 1] via the Bonnet recurrence (n+1) P_{n+1}(x) = (2n+1)x P_n(x) − n P_{n−1}(x). Provides orthogonality and numerically stable polynomial approximation.

### PINN Baseline (`pinn.py`)

A standard fully-connected MLP with the same `layer_sizes = [2, 64, 64, 64, 64, 1]` architecture. Tanh activations. Serves as the classical reference for comparison with all KAN architectures.

---

## Experimental Design

The full experimental matrix is:

| KAN Variant | Degree k=3 / ν=1.0 | Degree k=3 / ν=0.005 | Degree k=5 / ν=1.0 | Degree k=5 / ν=0.005 |
|---|:---:|:---:|:---:|:---:|
| B-Spline | ✓ | ✓ | ✓ | ✓ |
| Chebyshev | ✓ | ✓ | ✓ | ✓ |
| Hermite | ✓ | ✓ | ✓ | ✓ |
| Legendre | ✓ | ✓ | ✓ | ✓ |
| PINN (baseline) | — | — | — | — |

### Training Protocol

- **Optimizer**: Adam with learning rate 1×10⁻³
- **Epochs**: 10,000 per configuration (3,000 in full-sweep mode)
- **Loss**: MSE(IC) + MSE(BC) + MSE(PDE residual)
- **Evaluation**: L² error vs. FD reference at t ∈ {0.0, 0.3, 0.6, 1.0}

### Data Generation (`dataset.py`)

| Dataset | Sampling | Size |
|---|---|---|
| Initial condition (IC) | Uniform random on [x_lower, x_upper] at t=0 | 56 points |
| Boundary condition (BC) | Uniform random in time at x = ±3 | 56 points |
| Collocation points | Uniform random in (x, t) ∈ [−3,3]×[0,1] | 10,000 points |

---

## Results

Plots are automatically saved for each experimental configuration in:

```
results/{model_name}/degree_{k}/nu_{nu}/
    compare_t_0.0.png
    compare_t_0.3.png
    compare_t_0.6.png
    compare_t_1.0.png
```

Each figure shows the FD reference solution (solid blue) versus the KAN prediction (dashed red) at the given time snapshot.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025kanburgersieee,
  author    = {Last, First and Co-Author, Name},
  title     = {Solving the 1D Viscous Burgers' Equation via Kolmogorov–Arnold Networks:
               A Comparative Study of Polynomial Basis Functions and Viscosity Regimes},
  journal   = {IEEE Access},
  year      = {2025},
  volume    = {},
  pages     = {},
  doi       = {}
}
```

### Key References

- Liu, Z., et al. (2024). KAN: Kolmogorov–Arnold Networks. *arXiv:2404.19756*.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- Burgers, J. M. (1948). A mathematical model illustrating the theory of turbulence. *Advances in Applied Mechanics*, 1, 171–199.

---

## Acknowledgements

The authors acknowledge the use of open-source software including PyTorch, NumPy, and Matplotlib. 
