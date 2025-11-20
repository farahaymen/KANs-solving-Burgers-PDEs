# Kolmogorov–Arnold Networks (KANs) & PINNs for Solving Burgers’ Equation

This repository provides a full experimental framework for solving the **1D viscous Burgers’ equation** using:

- **Kolmogorov–Arnold Networks (KANs)**
- **Physics-Informed Neural Networks (PINNs)**
- **Finite Difference (FD)** reference solver

It includes **four KAN variants**, supports **two polynomial degrees**, two viscosity regimes, and automatically generates **visual comparisons** for every configuration.

---

## ✨ Features

- ✔ **Four KAN architectures**
  - B-Spline KAN  
  - Chebyshev KAN  
  - Hermite KAN  
  - Legendre KAN  

- ✔ **Two polynomial degrees**
  - Degree **3**
  - Degree **5**

- ✔ **Two viscosity values**
  - **ν = 1.0** (smooth regime)  
  - **ν = 0.005** (shock-forming regime)

- ✔ **PINN baseline model**
- ✔ Automatic FD solver for ground truth  
- ✔ Automatic visualization and result saving  
- ✔ Modular structure  
- ✔ CUDA acceleration supported  

---


---

## Background

We solve the **1D viscous Burgers’ equation**:

\[
u_t + u u_x - \nu u_{xx} = 0
\]

### Kolmogorov–Arnold Networks (KANs)
KANs use spline or polynomial expansions inside each layer to learn mappings with improved interpretability and smoothness.

### Physics-Informed Neural Networks (PINNs)
PINNs embed the PDE into the loss function, enforcing physics constraints during training.

### Finite Difference (FD)
A high-resolution explicit FD solver computes the reference solution used for all comparisons.

---

## Running All Experiments

Train every KAN model (for degrees 3 & 5, ν=1.0 & 0.005) and generate visualizations:

```bash
python run_all_experiments.py

---

## Running One Model Manually

To train a single configuration:

```bash
python main.py --model legendre --degree 5 --nu 0.005
