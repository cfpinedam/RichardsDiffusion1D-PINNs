# RichardsDiffusion1D-PINNs

This repository provides a **minimal, functional implementation** of a
Physics-Informed Neural Network (PINN) for solving the **1D diffusion form of
the Richards equation**, using the Hall mortar experiment as a reference case.

The goal is to offer a clear and reproducible starting point for PINN-based
simulations of unsaturated flow in porous media. More experiments and models
will be added over time.

---

## Overview

The main training pipeline is implemented in:

[`train_hall_mortar.py`](experiments/hall_mortar/train_hall_mortar.py)

This script contains all components necessary to train a PINN for the 1D
Richards diffusion equation:

- Hydraulic diffusivity  
  \[
  D(\theta) = 247.1 \, \theta^4
  \]
- Fully connected neural network with 5 hidden layers of 50 neurons (tanh)
- Interior sampling via **Latin Hypercube Sampling (LHS)**
- PDE residual computed using **automatic differentiation**
- Initial and boundary condition enforcement
- **Adaptive loss balancing** for all physics terms
- Two-stage optimization:
  1. **Adam** (mini-batch)
  2. **L-BFGS** (full-batch)

The script prints loss values, updates adaptive weights, and performs
full-batch refinement with L-BFGS.

---

## Running the experiment

### 1. Clone the repository

```bash
git clone https://github.com/cfpinedam/RichardsDiffusion1D-PINNs.git
cd RichardsDiffusion1D-PINNs
