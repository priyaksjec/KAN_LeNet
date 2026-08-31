# KAN-LeNet: Stability & Performance of Kolmogorov–Arnold Layers vs. Standard CNN Heads

This repo compares a classic LeNet-style CNN classifier head (fully-connected)
against a **FastKAN** head — a Kolmogorov–Arnold Network layer using RBF or
multiquadric radial basis kernels — on MNIST / `sklearn.digits`. It also
includes a standalone numerical study of the multiquadric kernel's
**Lipschitz stability** as a function of its shape parameter `epsilon`,
which motivates why `epsilon` needs to be chosen carefully when using the
multiquadric kernel in a KAN layer.

## What's in here

| File | Description |
|---|---|
| `src/kan_lenet_mnist.py` | Full LeNet-style CNN pipeline on MNIST with swappable pooling (`MaxPool2D`, `BlurPool2D`, `FrequencyPooling2D`) and swappable classifier heads (`FCHead`, `WideFCHead`, `KANHead` with RBF or multiquadric kernels). Trains multiple configurations, then produces accuracy/loss curves, ROC curves (macro/micro/per-class), confusion matrices, and per-class accuracy comparisons. |
| `src/multiquadric_stability_experiment.py` | Verifies the closed-form Lipschitz constant `L_phi(epsilon) = epsilon` for the multiquadric kernel `phi(r; eps) = sqrt(1 + eps^2 r^2)` against a finite-difference estimate, sweeps the conditioning `kappa(A)` of the multiquadric Gram matrix vs. `epsilon`, and trains the actual `FastKANLayer`/`KANHead` from this repo end-to-end at several `epsilon` values to compare the theoretical Lipschitz bound against the empirically measured one, alongside test accuracy and center conditioning. |

## Background

A **FastKAN layer** replaces a standard `Linear` layer's dot product with a
learned combination of radial basis functions centered at learnable points
per input dimension. Two kernel choices are implemented:

- **RBF (Gaussian):** `phi(r) = exp(-r^2 / (2 sigma^2))`
- **Multiquadric:** `phi(r; eps) = sqrt(1 + eps^2 r^2)`

The multiquadric kernel is unbounded and its derivative's supremum grows
linearly with `eps`, i.e. `L_phi(eps) = eps`. This repo empirically confirms
that relationship and studies the accuracy/conditioning trade-off it implies
in a trained network — larger `eps` gives a "sharper" basis but a worse
conditioned Gram matrix and a looser Lipschitz bound on the whole model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the main CNN vs. KAN comparison on MNIST (downloads MNIST automatically
via `torchvision` on first run):

```bash
python src/kan_lenet_mnist.py
```



Run the multiquadric stability study :

```bash
python src/multiquadric_stability_experiment.py
```

This writes `multiquadric_stability_results.csv` and
`multiquadric_stability_plot.png` to the working directory.


