"""
Empirical verification of Claim #3 for the MULTIQUADRIC kernel, using the
user's own FastKANLayer implementation verbatim.

Derived result (see chat): for phi(r;eps) = sqrt(1+eps^2 r^2),
    L_phi(eps) = eps        (sup |phi'(r)| as r -> infinity, approached not attained)

This script:
  1. Re-verifies L_phi(eps) = eps numerically (finite-difference over large r).
  2. Sweeps eps for a fixed synthetic set of centers and measures kappa(A) of
     the multiquadric Gram matrix (conditioning vs shape parameter).
  3. Trains the user's actual FastKANLayer/KANHead (kernel_type='multiquadric')
     end to end at several eps values, and measures: theoretical bound,
     empirically measured Lipschitz constant, mean conditioning of the trained
     centers, and test accuracy -- exactly parallel to the Gaussian experiment.

NOTE ON DATASET: MNIST could not be downloaded in this sandbox (network
egress here is restricted to a fixed allow-list that does not include
yann.lecun.com or the torchvision S3 mirror). This script substitutes
sklearn's `digits` dataset (8x8 images, 10 classes) so the experiment can
actually run and produce real numbers. The FastKANLayer class is used
completely unmodified from the user's code; only KANHead's hard-coded
320/50/10 dimensions are parameterized so they fit the 64-d digits input.
A companion script (kkan_multiquadric_mnist_local.py) reproduces this exact
sweep against the user's original MNIST pipeline, to be run locally.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)
np.random.seed(0)


# ----------------------------------------------------------------------
# Part 1: closed-form vs empirical Lipschitz constant for multiquadric
# ----------------------------------------------------------------------
def phi_mq(r, eps):
    return np.sqrt(1 + (eps ** 2) * (r ** 2))


def analytic_L_mq(eps):
    return eps  # derived: sup|phi'(r)| = eps, approached as r -> infinity


def empirical_L_mq(eps, r_max_factor=2000, n=500000):
    r = np.linspace(0, r_max_factor / eps, n)
    vals = phi_mq(r, eps)
    dphi = np.diff(vals) / np.diff(r)
    return np.max(np.abs(dphi))


print("=" * 70)
print("PART 1: Closed-form vs empirical Lipschitz constant, multiquadric")
print("=" * 70)
print(f"{'eps':>8} | {'L_phi analytic':>16} | {'L_phi empirical':>16} | {'rel. error':>10}")
for eps in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    la, le = analytic_L_mq(eps), empirical_L_mq(eps)
    print(f"{eps:8.2f} | {la:16.6f} | {le:16.6f} | {abs(la-le)/la:10.2e}")


# ----------------------------------------------------------------------
# Part 2: conditioning of the multiquadric Gram matrix vs eps
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: kappa(A) for multiquadric Gram matrix, fixed synthetic centers")
print("=" * 70)

rng = np.random.default_rng(0)
centers_1d = np.sort(rng.uniform(-1, 1, size=25))
eps_sweep = np.logspace(-2, 1.2, 40)
cond_numbers_mq = []
for eps in eps_sweep:
    diff = centers_1d[:, None] - centers_1d[None, :]
    A = phi_mq(diff, eps)
    cond_numbers_mq.append(np.linalg.cond(A))
cond_numbers_mq = np.array(cond_numbers_mq)

for eps, k in list(zip(eps_sweep, cond_numbers_mq))[::5]:
    print(f"eps={eps:8.4f}   kappa(A)={k:14.4e}")


# ----------------------------------------------------------------------
# Part 3: end-to-end -- user's ACTUAL FastKANLayer, kernel_type='multiquadric'
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART 3: Trained FastKANLayer (multiquadric) -- theory vs measured")
print("=" * 70)


# ---- verbatim from the user's script -----------------------------------
class FastKANLayer(nn.Module):
    """Single FastKAN layer with selectable RBF or Multiquadric kernel."""
    def __init__(self, input_dim, output_dim, num_centers=8,
                 kernel_type='rbf', sigma=1.0, epsilon=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.multiquadric_c = epsilon

        self.centers = nn.Parameter(torch.randn(input_dim, num_centers) * 0.5)
        self.weights = nn.Parameter(torch.randn(output_dim, input_dim * num_centers) * 0.01)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.amplitudes = nn.Parameter(torch.ones(input_dim, num_centers) * 0.1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def _rbf(self, x, centers):
        diff = x.unsqueeze(-1) - centers.unsqueeze(0)
        return torch.exp(-(diff ** 2) / (2 * self.sigma ** 2))

    def _multiquadric(self, x, centers):
        diff = x.unsqueeze(-1) - centers.unsqueeze(0)
        return torch.sqrt(1 + (self.multiquadric_c ** 2) * (diff ** 2))

    def forward(self, x):
        x_norm = self.layer_norm(x)
        if self.kernel_type == 'rbf':
            basis = self._rbf(x_norm, self.centers)
        elif self.kernel_type == 'multiquadric':
            basis = self._multiquadric(x_norm, self.centers)
        else:
            raise ValueError(f"Unknown kernel_type '{self.kernel_type}'.")
        basis = basis * self.amplitudes.unsqueeze(0)
        flat = basis.view(x.shape[0], -1)
        return torch.matmul(flat, self.weights.T) + self.bias

    def lambda_l1_sum(self):
        # sum_p ||lambda_pq||_1 per output q, matching the theorem's notation
        return self.weights.abs().view(
            self.output_dim, self.input_dim, self.num_centers
        ).sum(dim=2).sum(dim=1)


class KANHead(nn.Module):
    """Parameterized version of the user's KANHead (dims made configurable
    so it fits the 64-d digits input instead of hard-coded 320/50/10;
    FastKANLayer itself is untouched)."""
    def __init__(self, in_dim=320, hidden_dim=50, num_classes=10,
                 num_centers=8, epsilon=1.0, activation=None):
        super().__init__()
        self.act = activation if activation is not None else nn.ReLU()
        self.kan1 = FastKANLayer(in_dim, hidden_dim, num_centers=num_centers,
                                  kernel_type='multiquadric', epsilon=epsilon)
        self.kan2 = FastKANLayer(hidden_dim, num_classes, num_centers=num_centers,
                                  kernel_type='multiquadric', epsilon=epsilon)

    def forward(self, x):
        x = self.act(self.kan1(x))
        return self.kan2(x)
# --------------------------------------------------------------------------


digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)
scaler = StandardScaler()
X_train = np.tanh(scaler.fit_transform(X_train))
X_test = np.tanh(scaler.transform(X_test))
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

IN_DIM = X_train_t.shape[1]
HIDDEN = 16
NUM_CLASSES = 10
NUM_CENTERS = 8
EPOCHS = 60
LR = 1e-2
BATCH_SIZE = 64


def train_and_measure(eps, seed=0):
    torch.manual_seed(seed)
    model = KANHead(in_dim=IN_DIM, hidden_dim=HIDDEN, num_classes=NUM_CLASSES,
                     num_centers=NUM_CENTERS, epsilon=eps)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = X_train_t.shape[0]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()

    # theoretical bound: Lip(F) <= L_Phi * L_phi(eps) * sum||lambda_pq||_1 (per layer, composed)
    L_phi = analytic_L_mq(eps)
    l1_layer1 = model.kan1.lambda_l1_sum().max().item()
    l1_layer2 = model.kan2.lambda_l1_sum().max().item()
    theoretical_bound = (L_phi * l1_layer1) * (L_phi * l1_layer2)

    with torch.no_grad():
        n_samples = 2000
        idx_a = torch.randint(0, X_test_t.shape[0], (n_samples,))
        x_a = X_test_t[idx_a]
        delta = torch.randn_like(x_a) * 0.01
        x_b = x_a + delta
        num = (model(x_a) - model(x_b)).norm(dim=1)
        den = delta.norm(dim=1).clamp_min(1e-8)
        empirical_lip = (num / den).max().item()

    with torch.no_grad():
        c = model.kan1.centers.detach().numpy()
        conds = []
        for p in range(c.shape[0]):
            cc = c[p]
            diff = cc[:, None] - cc[None, :]
            A = phi_mq(diff, eps)
            conds.append(np.linalg.cond(A))
        mean_cond = float(np.mean(conds))

    return {"eps": eps, "test_acc": test_acc, "theoretical_bound": theoretical_bound,
            "empirical_lip": empirical_lip, "mean_cond": mean_cond}


eps_grid = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
records = []
for eps in eps_grid:
    r = train_and_measure(eps)
    records.append(r)
    print(f"eps={eps:5.2f}  test_acc={r['test_acc']:.4f}  "
          f"theor_bound={r['theoretical_bound']:.3e}  "
          f"emp_Lipschitz={r['empirical_lip']:.4f}  "
          f"mean_cond(A)={r['mean_cond']:.3e}")

# ----------------------------------------------------------------------
# Save plots + csv
# ----------------------------------------------------------------------
import csv
with open("/home/claude/multiquadric_stability_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["eps", "test_acc", "theoretical_bound", "empirical_lipschitz", "mean_cond_A"])
    for r in records:
        writer.writerow([r["eps"], r["test_acc"], r["theoretical_bound"],
                          r["empirical_lip"], r["mean_cond"]])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

eps_fine = np.logspace(-1.3, 1, 30)
la_vals = [analytic_L_mq(e) for e in eps_fine]
le_vals = [empirical_L_mq(e) for e in eps_fine]
axes[0, 0].plot(eps_fine, la_vals, label="Analytic: L_phi(eps)=eps", lw=2)
axes[0, 0].plot(eps_fine, le_vals, "--", label="Empirical (finite diff.)", lw=2)
axes[0, 0].set_xscale("log")
axes[0, 0].set_xlabel("Shape parameter eps")
axes[0, 0].set_ylabel("L_phi(eps)")
axes[0, 0].set_title("Multiquadric: kernel Lipschitz constant vs eps")
axes[0, 0].legend()

axes[0, 1].plot(eps_sweep, cond_numbers_mq)
axes[0, 1].set_xscale("log")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlabel("Shape parameter eps")
axes[0, 1].set_ylabel("kappa(A)")
axes[0, 1].set_title("Multiquadric Gram matrix conditioning vs eps")

eps_arr = [r["eps"] for r in records]
theor = [r["theoretical_bound"] for r in records]
emp = [r["empirical_lip"] for r in records]
ax2 = axes[1, 0]
ax2.plot(eps_arr, theor, "o-", label="Theoretical bound")
ax2.plot(eps_arr, emp, "s-", label="Empirical Lipschitz (measured)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Shape parameter eps")
ax2.set_ylabel("Lipschitz value")
ax2.set_title("Trained multiquadric FastKANLayer: theory vs measured")
ax2.legend()

ax3 = axes[1, 1]
acc_arr = [r["test_acc"] for r in records]
cond_arr = [r["mean_cond"] for r in records]
ax3.plot(eps_arr, acc_arr, "o-", color="tab:green", label="Test accuracy")
ax3.set_xscale("log")
ax3.set_xlabel("Shape parameter eps")
ax3.set_ylabel("Test accuracy", color="tab:green")
ax3.tick_params(axis="y", labelcolor="tab:green")
ax3b = ax3.twinx()
ax3b.plot(eps_arr, cond_arr, "s--", color="tab:red", label="mean kappa(A)")
ax3b.set_yscale("log")
ax3b.set_ylabel("mean kappa(A)", color="tab:red")
ax3b.tick_params(axis="y", labelcolor="tab:red")
ax3.set_title("Multiquadric: accuracy vs conditioning trade-off")

plt.tight_layout()
plt.savefig("/home/claude/multiquadric_stability_plot.png", dpi=150)
print("\nSaved multiquadric_stability_plot.png and multiquadric_stability_results.csv")
