import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import torch.nn.functional as F
import time
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ─────────────────────────────────────────────
# Device and Hyperparameters
# ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
NUM_EPOCHS    = 25
NUM_CLASSES =10
# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = DataLoader(
    datasets.MNIST('./data', train=True,  download=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000, shuffle=False
)


# ═════════════════════════════════════════════
# Building Blocks
# ═════════════════════════════════════════════

# ── 1. Pooling options ────────────────────────
class MaxPool2D(nn.Module):
    """Standard 2×2 max pooling (drop-in for FrequencyPooling2D)."""
    def __init__(self, pool_size=(2, 2), **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        return self.pool(x)


class BlurPool2D(nn.Module):
    """
    Anti-aliased downsampling layer (BlurPool) that dynamically adapts to input channels.
    """

    def __init__(self, pool_size=(2, 2), filt_size=5, stride=None, **kwargs):
        """
        Parameters
        ----------
        pool_size : tuple or int
            Downsampling factor (e.g., (2, 2) or 2)
        filt_size : int
            Kernel size (3 or 5)
        stride : int or None
            If None, uses pool_size as stride
        """
        super().__init__()

        # Handle stride
        if stride is None:
            if isinstance(pool_size, tuple):
                self.stride = pool_size[0]
            else:
                self.stride = pool_size
        else:
            self.stride = stride

        self.filt_size = filt_size

        # Create blur kernel (1D)
        if filt_size == 3:
            kernel = torch.tensor([1., 2., 1.])
        elif filt_size == 5:
            kernel = torch.tensor([1., 4., 6., 4., 1.])
        else:
            raise ValueError("filt_size must be 3 or 5")

        # Create 2D kernel
        kernel_2d = kernel[:, None] * kernel[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()

        # Register base kernel (without channel dimension)
        self.register_buffer('base_kernel', kernel_2d[None, None, :, :])
        self.pad = (filt_size - 1) // 2

    def forward(self, x):
        # Get input channels dynamically
        channels = x.shape[1]

        # Create kernel for exact number of input channels
        # Shape: [channels, 1, filt_size, filt_size]
        kernel = self.base_kernel.repeat(channels, 1, 1, 1)

        # Apply depthwise convolution (groups = channels)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=self.pad,
            groups=channels  # This ensures each channel is processed separately
        )
        return x


class FrequencyPooling2D(nn.Module):
    """FFT-based low-frequency pooling followed by average pooling."""
    def __init__(self, pool_size=(2, 2), keep_low_freq_ratio=0.5):
        super().__init__()
        self.pool_size            = pool_size
        self.keep_low_freq_ratio  = keep_low_freq_ratio

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_fft      = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
        mask       = torch.zeros_like(x_fft)
        ch, cw     = height // 2, width // 2
        hk         = int(height * self.keep_low_freq_ratio / 2)
        wk         = int(width  * self.keep_low_freq_ratio / 2)
        mask[:, :, ch - hk:ch + hk, cw - wk:cw + wk] = 1.0
        x_filtered = torch.real(torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask, dim=(-2, -1))))
        return nn.functional.avg_pool2d(x_filtered, kernel_size=self.pool_size, stride=self.pool_size)


# ── 2. Classifier (head) options ─────────────

class FCHead(nn.Module):
    """Traditional fully-connected head: 320 → 50 → 10."""
    def __init__(self, activation=None,**kwargs):
        super().__init__()
        act = activation if activation is not None else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(320, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)

class WideFCHead(nn.Module):
    """Parameter-matched control: same Linear+Tanh structure as FCHead,
    but widened to roughly match KANHead(num_centers=2)'s parameter count."""
    def __init__(self, activation=None, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(320, 211),
            nn.Tanh(),
            nn.Linear(211, 148),
            nn.Tanh(),
            nn.Linear(148, 10)
        )

    def forward(self, x):
        return self.net(x)


class FastKANLayer(nn.Module):
    """Single FastKAN layer with selectable RBF or Multiquadric kernel."""
    def __init__(self, input_dim, output_dim, num_centers=8,
                 kernel_type='rbf', sigma=1.0, epsilon=1.0 ):
        super().__init__()
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.num_centers = num_centers
        self.kernel_type = kernel_type
        self.sigma       = sigma
        self.multiquadric_c = epsilon #2 / math.sqrt(num_centers)

        self.centers    = nn.Parameter(torch.randn(input_dim, num_centers) * 0.5)
        self.weights    = nn.Parameter(torch.randn(output_dim, input_dim * num_centers) * 0.01)
        self.bias       = nn.Parameter(torch.zeros(output_dim))
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
            raise ValueError(f"Unknown kernel_type '{self.kernel_type}'. Use 'rbf' or 'multiquadric'.")
        basis = basis * self.amplitudes.unsqueeze(0)
        flat  = basis.view(x.shape[0], -1)
        return torch.matmul(flat, self.weights.T) + self.bias

    def repulsion_loss(self):
        # Penalize centers within the same input dimension for being close
        # together: sum over pairs of 1/(dist^2 + tiny), averaged.
        c = self.centers  # (input_dim, num_centers)
        diff = c.unsqueeze(2) - c.unsqueeze(1)          # (input_dim, N, N)
        dist2 = diff ** 2
        mask = ~torch.eye(self.num_centers, dtype=torch.bool, device=c.device)
        penalty = (1.0 / (dist2 + 1e-3))[:, mask.nonzero(as_tuple=True)[0], mask.nonzero(as_tuple=True)[1]] \
            if False else None
        # simpler: use mask directly
        eye = torch.eye(self.num_centers, device=c.device, dtype=torch.bool)
        dist2_masked = dist2.masked_fill(eye.unsqueeze(0), float('inf'))
        return (1.0 / (dist2_masked + 1e-3)).mean()

    def l1_weight_loss(self):
        return self.weights.abs().sum()

class KANHead(nn.Module):
    """FastKAN head: 320 → 50 (RBF) → 10 (Multiquadric)."""
    def __init__(self, num_centers=8, **kwargs):
        super().__init__()
        self.kernel_type = kwargs.get('kernel_type', 'multiquadric')
        e= 2 / math.sqrt(num_centers)
        self.ep = kwargs.get('ep', e)  # epsilon for multiquadric
        self.act = kwargs.get('activation', nn.Tanh())
        self.kan1 = FastKANLayer(320, 120,  num_centers=num_centers, kernel_type=self.kernel_type,epsilon=self.ep)
        self.kan2 = FastKANLayer(120,  84,  num_centers=num_centers, kernel_type=self.kernel_type,epsilon=self.ep)
        self.kan3 = FastKANLayer(84,  10,  num_centers=num_centers, kernel_type=self.kernel_type,epsilon=self.ep)
    def forward(self, x):
        #x = torch.relu(self.kan1(x))
        x = self.act(self.kan1(x))
        x = self.act(self.kan2(x))
        ##x = self.kan1(x)
        ##x = self.kan2(x)
        return self.kan3(x)
    def repulsion_loss(self):
        return self.kan1.repulsion_loss() + self.kan2.repulsion_loss()

    def l1_weight_loss(self):
        return self.kan1.l1_weight_loss() + self.kan2.l1_weight_loss()

# ═════════════════════════════════════════════
# Unified CNN model  ← swap pooling & head here
# ═════════════════════════════════════════════

class CNNClassifier(nn.Module):
    """
    Two-conv CNN whose pooling and classifier head are fully swappable.

    Parameters
    ----------
    pooling_cls : class
        MaxPool2D  (default) or FrequencyPooling2D or BlurPool2D
    head_cls    : class
        FCHead     (default) or KANHead
    pooling_kwargs / head_kwargs : dict
        Extra kwargs forwarded to the chosen class constructors.

    Quick-swap examples
    -------------------
    # Standard CNN  (max-pool + FC):
        model = CNNClassifier()

    # KAN head, max pooling:
        model = CNNClassifier(head_cls=KANHead, head_kwargs={'num_centers': 8})

    # Frequency pooling + FC:
        model = CNNClassifier(pooling_cls=FrequencyPooling2D,
                              pooling_kwargs={'keep_low_freq_ratio': 0.5})

    # Frequency pooling + KAN:
        model = CNNClassifier(pooling_cls=FrequencyPooling2D,
                              head_cls=KANHead)
    """
    def __init__(self,
                 pooling_cls=MaxPool2D,    pooling_kwargs=None,
                 head_cls=FCHead,          head_kwargs=None, 
                 activation = None):
        super().__init__()
        pooling_kwargs = pooling_kwargs or {}
        head_kwargs    = head_kwargs    or {}

        self.act    = activation if activation is not None else nn.ReLU()
        self.conv1  = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1  = pooling_cls(pool_size=(2, 2), **pooling_kwargs)
        self.conv2  = nn.Conv2d(10, 20, kernel_size=5)
        self.pool2  = pooling_cls(pool_size=(2, 2), **pooling_kwargs)
        self.head   = head_cls(**head_kwargs)

    def forward(self, x):
        #x = torch.relu(self.conv1(x))
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        #x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)   # flatten → 320
        return self.head(x)


# ═════════════════════════════════════════════
# Training / Evaluation helpers
# ═════════════════════════════════════════════
"""
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss   = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += output.argmax(1).eq(target).sum().item()
    return total_loss / len(loader), 100. * correct / len(loader.dataset)
"""
def train_one_epoch(model, loader, optimizer, criterion, beta_repulsion=0.0, alpha_l1=0.0):
    model.train()
    total_loss, correct = 0.0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if beta_repulsion > 0 and hasattr(model.head, 'repulsion_loss'):
            loss = loss + beta_repulsion * model.head.repulsion_loss()
        if alpha_l1 > 0 and hasattr(model.head, 'l1_weight_loss'):
            loss = loss + alpha_l1 * model.head.l1_weight_loss()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
    return total_loss / len(loader), 100. * correct / len(loader.dataset)



@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output     = model(data)
        total_loss += criterion(output, target).item()
        correct    += output.argmax(1).eq(target).sum().item()
    return total_loss / len(loader), 100. * correct / len(loader.dataset)




@torch.no_grad()
def get_roc_data(model, loader, num_classes=NUM_CLASSES):
    model.eval()
    all_probs, all_labels = [], []
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        probs = torch.softmax(output, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(target.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    y_true_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average (pools all classes' TP/FP counts together)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro-average (averages the per-class curves — treats each digit equally)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def plot_roc_comparison(roc_results: dict):
    """roc_results = {model_name: (fpr, tpr, roc_auc), ...}"""
    plt.figure(figsize=(7, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, (name, (fpr, tpr, roc_auc)) in enumerate(roc_results.items()):
        plt.plot(fpr["macro"], tpr["macro"], color=colors[idx % len(colors)],
                  label=f"{name} (macro AUC = {roc_auc['macro']:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Chance')
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Macro-average ROC Curve Comparison')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_comparison.png', dpi=150)
    plt.show()
    print("Saved roc_comparison.png")


def plot_roc_per_class(fpr, tpr, roc_auc, model_name, num_classes=NUM_CLASSES):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for i, ax in enumerate(axes.flat):
        ax.plot(fpr[i], tpr[i], label=f"AUC={roc_auc[i]:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_title(f"Digit {i}")
        ax.legend(fontsize=7, loc='lower right')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    plt.suptitle(f"Per-class ROC — {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'roc_per_class_{model_name.replace(" ", "_")}.png', dpi=150)
    plt.show()


@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = output.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(target.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)

def plot_confusion_matrix(y_true, y_pred, model_name, num_classes=NUM_CLASSES):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=range(num_classes), yticklabels=range(num_classes))
    axes[0].set_title(f'{model_name} — Raw Counts')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=range(num_classes), yticklabels=range(num_classes),
                vmin=0, vmax=1)
    axes[1].set_title(f'{model_name} — Row-Normalized (Recall per class)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.suptitle(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f'confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved {fname}")
    return cm


def plot_per_class_accuracy_comparison(results_cm: dict, num_classes=NUM_CLASSES):
    """results_cm = {model_name: cm, ...} from plot_confusion_matrix's return value"""
    plt.figure(figsize=(10, 5))
    x = np.arange(num_classes)
    width = 0.8 / len(results_cm)
    for idx, (name, cm) in enumerate(results_cm.items()):
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        plt.bar(x + idx * width, per_class_acc, width, label=name)
    plt.xlabel('Digit')
    plt.ylabel('Per-class accuracy (recall)')
    plt.title('Per-class Accuracy Comparison')
    plt.xticks(x + width * (len(results_cm) - 1) / 2, x)
    plt.legend(fontsize=8)
    plt.ylim([0, 1.05])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('per_class_accuracy_comparison.png', dpi=150)
    plt.show()
'''
def run_experiment(model, label):
    """Train a model for NUM_EPOCHS and return history dict."""
    #optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    history   = {'train_loss': [], 'test_loss': [],
                 'train_acc':  [], 'test_acc':  []}

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)

        print(f"  Epoch {epoch}/{NUM_EPOCHS}  |  "
              f"Train Acc={tr_acc:.2f}%  Test Acc={te_acc:.2f}%  |  "
              f"Train Loss={tr_loss:.4f}  Test Loss={te_loss:.4f}")

    print(f"\n  ► Final Test Acc : {history['test_acc'][-1]:.2f}%")
    print(f"  ► Final Test Loss: {history['test_loss'][-1]:.4f}")
    return history
'''

def run_experiment(model, label, beta_repulsion=0.0, alpha_l1=0.0):
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'epoch_time': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion,
                                           beta_repulsion=beta_repulsion, alpha_l1=alpha_l1)
        epoch_time = time.perf_counter() - t0
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['epoch_time'].append(epoch_time)

        print(f"  Epoch {epoch}/{NUM_EPOCHS}  Train Acc={tr_acc:.2f}%  Test Acc={te_acc:.2f}%  "
              f"Time={epoch_time:.2f}s")
    return history

def convergence_metrics(history, thresholds=(80, 90, 95)):
    """Compute epochs-to-threshold, wall-clock-to-threshold, and accuracy-curve AUC."""
    acc = history['test_acc']
    times = history['epoch_time']
    cum_time = [sum(times[:i+1]) for i in range(len(times))]

    result = {}
    for t in thresholds:
        epoch_idx = next((i for i, a in enumerate(acc) if a >= t), None)
        result[f'epochs_to_{t}%'] = (epoch_idx + 1) if epoch_idx is not None else None
        result[f'seconds_to_{t}%'] = cum_time[epoch_idx] if epoch_idx is not None else None

    # AUC of accuracy-vs-epoch, normalized by number of epochs so it's comparable
    # across runs with different NUM_EPOCHS
    auc = sum((acc[i] + acc[i+1]) / 2 for i in range(len(acc)-1)) / max(len(acc)-1, 1)
    result['accuracy_auc'] = auc
    result['final_acc'] = acc[-1]
    result['total_time_s'] = cum_time[-1]
    return result

# ═════════════════════════════════════════════
# Plot helper
# ═════════════════════════════════════════════

def plot_results(results: dict):
    """
    results = { 'Model Name': history_dict, ... }
    Plots accuracy and loss for all supplied models on shared axes.
    """
    epochs = range(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (label, hist) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        axes[0].plot(epochs, hist['train_acc'], color=c, linestyle='-', 
                     label=f'{label} Train')
        axes[0].plot(epochs, hist['test_acc'],  color=c, linestyle='--',
                     label=f'{label} Test')

        axes[1].plot(epochs, hist['train_loss'], color=c, linestyle='-',
                     label=f'{label} Train')
        axes[1].plot(epochs, hist['test_loss'],  color=c, linestyle='--',
                     label=f'{label} Test')

    for ax, ylabel, title in zip(
        axes,
        ['Accuracy (%)', 'Loss'],
        ['Accuracy over Epochs', 'Loss over Epochs']
    ):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()
    print("\nPlot saved to results.png")


# ═════════════════════════════════════════════
# ► CONFIGURE YOUR EXPERIMENTS HERE ◄
# ═════════════════════════════════════════════

experiments = {

        # ── ReLU baseline ─────────────────────────────────────────────────
    #"CNN (ReLU)": CNNClassifier(pooling_cls=FrequencyPooling2D,
    #                            pooling_kwargs={'keep_low_freq_ratio': 0.5}, 
    #                            head_cls=KANHead,
    #                            activation=nn.Tanh(),
    #                            head_kwargs={'num_centers': 2, 'kernel_type': 'multiquadric', 'activation':nn.Tanh()}  ),
 
    # ── GELU swap — change just the activation ────────────────────────
#    "LeNet (KNN_mq +  GELU)": CNNClassifier(pooling_cls=BlurPool2D,
#                                pooling_kwargs={'keep_low_freq_ratio': 0.5},
#                                head_cls=KANHead,
#                                activation=nn.GELU(),
#                                head_kwargs={'num_centers': 2, 'kernel_type': 'multiquadric', 'activation': nn.GELU()} ),
    
    #"LeNet (KNN_rbf + ReLU)": CNNClassifier(pooling_cls=BlurPool2D,
    #                            pooling_kwargs={'keep_low_freq_ratio': 0.5},
    #                            head_cls=KANHead,
    #                            activation=nn.ReLU(),
    #                            head_kwargs={'num_centers': 2, 'kernel_type': 'rbf', 'activation': nn.ReLU()} ),
    

    # ── Standard CNN (max pool + FC) ──────────────────────────────────
    "LeNet ": CNNClassifier( pooling_cls=BlurPool2D, head_cls=FCHead),
    "LeNet (MaxPool + WideFC)":  CNNClassifier(pooling_cls=MaxPool2D, head_cls=WideFCHead),
    # ── KAN head, max pooling ─────────────────────────────────────────
    # Uncomment to add this experiment:
    #"CNN (MaxPool + MQ-KAN)": CNNClassifier( pooling_cls=MaxPool2D, head_cls=KANHead, head_kwargs={'num_centers': 8} ),

    # ── Frequency pooling + FC ────────────────────────────────────────
    #"CNN (FreqPool + FC)": CNNClassifier( pooling_cls=FrequencyPooling2D, pooling_kwargs={'keep_low_freq_ratio': 0.5}, head_cls=FCHead ),

    # ── Frequency pooling + KAN ───────────────────────────────────────
    #"CNN (FreqPool + MQ-KAN)": CNNClassifier( pooling_cls=FrequencyPooling2D, pooling_kwargs={'keep_low_freq_ratio': 0.5}, head_cls=KANHead, head_kwargs={'num_centers': 2, 'kernel_type': 'multiquadric'} ),

    #"CNN (FreqPool + RBF KAN)": CNNClassifier( pooling_cls=FreqsuencyPooling2D, pooling_kwargs={'keep_low_freq_ratio': 0.5}, head_cls=KANHead, head_kwargs={'num_centers': 4, 'kernel_type': 'rbf'} ),
     
    # ── Frequency pooling + KAN ───────────────────────────────────────
     "MQ-KAN": CNNClassifier(
         pooling_cls=BlurPool2D,
         pooling_kwargs={'keep_low_freq_ratio': 0.5},
         head_cls=KANHead,
         head_kwargs={'num_centers': 2, 'kernel_type': 'multiquadric', 'ep': .9}
     ),
     #"CNN (BlurPool + MQ-KAN(4))": CNNClassifier(
     #         pooling_cls=FrequencyPooling2D,
     #         pooling_kwargs={'keep_low_freq_ratio': 0.5},
     #         head_cls=KANHead,
     #         head_kwargs={'num_centers': 2, 'kernel_type': 'multiquadric', 'ep': .9}
     #     ),

#     "CNN (FreqPool + MQ- KAN(epsilon=0.95))": CNNClassifier(
#         pooling_cls=FrequencyPooling2D,
#         pooling_kwargs={'keep_low_freq_ratio': 0.5},
#         head_cls=KANHead,
#         head_kwargs={'num_centers': 4, 'kernel_type': 'multiquadric', 'ep': 0.95}
#     ), 
     
    # ── Frequency pooling + KAN ───────────────────────────────────────
#     "CNN (FreqPool + MQ-KAN(epsilon=1.0))": CNNClassifier(
#         pooling_cls=FrequencyPooling2D,
#         pooling_kwargs={'keep_low_freq_ratio': 0.5},
#         head_cls=KANHead,
#         head_kwargs={'num_centers': 4, 'kernel_type': 'multiquadric', 'ep': 1.0}
#     ),

#     "CNN (FreqPool + MQ KAN(epsilon=1.05))": CNNClassifier(
#         pooling_cls=FrequencyPooling2D,
#         pooling_kwargs={'keep_low_freq_ratio': 0.5},
#         head_cls=KANHead,
#         head_kwargs={'num_centers': 4, 'kernel_type': 'multiquadric', 'ep': 1.05}
#     ), 
     
#     "CNN (FreqPool + MQ-KAN (epsilon=1.1))": CNNClassifier(
#         pooling_cls=FrequencyPooling2D,
#         pooling_kwargs={'keep_low_freq_ratio': 0.5},
#         head_cls=KANHead,
#         head_kwargs={'num_centers': 4, 'kernel_type': 'multiquadric', 'ep': 1.1}
#     ), 


}

# Move all models to device
for name, model in experiments.items():
    experiments[name] = model.to(device)

# ─────────────────────────────────────────────
# Run all experiments and plot
# ─────────────────────────────────────────────
results = {}
for name, model in experiments.items():
    results[name] = run_experiment(model, name)
    

for name, model in experiments.items():
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(name, ": total_params= ", total_params, ", trainable_params= ",trainable_params)

    

plot_results(results)
print(f"\n{'Config':<30} {'Ep→90%':<10} {'Sec→90%':<12} {'AUC':<8} {'Final Acc':<10} {'Total Time':<10}")
for name, hist in results.items():
    m = convergence_metrics(hist)
    print(f"{name:<30} {str(m['epochs_to_90%']):<10} {str(m['seconds_to_90%']):<12} "
          f"{m['accuracy_auc']:.2f}    {m['final_acc']:.2f}%     {m['total_time_s']:.1f}s")

roc_results = {}
for name, model in experiments.items():
    fpr, tpr, roc_auc = get_roc_data(model, test_loader)
    roc_results[name] = (fpr, tpr, roc_auc)
    print(f"{name}: macro AUC = {roc_auc['macro']:.4f}, micro AUC = {roc_auc['micro']:.4f}")


plot_roc_comparison(roc_results)

for name, (fpr, tpr, roc_auc) in roc_results.items():
    plot_roc_per_class(fpr, tpr, roc_auc, name)

confusion_matrices = {}
for name, model in experiments.items():
    y_true, y_pred = get_predictions(model, test_loader)
    print(f"\n{'='*60}\n  {name} — classification report\n{'='*60}")
    print(classification_report(y_true, y_pred, digits=4))
    cm = plot_confusion_matrix(y_true, y_pred, name)
    confusion_matrices[name] = cm

plot_per_class_accuracy_comparison(confusion_matrices)