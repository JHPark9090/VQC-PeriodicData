"""
SIREN-TCN: Classical Fourier Baseline for EEG Classification

SIREN (Sitzmann et al., NeurIPS 2020) uses sin(w0 * (Wx + b)) as activation,
making it a natural classical Fourier baseline for VQC comparison.

This implementation mirrors FourierQTCN_EEG.py exactly:
- Same sliding window extraction
- Same FFT frequency extraction (_extract_fft_features)
- Same linear projection
- SIRENBlock replaces quantum circuit
- Same learnable weighted aggregation

Key difference: SIREN processes full batches natively (no per-sample loop).

Design Philosophy:
- Mirror FourierQTCN architecture exactly
- Replace only the quantum circuit with a SIREN network
- Isolate the question: does quantum Fourier give advantage over classical Fourier?

Based on:
- Sitzmann et al. (2020): Implicit Neural Representations with Periodic Activation Functions
- Schuld et al. (2021): VQC as Fourier series
- Ziyin et al. (2020): Neural networks fail to learn periodic functions

Author: SIREN classical baseline for VQC comparison
Date: February 2026
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from typing import Tuple, List, Optional
import math
import os
import copy
import time
import random
import argparse

# Try to import the data loader
try:
    from Load_PhysioNet_EEG import load_eeg_ts_revised
except ImportError:
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from Load_PhysioNet_EEG import load_eeg_ts_revised
    except ImportError:
        print("Warning: Load_PhysioNet_EEG not found. You'll need to provide your own data loader.")

try:
    from dataset_dispatcher import add_dataset_args, load_dataset
except ImportError:
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from dataset_dispatcher import add_dataset_args, load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def get_args():
    parser = argparse.ArgumentParser(description="SIREN-TCN for EEG Classification")
    parser.add_argument("--freq", type=int, default=80, help="Sampling frequency")
    parser.add_argument("--n-sample", type=int, default=50, help="Number of subjects")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint")
    parser.add_argument("--siren-dim", type=int, default=8,
                        help="SIREN input dim (equivalent to n-qubits for comparison)")
    parser.add_argument("--n-siren-layers", type=int, default=2,
                        help="Number of SIREN layers (equivalent to circuit-depth)")
    parser.add_argument("--n-frequencies", type=int, default=None,
                        help="Number of FFT frequencies (default: siren_dim)")
    parser.add_argument("--w0", type=float, default=30.0,
                        help="Initial SIREN frequency (default: 30.0)")
    parser.add_argument("--learnable-w0", action="store_true", default=True,
                        help="Make w0 learnable")
    parser.add_argument("--no-learnable-w0", action="store_false", dest="learnable_w0")
    parser.add_argument("--siren-hidden-dim", type=int, default=64,
                        help="Hidden dimension for SIREN layers")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--kernel-size", type=int, default=12, help="Temporal kernel size")
    parser.add_argument("--dilation", type=int, default=3, help="Dilation factor")
    add_dataset_args(parser)
    return parser.parse_args()


print('Pytorch Version:', torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# SIREN LAYER
# =============================================================================

class SIRENLayer(nn.Module):
    """
    Core SIREN layer: sin(w0 * (Wx + b))

    Initialization follows Sitzmann et al. (2020):
    - First layer:  W ~ Uniform[-1/n, 1/n]
    - Hidden layers: W ~ Uniform[-sqrt(6/n)/w0, sqrt(6/n)/w0]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float = 30.0,
        is_first: bool = False,
        learnable_w0: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first

        # Learnable w0 via softplus reparameterization for stability
        if learnable_w0:
            raw_init = np.log(np.exp(w0) - 1.0)
            self.w0_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        else:
            self.register_buffer('w0_raw', torch.tensor(
                np.log(np.exp(w0) - 1.0), dtype=torch.float32
            ))

        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    @property
    def w0(self) -> torch.Tensor:
        """Get w0 via softplus to ensure positivity."""
        return F.softplus(self.w0_raw)

    def _init_weights(self):
        """Sitzmann initialization for proper gradient flow."""
        with torch.no_grad():
            w0_val = F.softplus(self.w0_raw).item()
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / w0_val
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))


# =============================================================================
# SIREN BLOCK (replaces quantum circuit)
# =============================================================================

class SIRENBlock(nn.Module):
    """
    Multi-layer SIREN network replacing the quantum convolutional + pooling circuit.

    Progressive dimension reduction mimics the conv-pool qubit reduction:
    Input [batch, siren_dim] -> Output [batch] scalar

    The output is unbounded (no tanh) to match BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        w0: float = 30.0,
        learnable_w0: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim

        layers = []

        # First SIREN layer
        layers.append(SIRENLayer(
            input_dim, hidden_dim, w0=w0,
            is_first=True, learnable_w0=learnable_w0
        ))

        # Hidden SIREN layers with progressive reduction
        current_dim = hidden_dim
        for i in range(n_layers - 1):
            next_dim = max(hidden_dim // (2 ** (i + 1)), 8)
            layers.append(SIRENLayer(
                current_dim, next_dim, w0=w0,
                is_first=False, learnable_w0=learnable_w0
            ))
            current_dim = next_dim

        self.siren_layers = nn.ModuleList(layers)

        # Final linear layer -> single scalar output (no activation for BCEWithLogitsLoss)
        self.output_linear = nn.Linear(current_dim, 1)
        with torch.no_grad():
            bound = np.sqrt(6.0 / current_dim) / 30.0
            self.output_linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SIREN block.

        Args:
            x: Input [batch, siren_dim]

        Returns:
            Output [batch] scalar (unbounded for BCEWithLogitsLoss)
        """
        h = x
        for layer in self.siren_layers:
            h = layer(h)
        return self.output_linear(h).squeeze(-1)

    def get_w0_values(self) -> List[float]:
        """Return learned w0 values for analysis."""
        return [layer.w0.item() for layer in self.siren_layers]


# =============================================================================
# SIREN-TCN MODEL (mirrors FourierQTCN)
# =============================================================================

class SIREN_TCN(nn.Module):
    """
    SIREN-based Temporal Convolutional Network for EEG Classification.

    Mirrors FourierQTCN exactly:

    Architecture:
        Input EEG [batch, channels, time]
            |
            v
        Sliding Window Extraction
            |
            v
        FFT-based Frequency Extraction (LOSSLESS)
            |
            v
        Simple Linear Projection
            |
            v
        SIREN Block (replaces quantum circuit)
            |
            v
        Learnable Weighted Aggregation
            |
            v
        Output Prediction
    """

    def __init__(
        self,
        siren_dim: int,
        n_siren_layers: int,
        input_dim: Tuple[int, int, int],
        kernel_size: int,
        dilation: int = 1,
        n_frequencies: int = None,
        w0: float = 30.0,
        learnable_w0: bool = True,
        siren_hidden_dim: int = 64,
        use_magnitude_phase: bool = True
    ):
        """
        Args:
            siren_dim: Input dimension for SIREN (equivalent to n_qubits)
            n_siren_layers: Number of SIREN layers (equivalent to circuit_depth)
            input_dim: (batch_size, n_channels, n_timepoints)
            kernel_size: Temporal convolution kernel size
            dilation: Dilation factor for temporal convolution
            n_frequencies: Number of FFT frequencies (default: siren_dim)
            w0: Initial SIREN frequency
            learnable_w0: Whether w0 is learnable
            siren_hidden_dim: Hidden dimension for SIREN layers
            use_magnitude_phase: If True, use magnitude and phase; else real/imag
        """
        super().__init__()

        self.siren_dim = siren_dim
        self.n_siren_layers = n_siren_layers
        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_magnitude_phase = use_magnitude_phase

        # Number of FFT frequencies to extract
        # Cap at available FFT bins: rfft of kernel_size gives kernel_size//2 + 1 bins
        max_fft_bins = kernel_size // 2 + 1
        requested_freqs = n_frequencies if n_frequencies else siren_dim
        self.n_frequencies = min(requested_freqs, max_fft_bins)

        # =================================================================
        # COMPONENT 1: FFT Feature Dimension Calculation
        # =================================================================
        self.fft_feature_dim = self.input_channels * self.n_frequencies * 2

        # =================================================================
        # COMPONENT 2: Simple Linear Projection
        # =================================================================
        self.projection = nn.Linear(self.fft_feature_dim, siren_dim)

        # =================================================================
        # COMPONENT 3: SIREN Block (replaces quantum circuit)
        # =================================================================
        self.siren_block = SIRENBlock(
            input_dim=siren_dim,
            hidden_dim=siren_hidden_dim,
            n_layers=n_siren_layers,
            w0=w0,
            learnable_w0=learnable_w0
        )

        # =================================================================
        # COMPONENT 4: Simple Weighted Aggregation
        # =================================================================
        max_windows = self.time_steps - self.dilation * (self.kernel_size - 1)
        self.agg_weights = nn.Parameter(torch.zeros(max_windows))

        # Store frequency bins for interpretability
        self.register_buffer(
            'freq_bin_indices',
            torch.arange(self.n_frequencies)
        )

        self._print_config()

    def _print_config(self):
        """Print model configuration."""
        print(f"\n{'='*60}")
        print("SIREN-TCN Initialized (Classical Fourier Baseline)")
        print(f"{'='*60}")
        print(f"SIREN dim (n_qubits equiv): {self.siren_dim}")
        print(f"SIREN layers (circuit_depth equiv): {self.n_siren_layers}")
        print(f"Kernel size: {self.kernel_size}")
        print(f"Dilation: {self.dilation}")
        print(f"Number of FFT frequencies: {self.n_frequencies}")
        print(f"FFT feature dimension: {self.fft_feature_dim}")
        print(f"Use magnitude/phase: {self.use_magnitude_phase}")
        print(f"Initial w0: {self.siren_block.get_w0_values()}")
        print(f"Total parameters: {count_parameters(self)}")
        print(f"{'='*60}\n")

    def _extract_fft_features(self, window: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency-domain features using FFT.

        Identical to FourierQTCN._extract_fft_features.

        Args:
            window: [batch, channels, kernel_size]

        Returns:
            features: [batch, fft_feature_dim]
        """
        batch_size = window.shape[0]

        # Apply FFT along time dimension
        fft_result = torch.fft.rfft(window, dim=-1)

        # Select first n_frequencies bins
        fft_selected = fft_result[:, :, :self.n_frequencies]

        if self.use_magnitude_phase:
            magnitude = torch.abs(fft_selected)
            phase = torch.angle(fft_selected)

            magnitude = torch.log1p(magnitude)
            phase = phase / np.pi

            features = torch.stack([magnitude, phase], dim=-1)
        else:
            features = torch.stack([fft_selected.real, fft_selected.imag], dim=-1)

        features = features.reshape(batch_size, -1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with FFT-based frequency extraction.

        Args:
            x: Input EEG [batch, channels, time_steps]

        Returns:
            output: Predictions [batch]
        """
        batch_size = x.shape[0]
        outputs = []

        # Calculate starting index for sliding window
        effective_start = self.dilation * (self.kernel_size - 1)

        # Process each time window
        for t in range(effective_start, self.time_steps):
            # Get window indices with dilation
            indices = [t - d * self.dilation for d in range(self.kernel_size)]
            indices.reverse()

            # Extract window: [batch, channels, kernel_size]
            window = x[:, :, indices]

            # =================================================================
            # STEP 1: FFT-based frequency extraction (LOSSLESS)
            # =================================================================
            fft_features = self._extract_fft_features(window)

            # =================================================================
            # STEP 2: Simple linear projection
            # =================================================================
            projected = self.projection(fft_features)

            # Normalize to prevent exploding gradients
            projected = torch.tanh(projected) * np.pi

            # =================================================================
            # STEP 3: SIREN block (replaces quantum circuit)
            # SIREN processes full batch natively (no per-sample loop)
            # =================================================================
            window_output = self.siren_block(projected)
            outputs.append(window_output)

        # =================================================================
        # STEP 4: Simple weighted aggregation
        # =================================================================
        outputs = torch.stack(outputs, dim=1)  # [batch, n_windows]
        n_windows = outputs.shape[1]

        # Softmax over learnable weights
        weights = F.softmax(self.agg_weights[:n_windows], dim=0)

        # Weighted sum
        output = (outputs * weights.unsqueeze(0)).sum(dim=1)

        return output

    def get_w0_values(self) -> List[float]:
        """Return learned w0 values for analysis."""
        return self.siren_block.get_w0_values()

    def get_aggregation_weights(self, n_windows: int) -> np.ndarray:
        """Return aggregation weights for analysis."""
        weights = F.softmax(self.agg_weights[:n_windows], dim=0)
        return weights.detach().cpu().numpy()


# =============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Calculate elapsed time in minutes and seconds."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    criterion,
    device,
    clip_grad: float = 1.0,
    task: str = 'classification'
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    all_labels = []
    all_outputs = []

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()
        train_loss += loss.item()

        all_labels.append(labels.cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    if task == 'classification':
        try:
            metric = roc_auc_score(all_labels, all_outputs)
        except ValueError:
            metric = 0.5
    else:
        metric = np.sqrt(np.mean((all_labels - all_outputs) ** 2))

    return train_loss / len(dataloader), metric


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device,
    task: str = 'classification'
) -> Tuple[float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    if task == 'classification':
        try:
            metric = roc_auc_score(all_labels, all_outputs)
        except ValueError:
            metric = 0.5
    else:
        metric = np.sqrt(np.mean((all_labels - all_outputs) ** 2))

    return running_loss / len(dataloader), metric


def run_training(
    seed: int,
    siren_dim: int,
    n_siren_layers: int,
    input_dim: Tuple[int, int, int],
    train_loader,
    val_loader,
    test_loader,
    kernel_size: int = 12,
    dilation: int = 3,
    n_frequencies: int = None,
    w0: float = 30.0,
    learnable_w0: bool = True,
    siren_hidden_dim: int = 64,
    num_epochs: int = 50,
    lr: float = 0.001,
    checkpoint_dir: str = None,
    resume: bool = False,
    args=None,
    task: str = 'classification'
):
    """Run full training pipeline for SIREN-TCN."""
    print(f"\n{'='*60}")
    print("Starting SIREN-TCN Training (Classical Fourier Baseline)")
    print(f"{'='*60}")

    set_all_seeds(seed)
    print(f"Random Seed: {seed}")

    model = SIREN_TCN(
        siren_dim=siren_dim,
        n_siren_layers=n_siren_layers,
        input_dim=input_dim,
        kernel_size=kernel_size,
        dilation=dilation,
        n_frequencies=n_frequencies,
        w0=w0,
        learnable_w0=learnable_w0,
        siren_hidden_dim=siren_hidden_dim
    ).to(device)

    # Task-specific criterion and tracking
    if task == 'classification':
        criterion = nn.BCEWithLogitsLoss()
        sched_mode, metric_name = 'max', 'auc'
        best_metric, is_better = 0.0, lambda new, old: new > old
    else:
        criterion = nn.MSELoss()
        sched_mode, metric_name = 'min', 'rmse'
        best_metric, is_better = float('inf'), lambda new, old: new < old

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_mode, factor=0.5, patience=5, verbose=True
    )

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(RESULTS_DIR, 'SIREN_TCN_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"siren_tcn_s{siren_dim}_l{n_siren_layers}_seed{seed}.pth"
    )

    start_epoch = 0
    train_metrics, valid_metrics = [], []
    best_model_state = None

    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_metrics = checkpoint.get('train_metrics', [])
        valid_metrics = checkpoint.get('valid_metrics', [])
        best_metric = checkpoint.get('best_val_metric', best_metric)
        print(f"Resuming from epoch {start_epoch + 1}, best {metric_name}: {best_metric:.4f}")

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        train_loss, train_m = train_epoch(
            model, train_loader, optimizer, criterion, device, task=task
        )
        train_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            f'train_{metric_name}': train_m
        })

        valid_loss, valid_m = evaluate(model, val_loader, criterion, device, task=task)
        valid_metrics.append({
            'epoch': epoch + 1,
            'valid_loss': valid_loss,
            f'valid_{metric_name}': valid_m
        })

        scheduler.step(valid_m)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if is_better(valid_m, best_metric):
            best_metric = valid_m
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"\nEpoch: {epoch + 1:02}/{num_epochs} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss: {train_loss:.4f}, {metric_name.upper()}: {train_m:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, {metric_name.upper()}: {valid_m:.4f} (Best: {best_metric:.4f})")

        if (epoch + 1) % 10 == 0:
            w0_vals = model.get_w0_values()
            print(f"Learned w0: {[f'{v:.2f}' for v in w0_vals]}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'best_val_metric': best_metric
        }, checkpoint_path)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_metric = evaluate(model, test_loader, criterion, device, task=task)

    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test {metric_name.upper()}: {test_metric:.4f}")
    print(f"Best Validation {metric_name.upper()}: {best_metric:.4f}")
    print(f"Total parameters: {count_parameters(model)}")

    w0_vals = model.get_w0_values()
    print(f"\nLearned w0 Values:")
    for i, val in enumerate(w0_vals):
        print(f"  Layer {i}: {val:.4f}")

    print(f"{'='*60}\n")

    metrics = []
    for i in range(len(train_metrics)):
        metrics.append({
            'epoch': i + 1,
            'train_loss': train_metrics[i]['train_loss'],
            f'train_{metric_name}': train_metrics[i][f'train_{metric_name}'],
            'valid_loss': valid_metrics[i]['valid_loss'],
            f'valid_{metric_name}': valid_metrics[i][f'valid_{metric_name}'],
            'test_loss': test_loss,
            f'test_{metric_name}': test_metric
        })

    metrics_dir = os.path.join(RESULTS_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_df = pd.DataFrame(metrics)
    csv_filename = os.path.join(metrics_dir, f"SIREN_TCN_s{siren_dim}_l{n_siren_layers}_seed{seed}_metrics.csv")
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    w0_df = pd.DataFrame({
        'layer': list(range(len(w0_vals))),
        'w0': w0_vals
    })
    w0_filename = os.path.join(metrics_dir, f"SIREN_TCN_s{siren_dim}_l{n_siren_layers}_seed{seed}_w0.csv")
    w0_df.to_csv(w0_filename, index=False)
    print(f"w0 values saved to {w0_filename}")

    return test_loss, test_metric, model


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = get_args()

    print(f"\n{'='*60}")
    print("SIREN-TCN: Classical Fourier Baseline")
    print("Mirrors FourierQTCN architecture (quantum circuit -> SIREN)")
    print(f"{'='*60}\n")

    # Load data via dispatcher
    train_loader, val_loader, test_loader, input_dim, task, scaler = load_dataset(args, device)
    print(f"Dataset: {args.dataset}, Task: {task}, Input dim: {input_dim}")

    # Adjust kernel_size and dilation based on sampling frequency
    if args.kernel_size is None:
        if getattr(args, 'freq', 80) == 80:
            kernel_size, dilation = 12, 3
        elif getattr(args, 'freq', 80) == 4:
            kernel_size, dilation = 7, 2
        else:
            kernel_size, dilation = 10, 2
    else:
        kernel_size = args.kernel_size
        dilation = args.dilation

    test_loss, test_metric, model = run_training(
        seed=args.seed,
        siren_dim=args.siren_dim,
        n_siren_layers=args.n_siren_layers,
        input_dim=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        kernel_size=kernel_size,
        dilation=dilation,
        n_frequencies=args.n_frequencies,
        w0=args.w0,
        learnable_w0=args.learnable_w0,
        siren_hidden_dim=args.siren_hidden_dim,
        num_epochs=args.num_epochs,
        lr=args.lr,
        resume=args.resume,
        args=args,
        task=task
    )

    metric_name = 'AUC' if task == 'classification' else 'RMSE'
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Final Test {metric_name}: {test_metric:.4f}")
    print(f"{'='*60}")
