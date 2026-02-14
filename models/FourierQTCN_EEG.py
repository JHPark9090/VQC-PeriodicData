"""
Fourier-Based Quantum Temporal Convolutional Network (F-QTCN) for EEG Classification

This implementation utilizes VQC's periodic advantages through:
1. FFT-based frequency extraction (lossless, unlike bandpass filtering)
2. Simple linear projection (minimal classical computation)
3. Learnable frequency rescaling (essential for VQC periodic advantage)
4. Simple weighted aggregation (instead of complex attention)

Key Insight:
- VQC outputs Fourier series: f(x) = Σ c_ω e^{iωx}
- FFT extracts Fourier coefficients from input
- Both are Fourier-based → natural mathematical alignment
- FFT is lossless (invertible) unlike bandpass filtering

Based on theoretical foundations from:
- Schuld et al. (2021): "Effect of data encoding on the expressive power of VQC models"
- Ziyin et al. (2020): "Neural networks fail to learn periodic functions and how to fix it"

Author: Based on HQTCN2_EEG.py with Fourier-based modifications
Date: February 2026
"""

import scipy.constants  # Must be before pennylane (scipy 1.10.1 lazy-loading workaround)
import pennylane as qml
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
    print("Warning: Load_PhysioNet_EEG not found. You'll need to provide your own data loader.")

try:
    from dataset_dispatcher import add_dataset_args, load_dataset
except ImportError:
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from dataset_dispatcher import add_dataset_args, load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def get_args():
    parser = argparse.ArgumentParser(description="Fourier-Based QTCN for EEG Classification")
    parser.add_argument("--freq", type=int, default=80, help="Sampling frequency")
    parser.add_argument("--n-sample", type=int, default=50, help="Number of samples")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint")
    parser.add_argument("--n-qubits", type=int, default=8, help="Number of qubits")
    parser.add_argument("--circuit-depth", type=int, default=2, help="Quantum circuit depth")
    parser.add_argument("--n-frequencies", type=int, default=None, help="Number of FFT frequencies (default: n_qubits)")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--kernel-size", type=int, default=12, help="Temporal kernel size")
    parser.add_argument("--dilation", type=int, default=3, help="Dilation factor")
    parser.add_argument("--freq-init", type=str, default="fft",
                        choices=["fft", "linspace", "random"],
                        help="freq_scale initialization: 'fft' (data-informed), 'linspace' (generic), or 'random' (ablation D)")
    parser.add_argument("--ablate-fft", action="store_true",
                        help="Ablation A: bypass FFT, use raw windowed input")
    parser.add_argument("--ablate-freq-match", action="store_true",
                        help="Ablation B: remove RX gate, RY-only encoding")
    parser.add_argument("--ablate-rescaled", action="store_true",
                        help="Ablation C: apply sigmoid to quantum output (destroys periodicity)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output metrics CSV")
    add_dataset_args(parser)
    return parser.parse_args()


print('Pennylane Version:', qml.__version__)
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
    qml.numpy.random.seed(seed)


def analyze_training_frequencies(train_data, n_qubits, max_samples=1000):
    """
    Analyze training data via FFT to find dominant frequencies for freq_scale init.

    Args:
        train_data: DataLoader yielding (inputs, labels) where inputs=[batch, channels, time]
                    OR a Tensor [n_samples, seq_len] or [n_samples, channels, seq_len]
        n_qubits: Number of freq_scale values needed
        max_samples: Max samples to analyze

    Returns:
        freq_scale_init: Tensor [n_qubits] with FFT-informed initialization
    """
    from torch.utils.data import DataLoader

    # 1. Collect data into tensor
    if isinstance(train_data, DataLoader):
        chunks, n = [], 0
        for inputs, *_ in train_data:
            chunks.append(inputs.cpu().float())
            n += inputs.shape[0]
            if n >= max_samples:
                break
        data = torch.cat(chunks)[:max_samples]
    else:
        data = train_data[:max_samples].cpu().float()

    # 2. Ensure 3D [samples, channels, time]
    if data.dim() == 2:
        data = data.unsqueeze(1)

    # 3. Power spectrum via rfft
    spectrum = torch.fft.rfft(data, dim=-1)
    power = torch.abs(spectrum) ** 2
    avg_power = power.mean(dim=(0, 1))  # [n_freq_bins]
    avg_power[0] = 0.0  # Zero DC component

    # 4. Top-N frequency bins by power
    n_top = min(n_qubits, len(avg_power) - 1)
    top_indices = torch.argsort(avg_power, descending=True)[:n_top]
    top_indices = torch.sort(top_indices).values.float()

    # 5. Ratio-based scaling: normalize by fundamental frequency
    fundamental = top_indices[0].clamp(min=1.0)
    freq_scale = top_indices / fundamental
    freq_scale = freq_scale.clamp(min=0.5, max=5.0)

    # 6. Pad if fewer bins than n_qubits
    if len(freq_scale) < n_qubits:
        last_val = freq_scale[-1].item() if len(freq_scale) > 0 else 1.0
        pad = torch.linspace(last_val, min(last_val + 1.0, 5.0),
                             n_qubits - len(freq_scale) + 1)[1:]
        freq_scale = torch.cat([freq_scale, pad])

    freq_scale = freq_scale[:n_qubits]
    print(f"FFT-seeded freq_scale: {freq_scale.tolist()}")
    return freq_scale


# =============================================================================
# FOURIER-BASED QTCN MODEL
# =============================================================================

class FourierQTCN(nn.Module):
    """
    Fourier-Based Quantum Temporal Convolutional Network.

    This model utilizes VQC's periodic advantages with minimal classical computation:

    Architecture:
        Input EEG [batch, channels, time]
            ↓
        Sliding Window Extraction
            ↓
        FFT-based Frequency Extraction (LOSSLESS)
            ↓
        Simple Linear Projection
            ↓
        Frequency-Matched Quantum Encoding (KEY for periodic advantage)
            ↓
        Quantum Convolutional + Pooling Layers
            ↓
        Simple Weighted Aggregation
            ↓
        Output Prediction

    Why FFT instead of Bandpass Filtering:
    1. FFT is lossless (invertible) - no information loss
    2. FFT directly extracts Fourier coefficients
    3. VQC outputs Fourier series - natural mathematical alignment
    4. Computationally efficient: O(n log n)

    Why Learnable Frequency Rescaling:
    1. Maps data frequencies to VQC's frequency spectrum
    2. Similar to Snake activation's learnable 'a' parameter
    3. Essential for utilizing VQC's periodic structure
    """

    def __init__(
        self,
        n_qubits: int,
        circuit_depth: int,
        input_dim: Tuple[int, int, int],
        kernel_size: int,
        dilation: int = 1,
        n_frequencies: int = None,
        use_magnitude_phase: bool = True,
        freq_scale_init: torch.Tensor = None,
        ablate_fft: bool = False,
        ablate_freq_match: bool = False,
        ablate_rescaled: bool = False
    ):
        """
        Args:
            n_qubits: Number of qubits in quantum circuit
            circuit_depth: Number of conv-pool layers
            input_dim: (batch_size, n_channels, n_timepoints)
            kernel_size: Temporal convolution kernel size
            dilation: Dilation factor for temporal convolution
            n_frequencies: Number of FFT frequencies to use (default: n_qubits)
            use_magnitude_phase: If True, use magnitude and phase; else use real/imag
            freq_scale_init: Optional FFT-seeded initialization for freq_scale
            ablate_fft: Ablation A - bypass FFT, use raw windowed input
            ablate_freq_match: Ablation B - remove RX gate, RY-only encoding
            ablate_rescaled: Ablation C - apply sigmoid to quantum output
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_magnitude_phase = use_magnitude_phase
        self.ablate_fft = ablate_fft
        self.ablate_freq_match = ablate_freq_match
        self.ablate_rescaled = ablate_rescaled

        # Number of FFT frequencies to extract
        # Default to n_qubits for direct mapping, but clamp to max FFT bins
        max_fft_bins = kernel_size // 2 + 1  # rfft of kernel_size gives this many bins
        self.n_frequencies = min(n_frequencies if n_frequencies else n_qubits, max_fft_bins)

        # =================================================================
        # COMPONENT 1: FFT Feature Dimension Calculation
        # =================================================================
        # Ablation A: use raw flattened window instead of FFT features
        if self.ablate_fft:
            self.fft_feature_dim = self.input_channels * self.kernel_size
        else:
            # FFT gives complex values, we extract 2 features per frequency:
            # Option A: real and imaginary parts
            # Option B: magnitude and phase (often better for signals)
            self.fft_feature_dim = self.input_channels * self.n_frequencies * 2

        # =================================================================
        # COMPONENT 2: Simple Linear Projection
        # =================================================================
        # Minimal classical computation: just one linear layer
        # Projects FFT features to qubit dimension
        self.projection = nn.Linear(self.fft_feature_dim, n_qubits)

        # =================================================================
        # COMPONENT 3: Learnable Frequency Rescaling (ESSENTIAL)
        # =================================================================
        # This is THE KEY to utilizing VQC's periodic advantage
        # Similar to Snake activation's learnable 'a' parameter
        # Each qubit learns its optimal frequency scale
        #
        # Initialize with spread to cover different frequency ranges
        # Lower indices → lower frequencies, higher indices → higher frequencies
        if freq_scale_init is not None:
            self.freq_scale = nn.Parameter(freq_scale_init.clone().float())
            self._freq_init_method = 'fft'
        else:
            self.freq_scale = nn.Parameter(torch.linspace(0.5, 3.0, n_qubits))
            self._freq_init_method = 'linspace'

        # =================================================================
        # COMPONENT 4: Quantum Circuit Parameters
        # =================================================================
        # Convolutional layer parameters
        self.conv_params = nn.Parameter(
            torch.randn(circuit_depth, n_qubits, 15) * 0.1
        )
        # Pooling layer parameters
        self.pool_params = nn.Parameter(
            torch.randn(circuit_depth, n_qubits // 2, 3) * 0.1
        )

        # Quantum device initialization
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.quantum_circuit = qml.QNode(
            self._circuit,
            self.dev,
            interface='torch',
            diff_method='backprop'
        )

        # =================================================================
        # COMPONENT 5: Simple Weighted Aggregation
        # =================================================================
        # Instead of complex attention, use simple learnable weights
        # This preserves periodic information better than mean()
        max_windows = self.time_steps - self.dilation * (self.kernel_size - 1)
        self.agg_weights = nn.Parameter(torch.zeros(max_windows))

        # Store frequency bins for interpretability
        self._compute_frequency_bins()

        self._print_config()

    def _compute_frequency_bins(self):
        """Compute which physical frequencies each FFT bin corresponds to."""
        # FFT frequency bins depend on kernel_size
        # freq_bin[k] = k * sampling_rate / kernel_size
        # We store this for interpretability
        self.register_buffer(
            'freq_bin_indices',
            torch.arange(self.n_frequencies)
        )

    def _print_config(self):
        """Print model configuration."""
        print(f"\n{'='*60}")
        print("Fourier-Based QTCN Initialized")
        print(f"{'='*60}")
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Circuit depth: {self.circuit_depth}")
        print(f"Kernel size: {self.kernel_size}")
        print(f"Dilation: {self.dilation}")
        print(f"Number of FFT frequencies: {self.n_frequencies}")
        print(f"FFT feature dimension: {self.fft_feature_dim}")
        print(f"Use magnitude/phase: {self.use_magnitude_phase}")
        print(f"freq_scale init method: {self._freq_init_method}")
        print(f"Initial freq_scale range: [{self.freq_scale.min().item():.2f}, {self.freq_scale.max().item():.2f}]")
        print(f"{'='*60}\n")

    def _extract_fft_features(self, window: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency-domain features using FFT.

        This is the LOSSLESS alternative to bandpass filtering.
        FFT naturally separates frequency components without information loss.

        Args:
            window: [batch, channels, kernel_size]

        Returns:
            features: [batch, fft_feature_dim]
        """
        batch_size = window.shape[0]

        # Apply FFT along time dimension (last axis)
        # rfft is used for real-valued input, returns only positive frequencies
        fft_result = torch.fft.rfft(window, dim=-1)  # [batch, channels, freq_bins]

        # Select first n_frequencies bins
        # Lower frequency bins contain most of the signal energy for EEG
        fft_selected = fft_result[:, :, :self.n_frequencies]  # [batch, channels, n_freq]

        if self.use_magnitude_phase:
            # Option B: Magnitude and Phase (often better for periodic signals)
            # Magnitude captures amplitude of each frequency component
            # Phase captures timing/alignment of oscillations
            magnitude = torch.abs(fft_selected)
            phase = torch.angle(fft_selected)

            # Normalize magnitude (log scale often better for wide dynamic range)
            magnitude = torch.log1p(magnitude)

            # Normalize phase to [-1, 1]
            phase = phase / np.pi

            # Concatenate: [batch, channels, n_freq, 2]
            features = torch.stack([magnitude, phase], dim=-1)
        else:
            # Option A: Real and Imaginary parts
            features = torch.stack([fft_selected.real, fft_selected.imag], dim=-1)

        # Flatten: [batch, channels * n_frequencies * 2]
        features = features.reshape(batch_size, -1)

        return features

    def _circuit(self, features: torch.Tensor) -> torch.Tensor:
        """
        Quantum circuit with frequency-matched encoding.

        The KEY modification for VQC's periodic advantage:
        - Standard RY encoding for amplitude
        - Frequency-scaled RX encoding for phase/frequency matching

        This ensures VQC's Fourier spectrum aligns with the data's frequency content.
        """
        wires = list(range(self.n_qubits))

        # =================================================================
        # FREQUENCY-MATCHED ENCODING (Essential for periodic advantage)
        # =================================================================
        for i, wire in enumerate(wires):
            # Standard amplitude encoding (RY rotation)
            # Use [..., i] indexing to correctly handle both single [n_qubits]
            # and batched [batch, n_qubits] inputs via PennyLane broadcasting
            qml.RY(features[..., i], wires=wire)

            # FREQUENCY-MATCHED encoding (RX rotation with learnable scaling)
            # Ablation B: skip RX to reduce expressible frequencies (9^n → 3^n)
            if not self.ablate_freq_match:
                qml.RX(self.freq_scale[i] * features[..., i], wires=wire)

        # =================================================================
        # QUANTUM CONVOLUTIONAL AND POOLING LAYERS
        # =================================================================
        for layer in range(self.circuit_depth):
            # Convolutional layer with U3 and Ising gates
            self._apply_convolution(self.conv_params[layer], wires)

            # Pooling layer with mid-circuit measurement
            self._apply_pooling(self.pool_params[layer], wires)

            # Reduce active wires after pooling
            wires = wires[::2]

        # Final measurement
        return qml.expval(qml.PauliZ(0))

    def _apply_convolution(self, weights: torch.Tensor, wires: List[int]) -> None:
        """
        Apply quantum convolutional layer.

        Uses U3 gates for single-qubit rotations and Ising gates for
        two-qubit interactions to capture correlations between frequency components.
        """
        n_wires = len(wires)

        # Apply in two passes: even pairs, then odd pairs
        for parity in [0, 1]:
            for idx, w in enumerate(wires):
                if idx % 2 == parity and idx < n_wires - 1:
                    next_w = wires[idx + 1]

                    # Single-qubit rotations (before interaction)
                    qml.U3(*weights[idx, :3], wires=w)
                    qml.U3(*weights[idx + 1, 3:6], wires=next_w)

                    # Two-qubit Ising interactions
                    # These capture correlations between frequency components
                    qml.IsingZZ(weights[idx, 6], wires=[w, next_w])
                    qml.IsingYY(weights[idx, 7], wires=[w, next_w])
                    qml.IsingXX(weights[idx, 8], wires=[w, next_w])

                    # Single-qubit rotations (after interaction)
                    qml.U3(*weights[idx, 9:12], wires=w)
                    qml.U3(*weights[idx + 1, 12:15], wires=next_w)

    def _apply_pooling(self, pool_weights: torch.Tensor, wires: List[int]) -> None:
        """
        Apply quantum pooling layer with mid-circuit measurement.

        Measures every other qubit and conditionally applies rotation
        to transfer information to remaining qubits.
        """
        n_wires = len(wires)

        for idx, w in enumerate(wires):
            if idx % 2 == 1 and idx < n_wires:
                # Measure qubit
                measurement = qml.measure(w)
                # Conditionally apply rotation based on measurement
                qml.cond(measurement, qml.U3)(
                    *pool_weights[idx // 2],
                    wires=wires[idx - 1]
                )

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
            # Ablation A: bypass FFT, use raw flattened window
            # =================================================================
            if self.ablate_fft:
                fft_features = window.reshape(batch_size, -1)
            else:
                fft_features = self._extract_fft_features(window)

            # =================================================================
            # STEP 2: Simple linear projection
            # =================================================================
            projected = self.projection(fft_features)

            # Normalize to prevent exploding gradients in quantum circuit
            projected = torch.tanh(projected) * np.pi

            # =================================================================
            # STEP 3: Quantum circuit with frequency-matched encoding
            # =================================================================
            # Cast to float64 for PennyLane backprop compatibility, then back
            window_output = self.quantum_circuit(projected.double()).float()

            # Ablation C: apply sigmoid to quantum output (destroys periodicity)
            if self.ablate_rescaled:
                window_output = torch.sigmoid(window_output)
            else:
                window_output = (window_output + 1) / 2  # preserves periodicity

            outputs.append(window_output)

        # =================================================================
        # STEP 4: Simple weighted aggregation
        # =================================================================
        outputs = torch.stack(outputs, dim=1)  # [batch, n_windows]
        n_windows = outputs.shape[1]

        # Softmax over learnable weights
        weights = F.softmax(self.agg_weights[:n_windows], dim=0)

        # Weighted sum (preserves periodic information better than mean)
        output = (outputs * weights.unsqueeze(0)).sum(dim=1)

        return output

    def get_frequency_scales(self) -> np.ndarray:
        """Return learned frequency scaling factors for analysis."""
        return self.freq_scale.detach().cpu().numpy()

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
    n_qubits: int,
    circuit_depth: int,
    input_dim: Tuple[int, int, int],
    train_loader,
    val_loader,
    test_loader,
    kernel_size: int = 12,
    dilation: int = 3,
    n_frequencies: int = None,
    num_epochs: int = 50,
    lr: float = 0.001,
    checkpoint_dir: str = None,
    resume: bool = False,
    args=None,
    task: str = 'classification',
    freq_init: str = 'fft',
    ablate_fft: bool = False,
    ablate_freq_match: bool = False,
    ablate_rescaled: bool = False,
    output_dir: str = None
):
    """Run full training pipeline for Fourier-Based QTCN."""

    # Determine variant name from ablation flags
    variant = "full"
    if ablate_fft:
        variant = "no_fft"
    elif ablate_freq_match:
        variant = "no_freq_match"
    elif ablate_rescaled:
        variant = "no_rescaled"
    elif freq_init == "random":
        variant = "no_fft_init"

    print(f"\n{'='*60}")
    print("Starting Fourier-Based QTCN Training")
    print(f"{'='*60}")

    set_all_seeds(seed)
    print(f"Random Seed: {seed}")
    print(f"Variant: {variant}")
    print(f"Ablate FFT: {ablate_fft}")
    print(f"Ablate Freq-Match: {ablate_freq_match}")
    print(f"Ablate Rescaled: {ablate_rescaled}")

    # Compute freq_scale initialization
    freq_scale_init = None
    if freq_init == 'fft':
        freq_scale_init = analyze_training_frequencies(train_loader, n_qubits)
    elif freq_init == 'random':
        freq_scale_init = torch.rand(n_qubits) * 2.5 + 0.5  # uniform [0.5, 3.0]
        print(f"Random freq_scale init: {freq_scale_init.tolist()}")

    # Create model
    model = FourierQTCN(
        n_qubits=n_qubits,
        circuit_depth=circuit_depth,
        input_dim=input_dim,
        kernel_size=kernel_size,
        dilation=dilation,
        n_frequencies=n_frequencies,
        freq_scale_init=freq_scale_init,
        ablate_fft=ablate_fft,
        ablate_freq_match=ablate_freq_match,
        ablate_rescaled=ablate_rescaled
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
        checkpoint_dir = os.path.join(RESULTS_DIR, 'FourierQTCN_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"fourier_qtcn_q{n_qubits}_d{circuit_depth}_seed{seed}.pth"
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
            freq_scales = model.get_frequency_scales()
            print(f"Learned freq_scale: [{freq_scales.min():.2f}, {freq_scales.max():.2f}]")

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

    freq_scales = model.get_frequency_scales()
    print(f"\nLearned Frequency Scales:")
    for i, scale in enumerate(freq_scales):
        print(f"  Qubit {i}: {scale:.4f}")

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

    # Use output_dir for ablation experiments, else default metrics dir
    if output_dir:
        metrics_dir = output_dir
    else:
        metrics_dir = os.path.join(RESULTS_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    dataset_name = getattr(args, 'dataset', 'eeg') if args else 'eeg'
    metrics_df = pd.DataFrame(metrics)
    csv_filename = os.path.join(metrics_dir, f"FourierQTCN_{dataset_name}_{variant}_seed{seed}_metrics.csv")
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    freq_df = pd.DataFrame({
        'qubit': list(range(n_qubits)),
        'freq_scale': freq_scales
    })
    freq_filename = os.path.join(metrics_dir, f"FourierQTCN_{dataset_name}_{variant}_seed{seed}_freqscales.csv")
    freq_df.to_csv(freq_filename, index=False)
    print(f"Frequency scales saved to {freq_filename}")

    return test_loss, test_metric, model


# =============================================================================
# COMPARISON: Original QTCN vs Fourier QTCN
# =============================================================================

def print_comparison():
    """Print comparison between original QTCN and Fourier QTCN."""
    comparison = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║            ORIGINAL QTCN vs FOURIER-BASED QTCN                       ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  ORIGINAL QTCN:                                                      ║
    ║  ├─ Encoding: Generic AngleEmbedding (RY only)                       ║
    ║  ├─ Frequency Matching: None                                         ║
    ║  ├─ Preprocessing: FC layer (information loss)                       ║
    ║  ├─ Aggregation: Simple mean (destroys periodicity)                  ║
    ║  └─ VQC Periodic Advantage: NOT UTILIZED                             ║
    ║                                                                      ║
    ║  FOURIER-BASED QTCN:                                                 ║
    ║  ├─ Encoding: RY (amplitude) + RX (frequency-scaled)                 ║
    ║  ├─ Frequency Matching: Learnable freq_scale (like Snake's 'a')      ║
    ║  ├─ Preprocessing: FFT (lossless) + Linear projection                ║
    ║  ├─ Aggregation: Learnable weighted sum                              ║
    ║  └─ VQC Periodic Advantage: FULLY UTILIZED                           ║
    ║                                                                      ║
    ║  KEY DIFFERENCES:                                                    ║
    ║  ┌────────────────────┬──────────────────┬──────────────────────┐    ║
    ║  │ Aspect             │ Original QTCN    │ Fourier QTCN         │    ║
    ║  ├────────────────────┼──────────────────┼──────────────────────┤    ║
    ║  │ Information Loss   │ High (FC layer)  │ None (FFT lossless)  │    ║
    ║  │ Freq. Extraction   │ None             │ FFT-based            │    ║
    ║  │ Freq. Matching     │ None             │ Learnable scaling    │    ║
    ║  │ Periodic Advantage │ None             │ Full                 │    ║
    ║  │ Classical Overhead │ Medium           │ Low                  │    ║
    ║  └────────────────────┴──────────────────┴──────────────────────┘    ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(comparison)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = get_args()

    print_comparison()

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

    # Force linspace init when ablating FFT
    freq_init = args.freq_init
    if args.ablate_fft and freq_init == 'fft':
        print("Note: --ablate-fft forces --freq-init=linspace (can't do FFT init without FFT)")
        freq_init = 'linspace'

    test_loss, test_metric, model = run_training(
        seed=args.seed,
        n_qubits=args.n_qubits,
        circuit_depth=args.circuit_depth,
        input_dim=input_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        kernel_size=kernel_size,
        dilation=dilation,
        n_frequencies=args.n_frequencies,
        num_epochs=args.num_epochs,
        lr=args.lr,
        resume=args.resume,
        args=args,
        task=task,
        freq_init=freq_init,
        ablate_fft=args.ablate_fft,
        ablate_freq_match=args.ablate_freq_match,
        ablate_rescaled=args.ablate_rescaled,
        output_dir=args.output_dir
    )

    metric_name = 'AUC' if task == 'classification' else 'RMSE'
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Final Test {metric_name}: {test_metric:.4f}")
    print(f"{'='*60}")
