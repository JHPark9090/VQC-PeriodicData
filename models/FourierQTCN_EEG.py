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
        use_magnitude_phase: bool = True
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
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_magnitude_phase = use_magnitude_phase

        # Number of FFT frequencies to extract
        # Default to n_qubits for direct mapping
        self.n_frequencies = n_frequencies if n_frequencies else n_qubits

        # =================================================================
        # COMPONENT 1: FFT Feature Dimension Calculation
        # =================================================================
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
        self.freq_scale = nn.Parameter(
            torch.linspace(0.5, 3.0, n_qubits)
        )

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
            qml.RY(features[i], wires=wire)

            # FREQUENCY-MATCHED encoding (RX rotation with learnable scaling)
            # This is analogous to Snake's learnable 'a' parameter
            # It maps the input to VQC's natural frequency spectrum
            qml.RX(self.freq_scale[i] * features[i], wires=wire)

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
            # =================================================================
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
            window_output = self.quantum_circuit(projected)
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
    clip_grad: float = 1.0
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

        # Gradient clipping for stability
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()
        train_loss += loss.item()

        all_labels.append(labels.cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    # Handle edge cases for AUC calculation
    try:
        train_auroc = roc_auc_score(all_labels, all_outputs)
    except ValueError:
        train_auroc = 0.5  # Default if only one class present

    return train_loss / len(dataloader), train_auroc


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device
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

    try:
        auroc = roc_auc_score(all_labels, all_outputs)
    except ValueError:
        auroc = 0.5

    return running_loss / len(dataloader), auroc


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
    checkpoint_dir: str = "FourierQTCN_checkpoints",
    resume: bool = False,
    args=None
):
    """
    Run full training pipeline for Fourier-Based QTCN.

    Args:
        seed: Random seed
        n_qubits: Number of qubits
        circuit_depth: Quantum circuit depth
        input_dim: Input dimensions
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        kernel_size: Temporal kernel size
        dilation: Dilation factor
        n_frequencies: Number of FFT frequencies
        num_epochs: Training epochs
        lr: Learning rate
        checkpoint_dir: Directory for checkpoints
        resume: Whether to resume from checkpoint
        args: Additional arguments

    Returns:
        test_loss, test_auc, model
    """
    print(f"\n{'='*60}")
    print("Starting Fourier-Based QTCN Training")
    print(f"{'='*60}")

    set_all_seeds(seed)
    print(f"Random Seed: {seed}")

    # Create model
    model = FourierQTCN(
        n_qubits=n_qubits,
        circuit_depth=circuit_depth,
        input_dim=input_dim,
        kernel_size=kernel_size,
        dilation=dilation,
        n_frequencies=n_frequencies
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Checkpoint handling
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"fourier_qtcn_q{n_qubits}_d{circuit_depth}_seed{seed}.pth"
    )

    start_epoch = 0
    train_metrics, valid_metrics = [], []
    best_val_auc = 0.0
    best_model_state = None

    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_metrics = checkpoint.get('train_metrics', [])
        valid_metrics = checkpoint.get('valid_metrics', [])
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        print(f"Resuming from epoch {start_epoch + 1}, best AUC: {best_val_auc:.4f}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc
        })

        # Validate
        valid_loss, valid_auc = evaluate(model, val_loader, criterion, device)
        valid_metrics.append({
            'epoch': epoch + 1,
            'valid_loss': valid_loss,
            'valid_auc': valid_auc
        })

        # Update learning rate
        scheduler.step(valid_auc)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Track best model
        if valid_auc > best_val_auc:
            best_val_auc = valid_auc
            best_model_state = copy.deepcopy(model.state_dict())

        # Print progress
        print(f"\nEpoch: {epoch + 1:02}/{num_epochs} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, AUC: {valid_auc:.4f} (Best: {best_val_auc:.4f})")

        # Print learned frequency scales periodically
        if (epoch + 1) % 10 == 0:
            freq_scales = model.get_frequency_scales()
            print(f"Learned freq_scale: [{freq_scales.min():.2f}, {freq_scales.max():.2f}]")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'best_val_auc': best_val_auc
        }, checkpoint_path)

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final test evaluation
    test_loss, test_auc = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Best Validation AUC: {best_val_auc:.4f}")

    # Print final learned parameters
    freq_scales = model.get_frequency_scales()
    print(f"\nLearned Frequency Scales:")
    for i, scale in enumerate(freq_scales):
        print(f"  Qubit {i}: {scale:.4f}")

    print(f"{'='*60}\n")

    # Save metrics to CSV
    test_metrics = [{
        'epoch': num_epochs,
        'test_loss': test_loss,
        'test_auc': test_auc
    }]

    metrics = []
    for i in range(len(train_metrics)):
        metrics.append({
            'epoch': i + 1,
            'train_loss': train_metrics[i]['train_loss'],
            'train_auc': train_metrics[i]['train_auc'],
            'valid_loss': valid_metrics[i]['valid_loss'],
            'valid_auc': valid_metrics[i]['valid_auc'],
            'test_loss': test_loss,
            'test_auc': test_auc
        })

    metrics_df = pd.DataFrame(metrics)
    csv_filename = f"FourierQTCN_q{n_qubits}_d{circuit_depth}_seed{seed}_metrics.csv"
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    # Save learned frequency scales
    freq_df = pd.DataFrame({
        'qubit': list(range(n_qubits)),
        'freq_scale': freq_scales
    })
    freq_filename = f"FourierQTCN_q{n_qubits}_d{circuit_depth}_seed{seed}_freqscales.csv"
    freq_df.to_csv(freq_filename, index=False)
    print(f"Frequency scales saved to {freq_filename}")

    return test_loss, test_auc, model


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

    # Print comparison
    print_comparison()

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=args.seed,
        device=device,
        batch_size=32,
        sampling_freq=args.freq,
        sample_size=args.n_sample
    )

    print(f"Input Dimension: {input_dim}")
    print(f"Sampling Frequency: {args.freq} Hz")

    # Adjust kernel_size and dilation based on sampling frequency
    if args.kernel_size is None:
        if args.freq == 80:
            kernel_size, dilation = 12, 3
        elif args.freq == 4:
            kernel_size, dilation = 7, 2
        else:
            kernel_size, dilation = 10, 2
    else:
        kernel_size = args.kernel_size
        dilation = args.dilation

    # Run training
    test_loss, test_auc, model = run_training(
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
        args=args
    )

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Final Test AUC: {test_auc:.4f}")
    print(f"{'='*60}")
