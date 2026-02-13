"""
Periodic-Aware Quantum Temporal Convolutional Network (PA-QTCN) for EEG Classification

This implementation fully utilizes VQC's periodic advantages through:
1. Frequency band decomposition before quantum encoding
2. Frequency-matched encoding with proper rescaling
3. Attention-based temporal aggregation (replacing simple mean)
4. VQC spectrum designed to match EEG frequency bands

Based on theoretical foundations from:
- Schuld et al. (2021): "Effect of data encoding on the expressive power of VQC models"
- Ziyin et al. (2020): "Neural networks fail to learn periodic functions and how to fix it"

Author: Modified from HQTCN2_EEG.py
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
from typing import Tuple, List, Dict, Optional
from scipy.signal import butter, filtfilt, hilbert
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

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def get_args():
    parser = argparse.ArgumentParser(description="Periodic-Aware QTCN for EEG Classification")
    parser.add_argument("--freq", type=int, default=80, help="Sampling frequency")
    parser.add_argument("--n-sample", type=int, default=50, help="Number of samples")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from checkpoint")
    parser.add_argument("--n-qubits", type=int, default=8, help="Number of qubits")
    parser.add_argument("--circuit-depth", type=int, default=2, help="Quantum circuit depth")
    parser.add_argument("--target-bands", nargs='+', default=['alpha', 'beta', 'gamma'],
                        help="Target EEG frequency bands")
    parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs")
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
# MODIFICATION 1: Frequency Band Decomposition
# =============================================================================

class FrequencyBandDecomposer(nn.Module):
    """
    Decomposes EEG signals into frequency bands before quantum encoding.

    This ensures that VQC receives frequency-separated inputs, allowing
    the quantum circuit to process each band with matched encoding.

    EEG Frequency Bands:
    - Delta: 0.5-4 Hz (deep sleep)
    - Theta: 4-8 Hz (drowsiness, memory)
    - Alpha: 8-13 Hz (relaxed wakefulness)
    - Beta: 13-30 Hz (active thinking)
    - Gamma: 30-100 Hz (cognitive processing)
    """

    EEG_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    def __init__(self, sampling_rate: float, target_bands: List[str], filter_order: int = 4):
        """
        Args:
            sampling_rate: EEG sampling rate in Hz
            target_bands: List of band names to extract ['alpha', 'beta', 'gamma']
            filter_order: Butterworth filter order
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.target_bands = target_bands
        self.filter_order = filter_order
        self.nyquist = sampling_rate / 2

        # Precompute filter coefficients
        self.filters = {}
        for band_name in target_bands:
            if band_name not in self.EEG_BANDS:
                raise ValueError(f"Unknown band: {band_name}. Available: {list(self.EEG_BANDS.keys())}")

            low, high = self.EEG_BANDS[band_name]
            # Ensure frequencies are within valid range
            low = max(low, 0.5)
            high = min(high, self.nyquist - 1)

            # Design bandpass filter
            b, a = butter(filter_order, [low/self.nyquist, high/self.nyquist], btype='band')
            self.filters[band_name] = (b, a)

        # Store center frequencies for rescaling
        self.center_frequencies = {
            band: (self.EEG_BANDS[band][0] + self.EEG_BANDS[band][1]) / 2
            for band in target_bands
        }

    def bandpass_filter(self, x: np.ndarray, band_name: str) -> np.ndarray:
        """Apply bandpass filter for a specific frequency band."""
        b, a = self.filters[band_name]
        # filtfilt applies filter twice (forward and backward) for zero phase
        return filtfilt(b, a, x, axis=-1)

    def extract_envelope(self, x: np.ndarray) -> np.ndarray:
        """Extract amplitude envelope using Hilbert transform."""
        analytic_signal = hilbert(x, axis=-1)
        return np.abs(analytic_signal)

    def extract_phase(self, x: np.ndarray) -> np.ndarray:
        """Extract instantaneous phase using Hilbert transform."""
        analytic_signal = hilbert(x, axis=-1)
        return np.angle(analytic_signal)

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Decompose input into frequency bands.

        Args:
            x: Input tensor [batch, channels, time]

        Returns:
            Dictionary with band names as keys, each containing:
                - 'filtered': Bandpass filtered signal
                - 'envelope': Amplitude envelope
                - 'phase': Instantaneous phase
        """
        # Convert to numpy for filtering
        x_np = x.detach().cpu().numpy()

        band_signals = {}
        for band_name in self.target_bands:
            # Apply bandpass filter
            filtered = self.bandpass_filter(x_np, band_name)

            # Extract envelope and phase
            envelope = self.extract_envelope(filtered)
            phase = self.extract_phase(filtered)

            band_signals[band_name] = {
                'filtered': torch.tensor(filtered, dtype=x.dtype, device=x.device),
                'envelope': torch.tensor(envelope, dtype=x.dtype, device=x.device),
                'phase': torch.tensor(phase, dtype=x.dtype, device=x.device)
            }

        return band_signals


# =============================================================================
# MODIFICATION 2 & 4: Frequency-Matched Encoding with VQC Spectrum Design
# =============================================================================

class FrequencyMatchedEncoder(nn.Module):
    """
    Computes rescaling factors to match EEG frequency bands to VQC spectrum.

    VQC Theory (Schuld et al., 2021):
    - VQC output is a Fourier series: f(x) = Σ c_ω e^{iωx}
    - Frequencies ω are determined by encoding Hamiltonian eigenvalues
    - For single-qubit Pauli rotations repeated r times: ω ∈ {-r, ..., 0, ..., r}

    Strategy:
    - Map each EEG band's center frequency to an integer in VQC's spectrum
    - Apply rescaling: x_scaled = α * x, where α = 2π * target_ω / f_center
    """

    def __init__(self, target_bands: List[str], n_qubits: int, center_frequencies: Dict[str, float]):
        """
        Args:
            target_bands: List of EEG band names
            n_qubits: Number of qubits in VQC
            center_frequencies: Dict mapping band names to center frequencies
        """
        super().__init__()
        self.target_bands = target_bands
        self.n_qubits = n_qubits
        self.n_bands = len(target_bands)
        self.center_frequencies = center_frequencies

        # Compute rescaling factors
        rescaling = self._compute_rescaling_factors()
        self.register_buffer('freq_rescaling', rescaling)

        # Qubits per band
        self.qubits_per_band = n_qubits // self.n_bands

    def _compute_rescaling_factors(self) -> torch.Tensor:
        """
        Compute rescaling factors to match EEG bands to VQC frequencies.

        Design principle:
        - Alpha (10.5 Hz) → VQC frequency 1
        - Beta (21.5 Hz) → VQC frequency 2
        - Gamma (65 Hz) → VQC frequency 3

        This ensures VQC's Fourier components align with EEG's periodic structure.
        """
        rescaling = []

        # Sort bands by center frequency for consistent mapping
        sorted_bands = sorted(self.target_bands,
                             key=lambda b: self.center_frequencies[b])

        for idx, band_name in enumerate(sorted_bands):
            center_freq = self.center_frequencies[band_name]

            # Map to VQC integer frequency (1, 2, 3, ...)
            target_vqc_freq = idx + 1

            # Rescaling factor: ensures encoded value has correct period
            # When x_scaled = α * x is encoded via e^{ix_scaled},
            # the effective frequency becomes α * f_data
            # We want α * f_center = target_vqc_freq
            # Therefore: α = target_vqc_freq / f_center * 2π (for period matching)
            scale = 2 * np.pi * target_vqc_freq / center_freq

            rescaling.append(scale)

        # Reorder to match original band order
        reordered = []
        for band in self.target_bands:
            idx = sorted_bands.index(band)
            reordered.append(rescaling[idx])

        return torch.tensor(reordered, dtype=torch.float32)

    def get_rescaling_for_band(self, band_idx: int) -> torch.Tensor:
        """Get rescaling factor for a specific band index."""
        return self.freq_rescaling[band_idx]

    def get_qubit_band_mapping(self) -> List[int]:
        """
        Get mapping from qubit index to band index.

        Returns:
            List where element i is the band index for qubit i
        """
        mapping = []
        for band_idx in range(self.n_bands):
            mapping.extend([band_idx] * self.qubits_per_band)

        # Handle remaining qubits
        remaining = self.n_qubits - len(mapping)
        mapping.extend([self.n_bands - 1] * remaining)

        return mapping


# =============================================================================
# MODIFICATION 3: Attention-Based Temporal Aggregation
# =============================================================================

class TemporalAttentionAggregator(nn.Module):
    """
    Replaces simple mean aggregation with attention-based aggregation.

    Problem with mean:
    - Averaging over time windows destroys periodic information
    - VQC's periodic patterns are lost

    Solution:
    - Use self-attention to learn which time windows are important
    - Preserve periodic structure through weighted combination
    """

    def __init__(self, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.embed_dim = embed_dim

        # Project scalar VQC output to embedding dimension
        self.input_projection = nn.Linear(1, embed_dim)

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Output projection back to scalar
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        # Learnable aggregation weights (fallback)
        self.aggregate_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate temporal outputs using attention.

        Args:
            x: VQC outputs over time [batch, time_steps]

        Returns:
            Aggregated output [batch]
        """
        batch_size, time_steps = x.shape

        # Add feature dimension: [batch, time, 1]
        x = x.unsqueeze(-1)

        # Project to embedding dimension: [batch, time, embed_dim]
        x_embed = self.input_projection(x)

        # Self-attention: [batch, time, embed_dim]
        attended, attention_weights = self.self_attention(x_embed, x_embed, x_embed)

        # Residual connection and layer norm
        x_norm = self.layer_norm(attended + x_embed)

        # Project back to scalar: [batch, time, 1]
        x_out = self.output_projection(x_norm)

        # Weighted aggregation (attention-weighted mean)
        # Use attention weights from last layer for weighting
        weights = attention_weights.mean(dim=1)  # [batch, time]
        weights = F.softmax(weights * self.aggregate_weight, dim=-1)

        # Weighted sum: [batch]
        output = (weights.unsqueeze(-1) * x_out).sum(dim=1).squeeze(-1)

        return output


# =============================================================================
# MAIN MODEL: Periodic-Aware QTCN
# =============================================================================

class PeriodicAwareQTCN(nn.Module):
    """
    Periodic-Aware Quantum Temporal Convolutional Network for EEG.

    This model fully utilizes VQC's periodic advantages by:
    1. Decomposing EEG into frequency bands before encoding
    2. Using frequency-matched quantum encoding
    3. Designing VQC spectrum to match EEG bands
    4. Using attention-based temporal aggregation

    Architecture:
        Input EEG [batch, channels, time]
            ↓
        Frequency Band Decomposition (alpha, beta, gamma)
            ↓
        Band-Specific Spatial Compression (classical)
            ↓
        Frequency-Matched Quantum Encoding
            ↓
        Quantum Convolutional Layers
            ↓
        Quantum Pooling Layers
            ↓
        Attention-Based Temporal Aggregation
            ↓
        Output Prediction
    """

    def __init__(
        self,
        n_qubits: int,
        circuit_depth: int,
        input_dim: Tuple[int, int, int],
        kernel_size: int,
        dilation: int = 1,
        sampling_rate: float = 160.0,
        target_bands: List[str] = ['alpha', 'beta', 'gamma'],
        attention_embed_dim: int = 32,
        attention_heads: int = 4,
        use_envelope: bool = True,
        use_phase: bool = True
    ):
        """
        Args:
            n_qubits: Number of qubits in quantum circuit
            circuit_depth: Number of conv-pool layers in quantum circuit
            input_dim: Input dimensions (batch_size, n_channels, n_timepoints)
            kernel_size: Temporal convolution kernel size
            dilation: Dilation factor for temporal convolution
            sampling_rate: EEG sampling rate in Hz
            target_bands: EEG frequency bands to process
            attention_embed_dim: Embedding dimension for temporal attention
            attention_heads: Number of attention heads
            use_envelope: Whether to use amplitude envelope features
            use_phase: Whether to use phase features
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sampling_rate = sampling_rate
        self.target_bands = target_bands
        self.n_bands = len(target_bands)
        self.use_envelope = use_envelope
        self.use_phase = use_phase

        # Validate n_qubits is divisible by n_bands for clean band allocation
        if n_qubits % self.n_bands != 0:
            print(f"Warning: n_qubits ({n_qubits}) not divisible by n_bands ({self.n_bands}). "
                  f"Some bands may have more qubits than others.")

        self.qubits_per_band = n_qubits // self.n_bands

        # =================================================================
        # MODIFICATION 1: Frequency Band Decomposer
        # =================================================================
        self.band_decomposer = FrequencyBandDecomposer(
            sampling_rate=sampling_rate,
            target_bands=target_bands
        )

        # =================================================================
        # MODIFICATION 2 & 4: Frequency-Matched Encoder
        # =================================================================
        self.freq_encoder = FrequencyMatchedEncoder(
            target_bands=target_bands,
            n_qubits=n_qubits,
            center_frequencies=self.band_decomposer.center_frequencies
        )

        # =================================================================
        # Band-Specific Spatial Compression
        # =================================================================
        # Calculate input features per band based on what we use
        features_per_point = 1  # filtered signal
        if use_envelope:
            features_per_point += 1
        if use_phase:
            features_per_point += 1

        spatial_input_dim = self.input_channels * self.kernel_size * features_per_point

        self.spatial_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(spatial_input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, self.qubits_per_band)
            )
            for _ in range(self.n_bands)
        ])

        # =================================================================
        # Quantum Circuit Parameters
        # =================================================================
        self.conv_params = nn.Parameter(torch.randn(circuit_depth, n_qubits, 15) * 0.1)
        self.pool_params = nn.Parameter(torch.randn(circuit_depth, n_qubits // 2, 3) * 0.1)

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.quantum_circuit = qml.QNode(
            self._circuit,
            self.dev,
            interface='torch',
            diff_method='backprop'
        )

        # =================================================================
        # MODIFICATION 3: Attention-Based Temporal Aggregation
        # =================================================================
        self.temporal_aggregator = TemporalAttentionAggregator(
            embed_dim=attention_embed_dim,
            num_heads=attention_heads
        )

        # Store frequency rescaling as buffer for circuit access
        self.register_buffer('freq_rescaling', self.freq_encoder.freq_rescaling)

        # Qubit to band mapping
        self.qubit_band_mapping = self.freq_encoder.get_qubit_band_mapping()

        print(f"\n{'='*60}")
        print("Periodic-Aware QTCN Initialized")
        print(f"{'='*60}")
        print(f"Target bands: {target_bands}")
        print(f"Center frequencies: {self.band_decomposer.center_frequencies}")
        print(f"Frequency rescaling factors: {self.freq_rescaling.tolist()}")
        print(f"Qubits per band: {self.qubits_per_band}")
        print(f"Qubit-to-band mapping: {self.qubit_band_mapping}")
        print(f"{'='*60}\n")

    def _circuit(self, features: torch.Tensor) -> torch.Tensor:
        """
        Quantum circuit with frequency-matched encoding.

        The encoding ensures VQC's Fourier spectrum aligns with EEG frequency bands.
        """
        wires = list(range(self.n_qubits))

        # =================================================================
        # FREQUENCY-MATCHED ENCODING
        # =================================================================
        for i, wire in enumerate(wires):
            # Get band index for this qubit
            band_idx = self.qubit_band_mapping[i]
            rescaling = self.freq_rescaling[band_idx]

            # Encode amplitude (RY rotation)
            qml.RY(features[i], wires=wire)

            # Encode with frequency matching (RX rotation with rescaling)
            # This ensures the VQC's Fourier components align with EEG bands
            qml.RX(rescaling * features[i], wires=wire)

        # =================================================================
        # QUANTUM CONVOLUTIONAL AND POOLING LAYERS
        # =================================================================
        for layer in range(self.circuit_depth):
            # Convolutional layer
            self._apply_convolution(self.conv_params[layer], wires)

            # Pooling layer
            self._apply_pooling(self.pool_params[layer], wires)

            # Reduce active wires after pooling
            wires = wires[::2]

        # Measurement
        return qml.expval(qml.PauliZ(0))

    def _apply_convolution(self, weights: torch.Tensor, wires: List[int]) -> None:
        """Apply quantum convolutional layer with U3 and Ising gates."""
        n_wires = len(wires)

        for parity in [0, 1]:  # Even and odd pairs
            for idx, w in enumerate(wires):
                if idx % 2 == parity and idx < n_wires - 1:
                    next_w = wires[idx + 1]

                    # Two-qubit unitary decomposition
                    qml.U3(*weights[idx, :3], wires=w)
                    qml.U3(*weights[idx + 1, 3:6], wires=next_w)

                    # Ising interactions (capture correlations)
                    qml.IsingZZ(weights[idx, 6], wires=[w, next_w])
                    qml.IsingYY(weights[idx, 7], wires=[w, next_w])
                    qml.IsingXX(weights[idx, 8], wires=[w, next_w])

                    # Final single-qubit rotations
                    qml.U3(*weights[idx, 9:12], wires=w)
                    qml.U3(*weights[idx + 1, 12:15], wires=next_w)

    def _apply_pooling(self, pool_weights: torch.Tensor, wires: List[int]) -> None:
        """Apply quantum pooling layer with mid-circuit measurement."""
        n_wires = len(wires)

        for idx, w in enumerate(wires):
            if idx % 2 == 1 and idx < n_wires:
                # Measure qubit and conditionally apply rotation
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(
                    *pool_weights[idx // 2],
                    wires=wires[idx - 1]
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with full periodic advantage utilization.

        Args:
            x: Input EEG [batch, channels, time_steps]

        Returns:
            Output predictions [batch]
        """
        batch_size = x.shape[0]

        # =================================================================
        # STEP 1: Frequency Band Decomposition
        # =================================================================
        band_signals = self.band_decomposer(x)

        # =================================================================
        # STEP 2: Process Each Time Window
        # =================================================================
        all_window_outputs = []

        effective_start = self.dilation * (self.kernel_size - 1)

        for t in range(effective_start, self.time_steps):
            # Get window indices with dilation
            indices = [t - d * self.dilation for d in range(self.kernel_size)]
            indices.reverse()

            # =================================================================
            # STEP 3: Band-Specific Feature Extraction
            # =================================================================
            band_features = []

            for band_idx, band_name in enumerate(self.target_bands):
                # Get band signals
                band_data = band_signals[band_name]

                # Extract window for each feature type
                features_list = []

                # Filtered signal (always used)
                filtered_window = band_data['filtered'][:, :, indices]
                features_list.append(filtered_window)

                # Envelope (optional)
                if self.use_envelope:
                    envelope_window = band_data['envelope'][:, :, indices]
                    features_list.append(envelope_window)

                # Phase (optional)
                if self.use_phase:
                    phase_window = band_data['phase'][:, :, indices]
                    features_list.append(phase_window)

                # Concatenate features: [batch, channels, kernel_size, n_features]
                window_features = torch.stack(features_list, dim=-1)

                # Flatten: [batch, channels * kernel_size * n_features]
                window_flat = window_features.reshape(batch_size, -1)

                # Apply band-specific spatial compression
                compressed = self.spatial_encoders[band_idx](window_flat)

                band_features.append(compressed)

            # =================================================================
            # STEP 4: Combine Band Features for Quantum Circuit
            # =================================================================
            # Concatenate: [batch, n_qubits]
            combined = torch.cat(band_features, dim=-1)

            # Ensure correct size
            if combined.shape[-1] < self.n_qubits:
                padding = torch.zeros(
                    batch_size,
                    self.n_qubits - combined.shape[-1],
                    device=x.device,
                    dtype=x.dtype
                )
                combined = torch.cat([combined, padding], dim=-1)
            else:
                combined = combined[:, :self.n_qubits]

            # Normalize to prevent exploding gradients
            combined = torch.tanh(combined) * np.pi

            # =================================================================
            # STEP 5: Quantum Circuit Execution
            # =================================================================
            window_output = self.quantum_circuit(combined)
            all_window_outputs.append(window_output)

        # =================================================================
        # STEP 6: Attention-Based Temporal Aggregation
        # =================================================================
        # Stack outputs: [batch, time_windows]
        outputs_stacked = torch.stack(all_window_outputs, dim=1)

        # Apply attention aggregation
        output = self.temporal_aggregator(outputs_stacked)

        return output


# =============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Calculate elapsed time in minutes and seconds."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_epoch(model: nn.Module, dataloader, optimizer, criterion, device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    all_labels = []
    all_outputs = []

    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

        all_labels.append(labels.cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    train_auroc = roc_auc_score(all_labels, all_outputs)

    return train_loss / len(dataloader), train_auroc


def evaluate(model: nn.Module, dataloader, criterion, device) -> Tuple[float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    auroc = roc_auc_score(all_labels, all_outputs)

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
    num_epochs: int = 50,
    lr: float = 0.001,
    target_bands: List[str] = ['alpha', 'beta', 'gamma'],
    sampling_rate: float = 160.0,
    checkpoint_dir: str = None,
    resume: bool = False,
    args=None
):
    """Run full training pipeline."""

    print(f"\n{'='*60}")
    print("Starting Periodic-Aware QTCN Training")
    print(f"{'='*60}")

    set_all_seeds(seed)
    print(f"Random Seed: {seed}")

    # Create model
    model = PeriodicAwareQTCN(
        n_qubits=n_qubits,
        circuit_depth=circuit_depth,
        input_dim=input_dim,
        kernel_size=kernel_size,
        dilation=dilation,
        sampling_rate=sampling_rate,
        target_bands=target_bands
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Checkpoint handling
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(RESULTS_DIR, 'PA_QTCN_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    bands_str = "_".join(target_bands)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"pa_qtcn_bands{bands_str}_qubits{n_qubits}_seed{seed}.pth"
    )

    start_epoch = 0
    train_metrics, valid_metrics = [], []
    best_val_auc = 0.0

    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_metrics = checkpoint['train_metrics']
        valid_metrics = checkpoint['valid_metrics']
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        print(f"Resuming from epoch {start_epoch + 1}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_metrics.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_auc': train_auc})

        valid_loss, valid_auc = evaluate(model, val_loader, criterion, device)
        valid_metrics.append({'epoch': epoch + 1, 'valid_loss': valid_loss, 'valid_auc': valid_auc})

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # Update best model
        if valid_auc > best_val_auc:
            best_val_auc = valid_auc
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"\nEpoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, AUC: {valid_auc:.4f} (Best: {best_val_auc:.4f})")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'best_val_auc': best_val_auc
        }, checkpoint_path)

    # Load best model for testing
    model.load_state_dict(best_model_state)

    # Final test evaluation
    test_loss, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"Final Test Results")
    print(f"Test Loss: {test_loss:.4f}, AUC: {test_auc:.4f}")
    print(f"{'='*60}\n")

    # Save metrics
    test_metrics = [{'epoch': num_epochs, 'test_loss': test_loss, 'test_auc': test_auc}]

    metrics = []
    for i in range(len(train_metrics)):
        metrics.append({
            'epoch': i + 1,
            'train_loss': train_metrics[i]['train_loss'],
            'train_auc': train_metrics[i]['train_auc'],
            'valid_loss': valid_metrics[i]['valid_loss'],
            'valid_auc': valid_metrics[i]['valid_auc'],
            'test_loss': test_metrics[0]['test_loss'],
            'test_auc': test_metrics[0]['test_auc']
        })

    metrics_dir = os.path.join(RESULTS_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    csv_filename = os.path.join(metrics_dir, f"PA_QTCN_bands{bands_str}_qubits{n_qubits}_seed{seed}_metrics.csv")
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    return test_loss, test_auc, model


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = get_args()

    # Load data
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=args.seed,
        device=device,
        batch_size=32,
        sampling_freq=args.freq,
        sample_size=args.n_sample
    )

    print(f"Input Dimension: {input_dim}")
    print(f"Target Bands: {args.target_bands}")

    # Determine kernel_size and dilation based on sampling frequency
    if args.freq == 80:
        kernel_size, dilation = 12, 3
    elif args.freq == 4:
        kernel_size, dilation = 7, 2
    else:
        # Default values
        kernel_size, dilation = 10, 2

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
        num_epochs=args.num_epochs,
        lr=args.lr,
        target_bands=args.target_bands,
        sampling_rate=float(args.freq),
        resume=args.resume,
        args=args
    )

    print(f"\nTraining Complete!")
    print(f"Final Test AUC: {test_auc:.4f}")
