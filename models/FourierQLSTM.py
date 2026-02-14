"""
Fourier-QLSTM: Quantum LSTM with Full Periodic Advantage Utilization

This implementation addresses all issues identified in QLSTM_v0.py:
1. Uses FFT preprocessing for frequency extraction (lossless)
2. Uses frequency-matched VQC encoding (RY + scaled RX)
3. Replaces sigmoid/tanh with rescaled VQC output (preserves periodicity)
4. Uses frequency-domain memory (magnitude + phase cell state)

Design Philosophy:
- Keep LSTM gate structure (familiar, proven)
- Work entirely in frequency domain (aligns with VQC's Fourier nature)
- Minimal classical computation (preserve quantum advantage)

Based on theoretical foundations from:
- Schuld et al. (2021): VQC as Fourier series
- Ziyin et al. (2020): Periodic function learning

Author: Fourier-QLSTM redesign based on QLSTM_v0.py
Date: February 2026
"""

import scipy.constants  # Must precede pennylane (scipy 1.10 lazy-loading workaround)
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Optional, List
import time
import os
import csv
import copy
import random

# Try to import data generator
try:
    from data.narma_generator import get_narma_data
except ImportError:
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.narma_generator import get_narma_data
    except ImportError:
        print("Warning: NARMA data generator not found.")


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


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
        train_data: DataLoader yielding (inputs, labels) OR
                    Tensor [n_samples, seq_len] or [n_samples, channels, seq_len]
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
# FREQUENCY-MATCHED VQC
# =============================================================================

class FrequencyMatchedVQC(nn.Module):
    """
    VQC with frequency-matched encoding for periodic advantage.

    Key differences from standard VQC:
    1. Uses RY + frequency-scaled RX encoding (not just RY)
    2. Learnable freq_scale parameter per qubit
    3. No post-processing that destroys periodicity

    The encoding ensures VQC's Fourier spectrum aligns with input frequencies.
    """

    def __init__(
        self,
        n_qubits: int,
        vqc_depth: int,
        n_outputs: int = 1,
        freq_scale_init: torch.Tensor = None,
        ablate_freq_match: bool = False
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.vqc_depth = vqc_depth
        self.n_outputs = n_outputs
        self.ablate_freq_match = ablate_freq_match

        # =================================================================
        # LEARNABLE FREQUENCY SCALING (Essential for periodic advantage)
        # =================================================================
        # Each qubit learns its optimal frequency scale
        # Similar to Snake activation's learnable 'a' parameter
        if freq_scale_init is not None:
            self.freq_scale = nn.Parameter(freq_scale_init.clone().float())
        else:
            self.freq_scale = nn.Parameter(torch.linspace(0.5, 3.0, n_qubits))

        # Variational parameters for entangling layers
        self.var_params = nn.Parameter(
            torch.randn(vqc_depth, n_qubits) * 0.1
        )

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.circuit = qml.QNode(
            self._circuit_fn,
            self.dev,
            interface='torch',
            diff_method='backprop'
        )

    def _circuit_fn(self, inputs: torch.Tensor) -> List:
        """
        Quantum circuit with frequency-matched encoding.

        Encoding: RY(x) + RX(freq_scale * x)
        This ensures VQC's frequency spectrum matches input data.
        """
        # =================================================================
        # FREQUENCY-MATCHED ENCODING
        # =================================================================
        for i in range(self.n_qubits):
            # Hadamard for superposition
            qml.Hadamard(wires=i)

            # RY for amplitude encoding
            qml.RY(inputs[i], wires=i)

            # RX with learnable frequency scaling (KEY for periodic advantage)
            # Ablation B: skip RX to reduce expressible frequencies (9^n → 3^n)
            if not self.ablate_freq_match:
                qml.RX(self.freq_scale[i] * inputs[i], wires=i)

        # =================================================================
        # VARIATIONAL LAYERS
        # =================================================================
        for layer in range(self.vqc_depth):
            # Entangling layer (CNOT ladder)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Parameterized rotations
            for i in range(self.n_qubits):
                qml.RY(self.var_params[layer, i], wires=i)

        # Measure first n_outputs qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_outputs)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VQC.

        Args:
            x: Input tensor [batch, n_qubits]

        Returns:
            Output tensor [batch, n_outputs]
        """
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            out = self.circuit(x[i])
            if isinstance(out, list):
                out = torch.stack(out)
            outputs.append(out)

        return torch.stack(outputs)

    def get_freq_scales(self) -> np.ndarray:
        """Return learned frequency scales for analysis."""
        return self.freq_scale.detach().cpu().numpy()


# =============================================================================
# FOURIER-QLSTM CELL
# =============================================================================

class FourierQLSTMCell(nn.Module):
    """
    QLSTM Cell with frequency-domain memory and periodic advantage.

    Key innovations:
    1. FFT preprocessing: Extract frequency components (lossless)
    2. Frequency-matched VQC: RY + scaled RX encoding
    3. Rescaled gating: (VQC + 1) / 2 instead of sigmoid (preserves periodicity)
    4. Frequency-domain memory: Cell state as (magnitude, phase)

    This design fully utilizes VQC's Fourier series nature.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int,
        vqc_depth: int,
        n_frequencies: int = None,
        window_size: int = 8,
        freq_scale_init: torch.Tensor = None,
        ablate_fft: bool = False,
        ablate_freq_match: bool = False,
        ablate_rescaled: bool = False
    ):
        """
        Args:
            input_size: Input dimension (1 for univariate)
            hidden_size: Hidden state dimension (in frequency domain)
            n_qubits: Number of qubits for VQC
            vqc_depth: Depth of variational circuit
            n_frequencies: Number of FFT frequencies to use
            window_size: Size of input window for FFT
            freq_scale_init: Optional FFT-seeded initialization for freq_scale
            ablate_fft: Ablation A - bypass FFT, use raw windowed input
            ablate_freq_match: Ablation B - remove RX gate, RY-only encoding
            ablate_rescaled: Ablation C - use sigmoid instead of (x+1)/2
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.window_size = window_size
        self.ablate_fft = ablate_fft
        self.ablate_freq_match = ablate_freq_match
        self.ablate_rescaled = ablate_rescaled

        # Number of frequencies from FFT (rfft gives n/2 + 1 frequencies)
        self.n_frequencies = n_frequencies if n_frequencies else window_size // 2 + 1

        # =================================================================
        # INPUT PROJECTION (FFT features → n_qubits)
        # =================================================================
        # Ablation A: use raw flattened window instead of FFT features
        if self.ablate_fft:
            self.fft_feature_dim = window_size * input_size
        else:
            # FFT gives magnitude + phase, so 2 * n_frequencies features
            self.fft_feature_dim = self.n_frequencies * 2 * input_size
        self.input_projection = nn.Linear(self.fft_feature_dim, n_qubits)

        # =================================================================
        # HIDDEN STATE PROJECTION (frequency-domain hidden → n_qubits)
        # =================================================================
        # Hidden state is also in frequency domain (magnitude + phase)
        self.hidden_projection = nn.Linear(hidden_size * 2, n_qubits)

        # =================================================================
        # VQC GATES (4 circuits, one for each LSTM gate)
        # =================================================================
        # These replace classical Linear + sigmoid/tanh
        self.input_gate = FrequencyMatchedVQC(n_qubits, vqc_depth, hidden_size, freq_scale_init=freq_scale_init, ablate_freq_match=ablate_freq_match)
        self.forget_gate = FrequencyMatchedVQC(n_qubits, vqc_depth, hidden_size, freq_scale_init=freq_scale_init, ablate_freq_match=ablate_freq_match)
        self.cell_gate = FrequencyMatchedVQC(n_qubits, vqc_depth, hidden_size, freq_scale_init=freq_scale_init, ablate_freq_match=ablate_freq_match)
        self.output_gate = FrequencyMatchedVQC(n_qubits, vqc_depth, hidden_size, freq_scale_init=freq_scale_init, ablate_freq_match=ablate_freq_match)

        # =================================================================
        # OUTPUT PROJECTION
        # =================================================================
        self.output_projection = nn.Linear(hidden_size, input_size)

        self._print_config()

    def _print_config(self):
        """Print cell configuration."""
        print(f"\n{'='*60}")
        print("Fourier-QLSTM Cell Initialized")
        print(f"{'='*60}")
        print(f"Input size: {self.input_size}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Window size: {self.window_size}")
        print(f"Number of frequencies: {self.n_frequencies}")
        print(f"FFT feature dimension: {self.fft_feature_dim}")
        print(f"{'='*60}\n")

    def _extract_fft_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency-domain features using FFT.

        This is LOSSLESS (unlike bandpass filtering) and naturally
        aligns with VQC's Fourier series structure.

        Args:
            x: Input [batch, window_size] or [batch, window_size, input_size]

        Returns:
            features: [batch, fft_feature_dim]
        """
        # Handle univariate vs multivariate
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, window, 1]

        batch_size = x.shape[0]

        # Apply FFT along time dimension
        # rfft for real input, gives positive frequencies only
        fft_result = torch.fft.rfft(x, dim=1)  # [batch, n_freq, input_size]

        # Select frequencies
        fft_selected = fft_result[:, :self.n_frequencies, :]

        # Extract magnitude and phase
        magnitude = torch.log1p(torch.abs(fft_selected))  # Log scale
        phase = torch.angle(fft_selected) / np.pi  # Normalize to [-1, 1]

        # Concatenate and flatten
        features = torch.cat([magnitude, phase], dim=1)  # [batch, 2*n_freq, input_size]
        features = features.reshape(batch_size, -1)  # [batch, fft_feature_dim]

        return features

    def _rescale_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rescale VQC output from [-1, 1] to [0, 1].

        This replaces sigmoid and PRESERVES periodicity!
        VQC outputs are already bounded by PauliZ measurement.

        sigmoid(x) destroys periodicity
        (x + 1) / 2 preserves periodicity
        """
        if self.ablate_rescaled:
            return torch.sigmoid(x)
        return (x + 1) / 2

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Fourier-QLSTM cell.

        Args:
            x: Input tensor [batch, window_size] or [batch, window_size, input_size]
            hidden: Tuple of (c_magnitude, c_phase), both [batch, hidden_size]

        Returns:
            output: [batch, input_size]
            hidden: (c_magnitude, c_phase) for next step
        """
        batch_size = x.shape[0]

        # Initialize hidden state if not provided
        if hidden is None:
            c_mag = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c_phase = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            c_mag, c_phase = hidden

        # =================================================================
        # STEP 1: FFT Preprocessing (Lossless frequency extraction)
        # Ablation A: bypass FFT, use raw flattened window
        # =================================================================
        if self.ablate_fft:
            fft_features = x.reshape(batch_size, -1) if x.dim() == 3 else x
        else:
            fft_features = self._extract_fft_features(x)

        # =================================================================
        # STEP 2: Project to qubit dimension
        # =================================================================
        x_proj = self.input_projection(fft_features)
        x_proj = torch.tanh(x_proj) * np.pi  # Normalize for angle encoding

        # Project hidden state (also frequency domain)
        h_features = torch.cat([c_mag, c_phase], dim=-1)
        h_proj = self.hidden_projection(h_features)
        h_proj = torch.tanh(h_proj) * np.pi

        # Combine input and hidden projections
        combined = x_proj + h_proj  # Additive combination

        # =================================================================
        # STEP 3: VQC Gates with Frequency-Matched Encoding
        # =================================================================
        # Input gate: controls what new information to store
        i_t = self._rescale_gate(self.input_gate(combined))  # [0, 1]

        # Forget gate: controls what to forget from cell state
        f_t = self._rescale_gate(self.forget_gate(combined))  # [0, 1]

        # Cell gate: candidate values (no activation needed, VQC output is bounded)
        g_t = self.cell_gate(combined)  # [-1, 1]

        # Output gate: controls what to output
        o_t = self._rescale_gate(self.output_gate(combined))  # [0, 1]

        # =================================================================
        # STEP 4: Frequency-Domain Cell State Update
        # =================================================================
        # Update magnitude (similar to classical LSTM)
        c_mag_new = f_t * c_mag + i_t * torch.abs(g_t)

        # Update phase (additive, preserves periodic structure)
        # Phase accumulates over time, which is natural for periodic signals
        c_phase_new = c_phase + i_t * torch.atan2(
            torch.sin(g_t * np.pi),
            torch.cos(g_t * np.pi)
        ) / np.pi

        # Normalize phase to [-1, 1] to prevent unbounded growth
        c_phase_new = torch.remainder(c_phase_new + 1, 2) - 1

        # =================================================================
        # STEP 5: Output Generation
        # =================================================================
        # Combine magnitude and phase for output
        # This is similar to h_t = o_t * tanh(c_t) but preserves periodicity
        h_t = o_t * c_mag_new * torch.cos(c_phase_new * np.pi)

        # Project to output dimension
        output = self.output_projection(h_t)

        return output, (c_mag_new, c_phase_new)


# =============================================================================
# FOURIER-QLSTM MODEL
# =============================================================================

class FourierQLSTM(nn.Module):
    """
    Full Fourier-QLSTM model for sequence processing.

    Processes sequences by:
    1. Sliding window over input sequence
    2. FFT each window
    3. Process through Fourier-QLSTM cell
    4. Aggregate outputs
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int,
        vqc_depth: int,
        output_size: int = 1,
        window_size: int = 8,
        n_frequencies: int = None,
        freq_scale_init: torch.Tensor = None,
        ablate_fft: bool = False,
        ablate_freq_match: bool = False,
        ablate_rescaled: bool = False
    ):
        super().__init__()

        self.window_size = window_size
        self.hidden_size = hidden_size

        # QLSTM Cell
        self.cell = FourierQLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            vqc_depth=vqc_depth,
            n_frequencies=n_frequencies,
            window_size=window_size,
            freq_scale_init=freq_scale_init,
            ablate_fft=ablate_fft,
            ablate_freq_match=ablate_freq_match,
            ablate_rescaled=ablate_rescaled
        )

        # Final output layer
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequence through Fourier-QLSTM.

        Args:
            x: Input sequence [batch, seq_len, input_size] or [batch, seq_len]
            hidden: Initial hidden state (optional)

        Returns:
            outputs: [batch, seq_len - window_size + 1, output_size]
            final_hidden: (c_mag, c_phase)
        """
        # Handle different input shapes
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        batch_size, seq_len, input_size = x.shape

        outputs = []

        # Process sequence with sliding window
        for t in range(seq_len - self.window_size + 1):
            # Extract window
            window = x[:, t:t + self.window_size, :]  # [batch, window_size, input_size]
            window = window.squeeze(-1) if input_size == 1 else window

            # Process through cell
            out, hidden = self.cell(window, hidden)
            outputs.append(out)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch, n_windows, input_size]

        # Final projection
        outputs = self.output_layer(outputs)

        return outputs, hidden


# =============================================================================
# COMPARISON: OLD QLSTM vs FOURIER-QLSTM
# =============================================================================

def print_comparison():
    """Print comparison between old QLSTM and Fourier-QLSTM."""
    comparison = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              OLD QLSTM vs FOURIER-QLSTM                                  ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                          ║
    ║  OLD QLSTM (QLSTM_v0.py):                                                ║
    ║  ├─ Input: Raw time-domain values                                        ║
    ║  ├─ Encoding: Generic RY only                                            ║
    ║  ├─ Gates: sigmoid(VQC) / tanh(VQC) ← DESTROYS PERIODICITY               ║
    ║  ├─ Memory: Classical (c_t, h_t)                                         ║
    ║  └─ VQC Periodic Advantage: NOT UTILIZED                                 ║
    ║                                                                          ║
    ║  FOURIER-QLSTM:                                                          ║
    ║  ├─ Input: FFT → (magnitude, phase)                                      ║
    ║  ├─ Encoding: RY(x) + RX(freq_scale * x)                                 ║
    ║  ├─ Gates: rescale(VQC) = (VQC + 1) / 2 ← PRESERVES PERIODICITY          ║
    ║  ├─ Memory: Frequency-domain (c_magnitude, c_phase)                      ║
    ║  └─ VQC Periodic Advantage: FULLY UTILIZED                               ║
    ║                                                                          ║
    ║  KEY DIFFERENCES:                                                        ║
    ║  ┌────────────────────┬────────────────────┬────────────────────────┐    ║
    ║  │ Aspect             │ Old QLSTM          │ Fourier-QLSTM          │    ║
    ║  ├────────────────────┼────────────────────┼────────────────────────┤    ║
    ║  │ Input Processing   │ Raw values         │ FFT (lossless)         │    ║
    ║  │ Frequency Matching │ None               │ Learnable freq_scale   │    ║
    ║  │ Gate Activation    │ sigmoid/tanh       │ Rescaling (periodic)   │    ║
    ║  │ Cell State         │ Single tensor      │ (magnitude, phase)     │    ║
    ║  │ Phase Handling     │ None               │ Additive accumulation  │    ║
    ║  │ Periodic Advantage │ Destroyed          │ Preserved              │    ║
    ║  └────────────────────┴────────────────────┴────────────────────────┘    ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    print(comparison)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_epoch(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int
) -> float:
    """Train for one epoch."""
    model.train()
    losses = []

    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]

        optimizer.zero_grad()

        outputs, _ = model(X_batch)

        # Use last output for prediction
        predictions = outputs[:, -1, :]

        loss = F.mse_loss(predictions, Y_batch)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor
) -> float:
    """Evaluate model on test set."""
    model.eval()

    with torch.no_grad():
        outputs, _ = model(X)
        predictions = outputs[:, -1, :]
        loss = F.mse_loss(predictions, Y)

    return loss.item()


# =============================================================================
# MAIN
# =============================================================================

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Fourier-QLSTM for Time-Series Prediction")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--vqc-depth", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--narma-order", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--dataset", type=str, default="narma",
                        choices=["narma", "multisine", "mackey_glass", "adding"],
                        help="Dataset to use (default: narma)")
    parser.add_argument("--freq-init", type=str, default="fft",
                        choices=["fft", "linspace", "random"],
                        help="freq_scale initialization: 'fft' (data-informed), 'linspace' (generic), or 'random' (ablation D)")
    parser.add_argument("--ablate-fft", action="store_true",
                        help="Ablation A: bypass FFT, use raw windowed input")
    parser.add_argument("--ablate-freq-match", action="store_true",
                        help="Ablation B: remove RX gate, RY-only encoding")
    parser.add_argument("--ablate-rescaled", action="store_true",
                        help="Ablation C: use sigmoid instead of (x+1)/2")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output metrics CSV")
    return parser.parse_args()


def main():
    """Main function to demonstrate Fourier-QLSTM."""

    print_comparison()

    args = get_args()

    # Force linspace init when ablating FFT (can't do FFT init without FFT)
    if args.ablate_fft and args.freq_init == 'fft':
        print("Note: --ablate-fft forces --freq-init=linspace (can't do FFT init without FFT)")
        args.freq_init = 'linspace'

    # Determine variant name from ablation flags
    variant = "full"
    if args.ablate_fft:
        variant = "no_fft"
    elif args.ablate_freq_match:
        variant = "no_freq_match"
    elif args.ablate_rescaled:
        variant = "no_rescaled"
    elif args.freq_init == "random":
        variant = "no_fft_init"

    # Configuration
    set_all_seeds(args.seed)

    # Model parameters
    input_size = 1
    hidden_size = args.hidden_size
    n_qubits = args.n_qubits
    vqc_depth = args.vqc_depth
    output_size = 1
    window_size = args.window_size

    # Training parameters
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    lr = args.lr

    print(f"\n{'='*60}")
    print("Fourier-QLSTM Training")
    print(f"{'='*60}")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"N qubits: {n_qubits}")
    print(f"VQC depth: {vqc_depth}")
    print(f"Window size: {window_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Variant: {variant}")
    print(f"Freq init: {args.freq_init}")
    print(f"Ablate FFT: {args.ablate_fft}")
    print(f"Ablate Freq-Match: {args.ablate_freq_match}")
    print(f"Ablate Rescaled: {args.ablate_rescaled}")
    print(f"{'='*60}\n")

    # Generate data
    if args.dataset == "narma":
        x, y = get_narma_data(n_0=args.narma_order, seq_len=window_size,
                              n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "multisine":
        from data.multisine_generator import get_multisine_data
        x, y = get_multisine_data(K=5, seq_len=window_size,
                                   n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "mackey_glass":
        from data.mackey_glass_generator import get_mackey_glass_data
        x, y = get_mackey_glass_data(tau=17, seq_len=window_size,
                                      n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "adding":
        from data.adding_problem_generator import get_adding_data
        x, y = get_adding_data(T=window_size, n_samples=args.n_samples,
                                seed=args.seed)
    y = y.unsqueeze(1)

    # Convert to double for PennyLane compatibility
    x = x.double()
    y = y.double()

    # Train/test split
    n_train = int(0.67 * len(x))
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input shape: {x_train.shape}")

    # Compute freq_scale initialization
    freq_scale_init = None
    if args.freq_init == 'fft':
        freq_scale_init = analyze_training_frequencies(x_train, n_qubits)
    elif args.freq_init == 'random':
        freq_scale_init = torch.rand(n_qubits) * 2.5 + 0.5  # uniform [0.5, 3.0]
        print(f"Random freq_scale init: {freq_scale_init.tolist()}")

    # Create model
    model = FourierQLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        n_qubits=n_qubits,
        vqc_depth=vqc_depth,
        output_size=output_size,
        window_size=window_size,
        freq_scale_init=freq_scale_init,
        ablate_fft=args.ablate_fft,
        ablate_freq_match=args.ablate_freq_match,
        ablate_rescaled=args.ablate_rescaled
    ).double()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # Training loop
    train_losses = []
    test_losses = []

    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, x_train, y_train, optimizer, batch_size)
        test_loss = evaluate(model, x_test, y_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{n_epochs}: "
                  f"Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")

            # Print learned frequency scales
            freq_scales = model.cell.input_gate.get_freq_scales()
            print(f"  Freq scales: [{freq_scales.min():.2f}, {freq_scales.max():.2f}]")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Test Loss: {test_losses[-1]:.6f}")
    print(f"{'='*60}")

    # Print final frequency scales for all gates
    print("\nLearned Frequency Scales:")
    for gate_name, gate in [
        ("Input Gate", model.cell.input_gate),
        ("Forget Gate", model.cell.forget_gate),
        ("Cell Gate", model.cell.cell_gate),
        ("Output Gate", model.cell.output_gate)
    ]:
        scales = gate.get_freq_scales()
        print(f"  {gate_name}: {scales}")

    # Save metrics CSV
    output_dir = args.output_dir or os.path.join(RESULTS_DIR, 'phase1_ablation')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir,
        f"FourierQLSTM_{args.dataset}_{variant}_seed{args.seed}_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'test_loss'])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], test_losses[i]])
    print(f"\nMetrics saved to {csv_path}")

    return model, train_losses, test_losses


if __name__ == "__main__":
    main()
