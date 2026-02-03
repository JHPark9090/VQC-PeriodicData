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

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Optional, List
import time
import os
import copy
import random

# Try to import data generator
try:
    from data.narma_generator import get_narma_data
except ImportError:
    print("Warning: NARMA data generator not found.")


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
        n_outputs: int = 1
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.vqc_depth = vqc_depth
        self.n_outputs = n_outputs

        # =================================================================
        # LEARNABLE FREQUENCY SCALING (Essential for periodic advantage)
        # =================================================================
        # Each qubit learns its optimal frequency scale
        # Similar to Snake activation's learnable 'a' parameter
        self.freq_scale = nn.Parameter(
            torch.linspace(0.5, 3.0, n_qubits)
        )

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
        window_size: int = 8
    ):
        """
        Args:
            input_size: Input dimension (1 for univariate)
            hidden_size: Hidden state dimension (in frequency domain)
            n_qubits: Number of qubits for VQC
            vqc_depth: Depth of variational circuit
            n_frequencies: Number of FFT frequencies to use
            window_size: Size of input window for FFT
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.window_size = window_size

        # Number of frequencies from FFT (rfft gives n/2 + 1 frequencies)
        self.n_frequencies = n_frequencies if n_frequencies else window_size // 2 + 1

        # =================================================================
        # INPUT PROJECTION (FFT features → n_qubits)
        # =================================================================
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
        self.input_gate = FrequencyMatchedVQC(n_qubits, vqc_depth, hidden_size)
        self.forget_gate = FrequencyMatchedVQC(n_qubits, vqc_depth, hidden_size)
        self.cell_gate = FrequencyMatchedVQC(n_qubits, vqc_depth, hidden_size)
        self.output_gate = FrequencyMatchedVQC(n_qubits, vqc_depth, hidden_size)

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
        # =================================================================
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
        n_frequencies: int = None
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
            window_size=window_size
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

def main():
    """Main function to demonstrate Fourier-QLSTM."""

    print_comparison()

    # Configuration
    set_all_seeds(2025)

    # Model parameters
    input_size = 1      # Univariate time-series
    hidden_size = 4     # Frequency-domain hidden size
    n_qubits = 6        # Number of qubits
    vqc_depth = 2       # VQC depth
    output_size = 1     # Prediction dimension
    window_size = 8     # FFT window size

    # Training parameters
    batch_size = 10
    n_epochs = 50
    lr = 0.01

    print(f"\n{'='*60}")
    print("Fourier-QLSTM Training")
    print(f"{'='*60}")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"N qubits: {n_qubits}")
    print(f"VQC depth: {vqc_depth}")
    print(f"Window size: {window_size}")
    print(f"{'='*60}\n")

    # Generate NARMA data
    try:
        x, y = get_narma_data(n_0=10, seq_len=window_size)
        y = y.unsqueeze(1)
    except:
        print("Generating synthetic periodic data...")
        # Generate synthetic periodic data
        t = torch.linspace(0, 10 * np.pi, 500)
        x = torch.sin(t) + 0.5 * torch.sin(3 * t) + 0.1 * torch.randn_like(t)
        y = torch.sin(t + 0.5).unsqueeze(1)  # Predict phase-shifted version

        # Create sequences
        seq_len = window_size + 4
        x_seqs = []
        y_seqs = []
        for i in range(len(x) - seq_len):
            x_seqs.append(x[i:i + seq_len])
            y_seqs.append(y[i + seq_len - 1])
        x = torch.stack(x_seqs)
        y = torch.stack(y_seqs)

    # Train/test split
    n_train = int(0.67 * len(x))
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input shape: {x_train.shape}")

    # Create model
    model = FourierQLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        n_qubits=n_qubits,
        vqc_depth=vqc_depth,
        output_size=output_size,
        window_size=window_size
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

    return model, train_losses, test_losses


if __name__ == "__main__":
    main()
