"""
SIREN-LSTM: Classical Fourier Baseline using Sinusoidal Representation Networks

SIREN (Sitzmann et al., NeurIPS 2020) uses sin(w0 * (Wx + b)) as activation,
making it a natural classical Fourier baseline for VQC comparison.

This implementation mirrors FourierQLSTM.py exactly:
- Same FFT preprocessing (_extract_fft_features)
- Same rescaled gating: (output + 1) / 2 (not sigmoid)
- Same frequency-domain cell state (c_mag, c_phase)
- 4 SIRENGate instances replace 4 FrequencyMatchedVQC instances

Key difference: SIREN processes full batches natively (no per-sample loop).

Design Philosophy:
- Mirror FourierQLSTM architecture exactly
- Replace only the VQC gates with SIREN networks
- Isolate the question: does quantum Fourier give advantage over classical Fourier?

Based on:
- Sitzmann et al. (2020): Implicit Neural Representations with Periodic Activation Functions
- Schuld et al. (2021): VQC as Fourier series
- Ziyin et al. (2020): Neural networks fail to learn periodic functions

Author: SIREN classical baseline for VQC comparison
Date: February 2026
"""

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
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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

    This ensures the pre-activation distribution stays in [-pi, pi]
    through the network, preventing saturation of sine.
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
            # Initialize so that softplus(raw) ≈ w0
            raw_init = np.log(np.exp(w0) - 1.0)  # inverse softplus
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
# SIREN GATE (replaces FrequencyMatchedVQC)
# =============================================================================

class SIRENGate(nn.Module):
    """
    Multi-layer SIREN network replacing FrequencyMatchedVQC.

    Input [batch, input_dim] -> Output [batch, output_dim] in [-1, 1]

    The output is bounded via tanh to match VQC's PauliZ measurement range.
    This ensures the same (output + 1) / 2 rescaling works for LSTM gating.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        w0: float = 30.0,
        learnable_w0: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []

        # First SIREN layer
        layers.append(SIRENLayer(
            input_dim, hidden_dim, w0=w0,
            is_first=True, learnable_w0=learnable_w0
        ))

        # Hidden SIREN layers
        for _ in range(n_layers - 1):
            layers.append(SIRENLayer(
                hidden_dim, hidden_dim, w0=w0,
                is_first=False, learnable_w0=learnable_w0
            ))

        self.siren_layers = nn.ModuleList(layers)

        # Final linear layer (no sine activation) -> tanh for [-1, 1] bounding
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        # Initialize output layer with small weights
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / 30.0
            self.output_linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SIREN gate.

        Args:
            x: Input [batch, input_dim]

        Returns:
            Output [batch, output_dim] in [-1, 1]
        """
        h = x
        for layer in self.siren_layers:
            h = layer(h)
        # Output bounded to [-1, 1] to match VQC PauliZ range
        return torch.tanh(self.output_linear(h))

    def get_w0_values(self) -> List[float]:
        """Return learned w0 values for analysis."""
        return [layer.w0.item() for layer in self.siren_layers]


# =============================================================================
# SIREN-LSTM CELL (mirrors FourierQLSTMCell)
# =============================================================================

class SIRENLSTMCell(nn.Module):
    """
    LSTM Cell with SIREN gates replacing VQC gates.

    Mirrors FourierQLSTMCell exactly:
    1. FFT preprocessing: Extract frequency components (lossless)
    2. Rescaled gating: (SIREN + 1) / 2 instead of sigmoid (preserves periodicity)
    3. Frequency-domain memory: Cell state as (magnitude, phase)

    Only difference: SIRENGate replaces FrequencyMatchedVQC.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int,
        vqc_depth: int,
        n_frequencies: int = None,
        window_size: int = 8,
        w0: float = 30.0,
        learnable_w0: bool = True,
        siren_hidden_dim: int = 64
    ):
        """
        Args:
            input_size: Input dimension (1 for univariate)
            hidden_size: Hidden state dimension (in frequency domain)
            n_qubits: Kept for API compatibility (controls SIREN input dim)
            vqc_depth: Kept for API compatibility (controls SIREN depth)
            n_frequencies: Number of FFT frequencies to use
            window_size: Size of input window for FFT
            w0: Initial SIREN frequency (default: 30.0 from paper)
            learnable_w0: Whether w0 is learnable
            siren_hidden_dim: Hidden dimension for SIREN layers
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.window_size = window_size

        # Number of frequencies from FFT (rfft gives n/2 + 1 frequencies)
        self.n_frequencies = n_frequencies if n_frequencies else window_size // 2 + 1

        # =================================================================
        # INPUT PROJECTION (FFT features -> n_qubits)
        # =================================================================
        self.fft_feature_dim = self.n_frequencies * 2 * input_size
        self.input_projection = nn.Linear(self.fft_feature_dim, n_qubits)

        # =================================================================
        # HIDDEN STATE PROJECTION (frequency-domain hidden -> n_qubits)
        # =================================================================
        self.hidden_projection = nn.Linear(hidden_size * 2, n_qubits)

        # =================================================================
        # SIREN GATES (4 networks, one for each LSTM gate)
        # These replace the 4 FrequencyMatchedVQC instances
        # =================================================================
        self.input_gate = SIRENGate(
            n_qubits, hidden_size, hidden_dim=siren_hidden_dim,
            n_layers=vqc_depth, w0=w0, learnable_w0=learnable_w0
        )
        self.forget_gate = SIRENGate(
            n_qubits, hidden_size, hidden_dim=siren_hidden_dim,
            n_layers=vqc_depth, w0=w0, learnable_w0=learnable_w0
        )
        self.cell_gate = SIRENGate(
            n_qubits, hidden_size, hidden_dim=siren_hidden_dim,
            n_layers=vqc_depth, w0=w0, learnable_w0=learnable_w0
        )
        self.output_gate = SIRENGate(
            n_qubits, hidden_size, hidden_dim=siren_hidden_dim,
            n_layers=vqc_depth, w0=w0, learnable_w0=learnable_w0
        )

        # =================================================================
        # OUTPUT PROJECTION
        # =================================================================
        self.output_projection = nn.Linear(hidden_size, input_size)

        self._print_config()

    def _print_config(self):
        """Print cell configuration."""
        print(f"\n{'='*60}")
        print("SIREN-LSTM Cell Initialized")
        print(f"{'='*60}")
        print(f"Input size: {self.input_size}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"SIREN input dim (n_qubits equiv): {self.n_qubits}")
        print(f"Window size: {self.window_size}")
        print(f"Number of frequencies: {self.n_frequencies}")
        print(f"FFT feature dimension: {self.fft_feature_dim}")
        print(f"Initial w0: {self.input_gate.get_w0_values()}")
        print(f"{'='*60}\n")

    def _extract_fft_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency-domain features using FFT.

        Identical to FourierQLSTMCell._extract_fft_features.

        Args:
            x: Input [batch, window_size] or [batch, window_size, input_size]

        Returns:
            features: [batch, fft_feature_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, window, 1]

        batch_size = x.shape[0]

        # Apply FFT along time dimension
        fft_result = torch.fft.rfft(x, dim=1)  # [batch, n_freq, input_size]

        # Select frequencies
        fft_selected = fft_result[:, :self.n_frequencies, :]

        # Extract magnitude and phase
        magnitude = torch.log1p(torch.abs(fft_selected))  # Log scale
        phase = torch.angle(fft_selected) / np.pi  # Normalize to [-1, 1]

        # Concatenate and flatten
        features = torch.cat([magnitude, phase], dim=1)
        features = features.reshape(batch_size, -1)

        return features

    def _rescale_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rescale SIREN output from [-1, 1] to [0, 1].

        Identical to FourierQLSTMCell._rescale_gate.
        SIREN output is bounded by tanh, matching VQC's PauliZ range.
        """
        return (x + 1) / 2

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through SIREN-LSTM cell.

        Identical logic to FourierQLSTMCell.forward but with SIREN gates.

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
        # STEP 2: Project to input dimension
        # =================================================================
        x_proj = self.input_projection(fft_features)
        x_proj = torch.tanh(x_proj) * np.pi  # Normalize for consistency

        # Project hidden state (also frequency domain)
        h_features = torch.cat([c_mag, c_phase], dim=-1)
        h_proj = self.hidden_projection(h_features)
        h_proj = torch.tanh(h_proj) * np.pi

        # Combine input and hidden projections
        combined = x_proj + h_proj  # Additive combination

        # =================================================================
        # STEP 3: SIREN Gates (replacing VQC gates)
        # =================================================================
        # SIREN processes full batch natively (no per-sample loop needed)
        i_t = self._rescale_gate(self.input_gate(combined))   # [0, 1]
        f_t = self._rescale_gate(self.forget_gate(combined))  # [0, 1]
        g_t = self.cell_gate(combined)                        # [-1, 1]
        o_t = self._rescale_gate(self.output_gate(combined))  # [0, 1]

        # =================================================================
        # STEP 4: Frequency-Domain Cell State Update
        # =================================================================
        # Update magnitude (same as FourierQLSTMCell)
        c_mag_new = f_t * c_mag + i_t * torch.abs(g_t)

        # Update phase (additive, preserves periodic structure)
        c_phase_new = c_phase + i_t * torch.atan2(
            torch.sin(g_t * np.pi),
            torch.cos(g_t * np.pi)
        ) / np.pi

        # Normalize phase to [-1, 1]
        c_phase_new = torch.remainder(c_phase_new + 1, 2) - 1

        # =================================================================
        # STEP 5: Output Generation
        # =================================================================
        h_t = o_t * c_mag_new * torch.cos(c_phase_new * np.pi)
        output = self.output_projection(h_t)

        return output, (c_mag_new, c_phase_new)


# =============================================================================
# SIREN-LSTM MODEL (mirrors FourierQLSTM)
# =============================================================================

class SIREN_LSTM(nn.Module):
    """
    Full SIREN-LSTM model for sequence processing.

    Mirrors FourierQLSTM exactly:
    1. Sliding window over input sequence
    2. FFT each window
    3. Process through SIREN-LSTM cell
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
        w0: float = 30.0,
        learnable_w0: bool = True,
        siren_hidden_dim: int = 64
    ):
        super().__init__()

        self.window_size = window_size
        self.hidden_size = hidden_size

        self.cell = SIRENLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_qubits=n_qubits,
            vqc_depth=vqc_depth,
            n_frequencies=n_frequencies,
            window_size=window_size,
            w0=w0,
            learnable_w0=learnable_w0,
            siren_hidden_dim=siren_hidden_dim
        )

        # Final output layer
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process sequence through SIREN-LSTM.

        Args:
            x: Input sequence [batch, seq_len, input_size] or [batch, seq_len]
            hidden: Initial hidden state (optional)

        Returns:
            outputs: [batch, seq_len - window_size + 1, output_size]
            final_hidden: (c_mag, c_phase)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [batch, seq_len, 1]

        batch_size, seq_len, input_size = x.shape

        outputs = []

        # Process sequence with sliding window
        for t in range(seq_len - self.window_size + 1):
            window = x[:, t:t + self.window_size, :]
            window = window.squeeze(-1) if input_size == 1 else window

            out, hidden = self.cell(window, hidden)
            outputs.append(out)

        # Stack outputs
        outputs = torch.stack(outputs, dim=1)

        # Final projection
        outputs = self.output_layer(outputs)

        return outputs, hidden


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
# ARGUMENT PARSING
# =============================================================================

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="SIREN-LSTM for NARMA Prediction")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--n-qubits", type=int, default=6,
                        help="SIREN input dim (equivalent to n_qubits for comparison)")
    parser.add_argument("--vqc-depth", type=int, default=2,
                        help="Number of SIREN layers (equivalent to vqc_depth)")
    parser.add_argument("--hidden-size", type=int, default=4,
                        help="Hidden state dimension")
    parser.add_argument("--window-size", type=int, default=8,
                        help="FFT window size")
    parser.add_argument("--w0", type=float, default=30.0,
                        help="Initial SIREN frequency (default: 30.0)")
    parser.add_argument("--learnable-w0", action="store_true", default=True,
                        help="Make w0 learnable")
    parser.add_argument("--no-learnable-w0", action="store_false", dest="learnable_w0")
    parser.add_argument("--siren-hidden-dim", type=int, default=64,
                        help="Hidden dimension for SIREN layers")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--narma-order", type=int, default=10,
                        help="NARMA order (5, 10, or 30)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of NARMA samples")
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run SIREN-LSTM on NARMA."""

    args = get_args()

    # Configuration
    set_all_seeds(args.seed)

    # Model parameters
    input_size = 1
    hidden_size = args.hidden_size
    n_qubits = args.n_qubits
    vqc_depth = args.vqc_depth
    output_size = 1
    window_size = args.window_size

    print(f"\n{'='*60}")
    print("SIREN-LSTM Training (Classical Fourier Baseline)")
    print(f"{'='*60}")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"SIREN input dim (n_qubits equiv): {n_qubits}")
    print(f"SIREN depth (vqc_depth equiv): {vqc_depth}")
    print(f"Window size: {window_size}")
    print(f"w0 (initial): {args.w0}")
    print(f"Learnable w0: {args.learnable_w0}")
    print(f"SIREN hidden dim: {args.siren_hidden_dim}")
    print(f"NARMA order: {args.narma_order}")
    print(f"{'='*60}\n")

    # Generate NARMA data
    try:
        x, y = get_narma_data(
            n_0=args.narma_order,
            seq_len=window_size,
            n_samples=args.n_samples,
            seed=args.seed
        )
        y = y.unsqueeze(1)
    except Exception as e:
        print(f"NARMA data generation failed ({e}), using synthetic data...")
        t = torch.linspace(0, 10 * np.pi, 500)
        x = torch.sin(t) + 0.5 * torch.sin(3 * t) + 0.1 * torch.randn_like(t)
        y = torch.sin(t + 0.5).unsqueeze(1)

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
    model = SIREN_LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        n_qubits=n_qubits,
        vqc_depth=vqc_depth,
        output_size=output_size,
        window_size=window_size,
        w0=args.w0,
        learnable_w0=args.learnable_w0,
        siren_hidden_dim=args.siren_hidden_dim
    ).float()

    # Print parameter count
    n_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {n_params}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Training loop
    train_losses = []
    test_losses = []

    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}")

    for epoch in range(args.n_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, x_train, y_train, optimizer, args.batch_size)
        test_loss = evaluate(model, x_test, y_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        elapsed = time.time() - start_time

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{args.n_epochs}: "
                  f"Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f} "
                  f"({elapsed:.1f}s)")

            # Print learned w0 values
            w0_vals = model.cell.input_gate.get_w0_values()
            print(f"  Input gate w0: {[f'{v:.2f}' for v in w0_vals]}")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Test Loss: {test_losses[-1]:.6f}")
    print(f"Total parameters: {n_params}")
    print(f"{'='*60}")

    # Print final w0 values for all gates
    print("\nLearned w0 Values:")
    for gate_name, gate in [
        ("Input Gate", model.cell.input_gate),
        ("Forget Gate", model.cell.forget_gate),
        ("Cell Gate", model.cell.cell_gate),
        ("Output Gate", model.cell.output_gate)
    ]:
        w0_vals = gate.get_w0_values()
        print(f"  {gate_name}: {[f'{v:.2f}' for v in w0_vals]}")

    return model, train_losses, test_losses


if __name__ == "__main__":
    main()
