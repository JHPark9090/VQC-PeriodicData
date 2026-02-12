"""
Snake-LSTM: Classical Baseline using Snake Activation (Ziyin et al., NeurIPS 2020)

Snake activation: Snake_a(x) = x + (1/a) * sin²(a*x)
This is a PARTIAL periodic activation — it has a linear term (x) plus
a periodic term (sin²), enabling both trend modeling and periodic learning.

Snake is stronger than ReLU/tanh for periodic data but weaker than SIREN
because SIREN computes pure Fourier series while Snake adds periodicity
on top of a linear backbone.

This implementation mirrors FourierQLSTM.py exactly:
- Same FFT preprocessing (_extract_fft_features)
- Same rescaled gating: (output + 1) / 2 (not sigmoid)
- Same frequency-domain cell state (c_mag, c_phase)
- 4 SnakeGate instances replace 4 FrequencyMatchedVQC instances

Based on:
- Ziyin et al. (2020): Neural networks fail to learn periodic functions and how to fix it
- Schuld et al. (2021): VQC as Fourier series

Author: Snake classical baseline for VQC comparison
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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# SNAKE ACTIVATION (Ziyin et al., NeurIPS 2020)
# =============================================================================

class SnakeActivation(nn.Module):
    """
    Snake activation: Snake_a(x) = x + (1/a) * sin²(a*x)

    Properties:
    - Has both linear (x) and periodic (sin²) components
    - Learnable frequency parameter 'a' (analogous to VQC's freq_scale)
    - Reduces to identity when a→0
    - Periodic component has frequency 2a

    From Ziyin et al. (2020): "Neural networks fail to learn periodic
    functions and how to fix it"
    """

    def __init__(self, in_features: int, a_init: float = 1.0, learnable_a: bool = True):
        super().__init__()
        if learnable_a:
            self.a = nn.Parameter(torch.ones(in_features) * a_init)
        else:
            self.register_buffer('a', torch.ones(in_features) * a_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Snake_a(x) = x + (1/a) * sin²(a*x)
        # Use a.abs() + eps to prevent division by zero
        a = self.a.abs() + 1e-8
        return x + (1.0 / a) * torch.sin(a * x) ** 2

    def get_a_values(self) -> np.ndarray:
        return self.a.detach().cpu().numpy()


# =============================================================================
# SNAKE LAYER
# =============================================================================

class SnakeLayer(nn.Module):
    """Linear + Snake activation."""

    def __init__(self, in_features, out_features, a_init=1.0, learnable_a=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = SnakeActivation(out_features, a_init, learnable_a)

    def forward(self, x):
        return self.activation(self.linear(x))

    def get_a_values(self):
        return self.activation.get_a_values()


# =============================================================================
# SNAKE GATE (replaces FrequencyMatchedVQC)
# =============================================================================

class SnakeGate(nn.Module):
    """
    Multi-layer Snake network replacing FrequencyMatchedVQC.

    Input [batch, input_dim] -> Output [batch, output_dim] in [-1, 1]
    Output bounded via tanh to match VQC's PauliZ range.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64, n_layers=2,
                 a_init=1.0, learnable_a=True):
        super().__init__()

        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(SnakeLayer(in_d, hidden_dim, a_init, learnable_a))
            in_d = hidden_dim

        self.snake_layers = nn.ModuleList(layers)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = x
        for layer in self.snake_layers:
            h = layer(h)
        return torch.tanh(self.output_linear(h))

    def get_a_values(self) -> List[np.ndarray]:
        return [layer.get_a_values() for layer in self.snake_layers]


# =============================================================================
# SNAKE-LSTM CELL (mirrors FourierQLSTMCell)
# =============================================================================

class SnakeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, vqc_depth,
                 n_frequencies=None, window_size=8, hidden_dim=64,
                 a_init=1.0, learnable_a=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.window_size = window_size
        self.n_frequencies = n_frequencies if n_frequencies else window_size // 2 + 1

        self.fft_feature_dim = self.n_frequencies * 2 * input_size
        self.input_projection = nn.Linear(self.fft_feature_dim, n_qubits)
        self.hidden_projection = nn.Linear(hidden_size * 2, n_qubits)

        self.input_gate = SnakeGate(n_qubits, hidden_size, hidden_dim, vqc_depth, a_init, learnable_a)
        self.forget_gate = SnakeGate(n_qubits, hidden_size, hidden_dim, vqc_depth, a_init, learnable_a)
        self.cell_gate = SnakeGate(n_qubits, hidden_size, hidden_dim, vqc_depth, a_init, learnable_a)
        self.output_gate = SnakeGate(n_qubits, hidden_size, hidden_dim, vqc_depth, a_init, learnable_a)

        self.output_projection = nn.Linear(hidden_size, input_size)
        self._print_config()

    def _print_config(self):
        print(f"\n{'='*60}")
        print("Snake-LSTM Cell Initialized")
        print(f"{'='*60}")
        print(f"Input size: {self.input_size}, Hidden size: {self.hidden_size}")
        print(f"Snake input dim (n_qubits equiv): {self.n_qubits}")
        print(f"Window: {self.window_size}, Frequencies: {self.n_frequencies}")
        print(f"{'='*60}\n")

    def _extract_fft_features(self, x):
        if x.dim() == 2: x = x.unsqueeze(-1)
        batch_size = x.shape[0]
        fft_result = torch.fft.rfft(x, dim=1)
        fft_selected = fft_result[:, :self.n_frequencies, :]
        magnitude = torch.log1p(torch.abs(fft_selected))
        phase = torch.angle(fft_selected) / np.pi
        return torch.cat([magnitude, phase], dim=1).reshape(batch_size, -1)

    def _rescale_gate(self, x): return (x + 1) / 2

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            c_mag = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c_phase = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            c_mag, c_phase = hidden

        fft_features = self._extract_fft_features(x)
        x_proj = torch.tanh(self.input_projection(fft_features)) * np.pi
        h_proj = torch.tanh(self.hidden_projection(torch.cat([c_mag, c_phase], dim=-1))) * np.pi
        combined = x_proj + h_proj

        i_t = self._rescale_gate(self.input_gate(combined))
        f_t = self._rescale_gate(self.forget_gate(combined))
        g_t = self.cell_gate(combined)
        o_t = self._rescale_gate(self.output_gate(combined))

        c_mag_new = f_t * c_mag + i_t * torch.abs(g_t)
        c_phase_new = c_phase + i_t * torch.atan2(
            torch.sin(g_t * np.pi), torch.cos(g_t * np.pi)) / np.pi
        c_phase_new = torch.remainder(c_phase_new + 1, 2) - 1

        h_t = o_t * c_mag_new * torch.cos(c_phase_new * np.pi)
        return self.output_projection(h_t), (c_mag_new, c_phase_new)


# =============================================================================
# SNAKE-LSTM MODEL
# =============================================================================

class Snake_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, vqc_depth,
                 output_size=1, window_size=8, n_frequencies=None,
                 hidden_dim=64, a_init=1.0, learnable_a=True):
        super().__init__()
        self.window_size = window_size
        self.cell = SnakeLSTMCell(input_size, hidden_size, n_qubits, vqc_depth,
                                  n_frequencies, window_size, hidden_dim, a_init, learnable_a)
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x, hidden=None):
        if x.dim() == 2: x = x.unsqueeze(-1)
        batch_size, seq_len, input_size = x.shape
        outputs = []
        for t in range(seq_len - self.window_size + 1):
            window = x[:, t:t+self.window_size, :]
            window = window.squeeze(-1) if input_size == 1 else window
            out, hidden = self.cell(window, hidden)
            outputs.append(out)
        return self.output_layer(torch.stack(outputs, dim=1)), hidden


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_epoch(model, X, Y, optimizer, batch_size):
    model.train(); losses = []
    for i in range(0, X.shape[0], batch_size):
        optimizer.zero_grad()
        out, _ = model(X[i:i+batch_size])
        loss = F.mse_loss(out[:, -1, :], Y[i:i+batch_size])
        loss.backward(); optimizer.step(); losses.append(loss.item())
    return np.mean(losses)

def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        out, _ = model(X)
        return F.mse_loss(out[:, -1, :], Y).item()

def get_args():
    import argparse
    p = argparse.ArgumentParser(description="Snake-LSTM for NARMA Prediction")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--vqc-depth", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=4)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--snake-hidden-dim", type=int, default=64)
    p.add_argument("--a-init", type=float, default=1.0, help="Initial Snake frequency parameter")
    p.add_argument("--learnable-a", action="store_true", default=True)
    p.add_argument("--no-learnable-a", action="store_false", dest="learnable_a")
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--narma-order", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=500)
    return p.parse_args()

def main():
    args = get_args()
    set_all_seeds(args.seed)

    print(f"\n{'='*60}")
    print("Snake-LSTM Training (Partial Periodic Baseline)")
    print(f"{'='*60}")
    print(f"Hidden: {args.hidden_size}, Input dim: {args.n_qubits}, Depth: {args.vqc_depth}")
    print(f"Window: {args.window_size}, Snake a_init: {args.a_init}")
    print(f"Learnable a: {args.learnable_a}, NARMA order: {args.narma_order}")
    print(f"{'='*60}\n")

    try:
        x, y = get_narma_data(n_0=args.narma_order, seq_len=args.window_size,
                              n_samples=args.n_samples, seed=args.seed)
        y = y.unsqueeze(1)
    except Exception as e:
        print(f"NARMA failed ({e}), using synthetic data...")
        t = torch.linspace(0, 10*np.pi, 500)
        x = torch.sin(t) + 0.5*torch.sin(3*t) + 0.1*torch.randn_like(t)
        y = torch.sin(t+0.5).unsqueeze(1)
        seq_len = args.window_size + 4
        x_s, y_s = [], []
        for i in range(len(x)-seq_len):
            x_s.append(x[i:i+seq_len]); y_s.append(y[i+seq_len-1])
        x, y = torch.stack(x_s), torch.stack(y_s)

    n_train = int(0.67 * len(x))
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    print(f"Train: {len(x_train)}, Test: {len(x_test)}")

    model = Snake_LSTM(1, args.hidden_size, args.n_qubits, args.vqc_depth,
                       1, args.window_size, hidden_dim=args.snake_hidden_dim,
                       a_init=args.a_init, learnable_a=args.learnable_a).float()
    n_params = count_parameters(model)
    print(f"Total trainable parameters: {n_params}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    print(f"\nTraining...")
    for epoch in range(args.n_epochs):
        t0 = time.time()
        tl = train_epoch(model, x_train, y_train, optimizer, args.batch_size)
        te = evaluate(model, x_test, y_test)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{args.n_epochs}: Train={tl:.6f}, Test={te:.6f} ({time.time()-t0:.1f}s)")
            # Print learned 'a' values
            a_vals = model.cell.input_gate.get_a_values()
            print(f"  Input gate 'a' range: [{min(a.min() for a in a_vals):.2f}, {max(a.max() for a in a_vals):.2f}]")

    print(f"\n{'='*60}")
    print(f"Training Complete! Train={tl:.6f}, Test={te:.6f}, Params={n_params}")

    # Print final learned 'a' values
    print("\nLearned Snake 'a' values:")
    for name, gate in [("Input", model.cell.input_gate), ("Forget", model.cell.forget_gate),
                       ("Cell", model.cell.cell_gate), ("Output", model.cell.output_gate)]:
        a_vals = gate.get_a_values()
        for i, a in enumerate(a_vals):
            print(f"  {name} Gate Layer {i}: mean={a.mean():.3f}, std={a.std():.3f}")
    print(f"{'='*60}")
    return model

if __name__ == "__main__":
    main()
