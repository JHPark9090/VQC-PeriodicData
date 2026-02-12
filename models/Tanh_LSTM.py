"""
Tanh-LSTM: Classical Baseline using Tanh Activation

Tanh has LIMITED periodic structure — it's a monotonic sigmoid-like function
that saturates to ±1. While bounded like VQC output, tanh has no inherent
periodicity and fails for periodic extrapolation (saturates to constant).

This implementation mirrors FourierQLSTM.py exactly:
- Same FFT preprocessing (_extract_fft_features)
- Same rescaled gating: (output + 1) / 2 (not sigmoid)
- Same frequency-domain cell state (c_mag, c_phase)
- 4 TanhGate instances replace 4 FrequencyMatchedVQC instances

Based on:
- Schuld et al. (2021): VQC as Fourier series
- Ziyin et al. (2020): Neural networks fail to learn periodic functions

Author: Tanh classical baseline for VQC comparison
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
# TANH GATE (replaces FrequencyMatchedVQC)
# =============================================================================

class TanhGate(nn.Module):
    """
    Multi-layer MLP with tanh activation replacing FrequencyMatchedVQC.

    Input [batch, input_dim] -> Output [batch, output_dim] in [-1, 1]
    Tanh naturally bounds output to [-1, 1], matching VQC's PauliZ range.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64, n_layers=2):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.Tanh())
            in_d = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return torch.tanh(self.output_linear(self.mlp(x)))


# =============================================================================
# TANH-LSTM CELL (mirrors FourierQLSTMCell)
# =============================================================================

class TanhLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, vqc_depth,
                 n_frequencies=None, window_size=8, hidden_dim=64):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.window_size = window_size
        self.n_frequencies = n_frequencies if n_frequencies else window_size // 2 + 1

        self.fft_feature_dim = self.n_frequencies * 2 * input_size
        self.input_projection = nn.Linear(self.fft_feature_dim, n_qubits)
        self.hidden_projection = nn.Linear(hidden_size * 2, n_qubits)

        self.input_gate = TanhGate(n_qubits, hidden_size, hidden_dim, vqc_depth)
        self.forget_gate = TanhGate(n_qubits, hidden_size, hidden_dim, vqc_depth)
        self.cell_gate = TanhGate(n_qubits, hidden_size, hidden_dim, vqc_depth)
        self.output_gate = TanhGate(n_qubits, hidden_size, hidden_dim, vqc_depth)

        self.output_projection = nn.Linear(hidden_size, input_size)
        self._print_config()

    def _print_config(self):
        print(f"\n{'='*60}")
        print("Tanh-LSTM Cell Initialized")
        print(f"{'='*60}")
        print(f"Input size: {self.input_size}, Hidden size: {self.hidden_size}")
        print(f"MLP input dim (n_qubits equiv): {self.n_qubits}")
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
# TANH-LSTM MODEL
# =============================================================================

class Tanh_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, vqc_depth,
                 output_size=1, window_size=8, n_frequencies=None, hidden_dim=64):
        super().__init__()
        self.window_size = window_size
        self.cell = TanhLSTMCell(input_size, hidden_size, n_qubits, vqc_depth,
                                 n_frequencies, window_size, hidden_dim)
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
    p = argparse.ArgumentParser(description="Tanh-LSTM for NARMA Prediction")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--vqc-depth", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=4)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--mlp-hidden-dim", type=int, default=64)
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
    print("Tanh-LSTM Training (Classical Baseline — No Periodic Structure)")
    print(f"{'='*60}")
    print(f"Hidden: {args.hidden_size}, MLP input: {args.n_qubits}, Depth: {args.vqc_depth}")
    print(f"Window: {args.window_size}, NARMA order: {args.narma_order}")
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

    model = Tanh_LSTM(1, args.hidden_size, args.n_qubits, args.vqc_depth,
                      1, args.window_size, hidden_dim=args.mlp_hidden_dim).float()
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

    print(f"\n{'='*60}")
    print(f"Training Complete! Train={tl:.6f}, Test={te:.6f}, Params={n_params}")
    print(f"{'='*60}")
    return model

if __name__ == "__main__":
    main()
