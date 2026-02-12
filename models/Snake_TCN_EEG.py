"""
Snake-TCN: Classical Baseline using Snake Activation for EEG Classification

Snake activation: Snake_a(x) = x + (1/a) * sin²(a*x)
This is a PARTIAL periodic activation — it has a linear term (x) plus
a periodic term (sin²), enabling both trend modeling and periodic learning.

Snake is stronger than ReLU/tanh for periodic data but weaker than SIREN
because SIREN computes pure Fourier series while Snake adds periodicity
on top of a linear backbone.

Mirrors FourierQTCN_EEG.py exactly:
- Same sliding window extraction
- Same FFT frequency extraction (_extract_fft_features)
- Same linear projection
- SnakeBlock replaces quantum circuit
- Same learnable weighted aggregation

Based on:
- Ziyin et al. (2020): Neural networks fail to learn periodic functions and how to fix it
- Schuld et al. (2021): VQC as Fourier series

Author: Snake classical baseline for VQC comparison
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
import os
import copy
import time
import random
import argparse

try:
    from Load_PhysioNet_EEG import load_eeg_ts_revised
except ImportError:
    try:
        import sys; sys.path.insert(0, os.path.dirname(__file__))
        from Load_PhysioNet_EEG import load_eeg_ts_revised
    except ImportError:
        print("Warning: Load_PhysioNet_EEG not found.")


def get_args():
    parser = argparse.ArgumentParser(description="Snake-TCN for EEG Classification")
    parser.add_argument("--freq", type=int, default=80)
    parser.add_argument("--n-sample", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--mlp-dim", type=int, default=8, help="MLP input dim (equiv to n-qubits)")
    parser.add_argument("--n-mlp-layers", type=int, default=2, help="MLP depth (equiv to circuit-depth)")
    parser.add_argument("--n-frequencies", type=int, default=None)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--a-init", type=float, default=1.0, help="Initial Snake frequency parameter")
    parser.add_argument("--learnable-a", action="store_true", default=True)
    parser.add_argument("--no-learnable-a", action="store_false", dest="learnable_a")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--kernel-size", type=int, default=12)
    parser.add_argument("--dilation", type=int, default=3)
    return parser.parse_args()


print('Pytorch Version:', torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False; os.environ["PL_GLOBAL_SEED"] = str(seed)

def count_parameters(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


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
        a = self.a.abs() + 1e-8
        return x + (1.0 / a) * torch.sin(a * x) ** 2

    def get_a_values(self) -> np.ndarray:
        return self.a.detach().cpu().numpy()


# =============================================================================
# SNAKE BLOCK (replaces quantum circuit)
# =============================================================================

class SnakeBlock(nn.Module):
    """MLP with Snake activation replacing quantum circuit. Output unbounded for BCEWithLogitsLoss."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2, a_init=1.0, learnable_a=True):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(n_layers):
            out_d = max(hidden_dim // (2**i), 8)
            layers.append(nn.Linear(in_d, out_d))
            layers.append(SnakeActivation(out_d, a_init, learnable_a))
            in_d = out_d
        self.mlp = nn.Sequential(*layers)
        self.output_linear = nn.Linear(in_d, 1)

    def forward(self, x):
        return self.output_linear(self.mlp(x)).squeeze(-1)

    def get_a_values(self) -> List[np.ndarray]:
        return [m.get_a_values() for m in self.mlp if isinstance(m, SnakeActivation)]


# =============================================================================
# SNAKE-TCN MODEL
# =============================================================================

class Snake_TCN(nn.Module):
    def __init__(self, mlp_dim, n_mlp_layers, input_dim, kernel_size,
                 dilation=1, n_frequencies=None, mlp_hidden_dim=64,
                 a_init=1.0, learnable_a=True, use_magnitude_phase=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_magnitude_phase = use_magnitude_phase

        max_fft_bins = kernel_size // 2 + 1
        self.n_frequencies = min(n_frequencies or mlp_dim, max_fft_bins)
        self.fft_feature_dim = self.input_channels * self.n_frequencies * 2

        self.projection = nn.Linear(self.fft_feature_dim, mlp_dim)
        self.mlp_block = SnakeBlock(mlp_dim, mlp_hidden_dim, n_mlp_layers, a_init, learnable_a)

        max_windows = self.time_steps - self.dilation * (self.kernel_size - 1)
        self.agg_weights = nn.Parameter(torch.zeros(max_windows))
        self._print_config()

    def _print_config(self):
        print(f"\n{'='*60}")
        print("Snake-TCN Initialized (Partial Periodic Baseline)")
        print(f"{'='*60}")
        print(f"MLP dim (n_qubits equiv): {self.mlp_dim}")
        print(f"Kernel size: {self.kernel_size}, Dilation: {self.dilation}")
        print(f"FFT frequencies: {self.n_frequencies}, FFT dim: {self.fft_feature_dim}")
        print(f"Parameters: {count_parameters(self)}")
        print(f"{'='*60}\n")

    def _extract_fft_features(self, window):
        batch_size = window.shape[0]
        fft_result = torch.fft.rfft(window, dim=-1)
        fft_selected = fft_result[:, :, :self.n_frequencies]
        if self.use_magnitude_phase:
            mag = torch.log1p(torch.abs(fft_selected))
            ph = torch.angle(fft_selected) / np.pi
            features = torch.stack([mag, ph], dim=-1)
        else:
            features = torch.stack([fft_selected.real, fft_selected.imag], dim=-1)
        return features.reshape(batch_size, -1)

    def forward(self, x):
        outputs = []
        effective_start = self.dilation * (self.kernel_size - 1)
        for t in range(effective_start, self.time_steps):
            indices = sorted([t - d*self.dilation for d in range(self.kernel_size)])
            window = x[:, :, indices]
            fft_features = self._extract_fft_features(window)
            projected = torch.tanh(self.projection(fft_features)) * np.pi
            outputs.append(self.mlp_block(projected))
        outputs = torch.stack(outputs, dim=1)
        weights = F.softmax(self.agg_weights[:outputs.shape[1]], dim=0)
        return (outputs * weights.unsqueeze(0)).sum(dim=1)

    def get_a_values(self) -> List[np.ndarray]:
        return self.mlp_block.get_a_values()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def epoch_time(s, e): t = e-s; return int(t/60), int(t%60)

def train_epoch(model, dl, opt, crit, dev, clip=1.0):
    model.train(); ls = 0; al, ao = [], []
    for inp, lab in tqdm(dl, desc="Training"):
        inp, lab = inp.to(dev), lab.to(dev).float()
        opt.zero_grad(); out = model(inp); loss = crit(out, lab)
        loss.backward()
        if clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step(); ls += loss.item()
        al.append(lab.cpu().numpy()); ao.append(out.detach().cpu().numpy())
    al, ao = np.concatenate(al), np.concatenate(ao)
    try: auc = roc_auc_score(al, ao)
    except: auc = 0.5
    return ls/len(dl), auc

def evaluate(model, dl, crit, dev):
    model.eval(); ls = 0; al, ao = [], []
    with torch.no_grad():
        for inp, lab in tqdm(dl, desc="Evaluating"):
            inp, lab = inp.to(dev), lab.to(dev).float()
            out = model(inp); ls += crit(out, lab).item()
            al.append(lab.cpu().numpy()); ao.append(out.cpu().numpy())
    al, ao = np.concatenate(al), np.concatenate(ao)
    try: auc = roc_auc_score(al, ao)
    except: auc = 0.5
    return ls/len(dl), auc

def run_training(seed, mlp_dim, n_mlp_layers, input_dim, train_loader, val_loader,
                 test_loader, kernel_size=12, dilation=3, n_frequencies=None,
                 mlp_hidden_dim=64, a_init=1.0, learnable_a=True,
                 num_epochs=50, lr=0.001,
                 checkpoint_dir="Snake_TCN_checkpoints", resume=False, args=None):
    set_all_seeds(seed)
    model = Snake_TCN(mlp_dim, n_mlp_layers, input_dim, kernel_size, dilation,
                      n_frequencies, mlp_hidden_dim, a_init, learnable_a).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, verbose=True)

    os.makedirs(checkpoint_dir, exist_ok=True)
    cp = os.path.join(checkpoint_dir, f"snake_tcn_d{mlp_dim}_l{n_mlp_layers}_seed{seed}.pth")
    se, tm, vm, ba, bs = 0, [], [], 0.0, None
    if resume and os.path.exists(cp):
        c = torch.load(cp, map_location=device)
        model.load_state_dict(c['model_state_dict']); opt.load_state_dict(c['optimizer_state_dict'])
        se=c['epoch']+1; tm=c.get('train_metrics',[]); vm=c.get('valid_metrics',[]); ba=c.get('best_val_auc',0.0)

    for epoch in range(se, num_epochs):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, opt, crit, device)
        vl, va = evaluate(model, val_loader, crit, device)
        sched.step(va)
        tm.append({'epoch':epoch+1,'train_loss':tl,'train_auc':ta})
        vm.append({'epoch':epoch+1,'valid_loss':vl,'valid_auc':va})
        if va > ba: ba = va; bs = copy.deepcopy(model.state_dict())
        em, es = epoch_time(t0, time.time())
        print(f"\nEpoch {epoch+1:02}/{num_epochs} | {em}m {es}s | Train: {tl:.4f}/{ta:.4f} | Val: {vl:.4f}/{va:.4f} (Best: {ba:.4f})")
        if (epoch+1) % 10 == 0:
            a_vals = model.get_a_values()
            for i, a in enumerate(a_vals):
                print(f"  Snake Layer {i} 'a': mean={a.mean():.3f}, std={a.std():.3f}")
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),
                     'train_metrics':tm,'valid_metrics':vm,'best_val_auc':ba}, cp)

    if bs: model.load_state_dict(bs)
    test_loss, test_auc = evaluate(model, test_loader, crit, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Params: {count_parameters(model)}")

    # Print final learned 'a' values
    print("\nLearned Snake 'a' values:")
    a_vals = model.get_a_values()
    for i, a in enumerate(a_vals):
        print(f"  Layer {i}: mean={a.mean():.3f}, std={a.std():.3f}, range=[{a.min():.3f}, {a.max():.3f}]")

    metrics = [{'epoch':i+1,'train_loss':tm[i]['train_loss'],'train_auc':tm[i]['train_auc'],
                'valid_loss':vm[i]['valid_loss'],'valid_auc':vm[i]['valid_auc'],
                'test_loss':test_loss,'test_auc':test_auc} for i in range(len(tm))]
    pd.DataFrame(metrics).to_csv(f"Snake_TCN_d{mlp_dim}_l{n_mlp_layers}_seed{seed}_metrics.csv", index=False)
    return test_loss, test_auc, model

if __name__ == "__main__":
    args = get_args()
    print(f"\n{'='*60}\nSnake-TCN: Partial Periodic Baseline\n{'='*60}\n")
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=args.seed, device=device, batch_size=32, sampling_freq=args.freq, sample_size=args.n_sample)
    run_training(args.seed, args.mlp_dim, args.n_mlp_layers, input_dim, train_loader, val_loader,
                 test_loader, args.kernel_size, args.dilation, args.n_frequencies, args.mlp_hidden_dim,
                 args.a_init, args.learnable_a, args.num_epochs, args.lr, resume=args.resume, args=args)
