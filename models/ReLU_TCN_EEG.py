"""
ReLU-TCN: Classical Baseline using ReLU Activation for EEG Classification

ReLU has NO periodic structure — piecewise linear approximation fails for
periodic extrapolation. This is the weakest classical baseline.

Mirrors FourierQTCN_EEG.py exactly:
- Same sliding window extraction
- Same FFT frequency extraction
- Same linear projection
- ReLU MLP replaces quantum circuit
- Same learnable weighted aggregation

Author: ReLU classical baseline for VQC comparison
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
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from Load_PhysioNet_EEG import load_eeg_ts_revised
    except ImportError:
        print("Warning: Load_PhysioNet_EEG not found.")


def get_args():
    parser = argparse.ArgumentParser(description="ReLU-TCN for EEG Classification")
    parser.add_argument("--freq", type=int, default=80)
    parser.add_argument("--n-sample", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--mlp-dim", type=int, default=8, help="MLP input dim (equiv to n-qubits)")
    parser.add_argument("--n-mlp-layers", type=int, default=2, help="MLP depth (equiv to circuit-depth)")
    parser.add_argument("--n-frequencies", type=int, default=None)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--kernel-size", type=int, default=12)
    parser.add_argument("--dilation", type=int, default=3)
    return parser.parse_args()


print('Pytorch Version:', torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# ReLU BLOCK (replaces quantum circuit)
# =============================================================================

class ReLUBlock(nn.Module):
    """MLP with ReLU activation replacing quantum circuit. Output unbounded for BCEWithLogitsLoss."""

    def __init__(self, input_dim, hidden_dim=64, n_layers=2):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(n_layers):
            out_d = max(hidden_dim // (2 ** i), 8)
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
            in_d = out_d
        self.mlp = nn.Sequential(*layers)
        self.output_linear = nn.Linear(in_d, 1)

    def forward(self, x):
        return self.output_linear(self.mlp(x)).squeeze(-1)


# =============================================================================
# ReLU-TCN MODEL (mirrors FourierQTCN)
# =============================================================================

class ReLU_TCN(nn.Module):
    def __init__(self, mlp_dim, n_mlp_layers, input_dim, kernel_size,
                 dilation=1, n_frequencies=None, mlp_hidden_dim=64,
                 use_magnitude_phase=True):
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
        self.mlp_block = ReLUBlock(mlp_dim, mlp_hidden_dim, n_mlp_layers)

        max_windows = self.time_steps - self.dilation * (self.kernel_size - 1)
        self.agg_weights = nn.Parameter(torch.zeros(max_windows))
        self._print_config()

    def _print_config(self):
        print(f"\n{'='*60}")
        print("ReLU-TCN Initialized (Classical Baseline — No Periodic Structure)")
        print(f"{'='*60}")
        print(f"MLP dim (n_qubits equiv): {self.mlp_dim}")
        print(f"Kernel size: {self.kernel_size}, Dilation: {self.dilation}")
        print(f"FFT frequencies: {self.n_frequencies}, FFT feature dim: {self.fft_feature_dim}")
        print(f"Total parameters: {count_parameters(self)}")
        print(f"{'='*60}\n")

    def _extract_fft_features(self, window):
        batch_size = window.shape[0]
        fft_result = torch.fft.rfft(window, dim=-1)
        fft_selected = fft_result[:, :, :self.n_frequencies]
        if self.use_magnitude_phase:
            magnitude = torch.log1p(torch.abs(fft_selected))
            phase = torch.angle(fft_selected) / np.pi
            features = torch.stack([magnitude, phase], dim=-1)
        else:
            features = torch.stack([fft_selected.real, fft_selected.imag], dim=-1)
        return features.reshape(batch_size, -1)

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        effective_start = self.dilation * (self.kernel_size - 1)
        for t in range(effective_start, self.time_steps):
            indices = [t - d * self.dilation for d in range(self.kernel_size)]
            indices.reverse()
            window = x[:, :, indices]
            fft_features = self._extract_fft_features(window)
            projected = torch.tanh(self.projection(fft_features)) * np.pi
            outputs.append(self.mlp_block(projected))
        outputs = torch.stack(outputs, dim=1)
        n_windows = outputs.shape[1]
        weights = F.softmax(self.agg_weights[:n_windows], dim=0)
        return (outputs * weights.unsqueeze(0)).sum(dim=1)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def epoch_time(s, e):
    t = e - s; return int(t/60), int(t%60)

def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    model.train(); loss_sum = 0; all_l, all_o = [], []
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad(); out = model(inputs)
        loss = criterion(out, labels); loss.backward()
        if clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step(); loss_sum += loss.item()
        all_l.append(labels.cpu().numpy()); all_o.append(out.detach().cpu().numpy())
    all_l, all_o = np.concatenate(all_l), np.concatenate(all_o)
    try: auc = roc_auc_score(all_l, all_o)
    except: auc = 0.5
    return loss_sum / len(dataloader), auc

def evaluate(model, dataloader, criterion, device):
    model.eval(); loss_sum = 0; all_l, all_o = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            out = model(inputs); loss_sum += criterion(out, labels).item()
            all_l.append(labels.cpu().numpy()); all_o.append(out.cpu().numpy())
    all_l, all_o = np.concatenate(all_l), np.concatenate(all_o)
    try: auc = roc_auc_score(all_l, all_o)
    except: auc = 0.5
    return loss_sum / len(dataloader), auc

def run_training(seed, mlp_dim, n_mlp_layers, input_dim, train_loader, val_loader,
                 test_loader, kernel_size=12, dilation=3, n_frequencies=None,
                 mlp_hidden_dim=64, num_epochs=50, lr=0.001,
                 checkpoint_dir="ReLU_TCN_checkpoints", resume=False, args=None):
    set_all_seeds(seed)
    model = ReLU_TCN(mlp_dim, n_mlp_layers, input_dim, kernel_size, dilation,
                     n_frequencies, mlp_hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    os.makedirs(checkpoint_dir, exist_ok=True)
    cp_path = os.path.join(checkpoint_dir, f"relu_tcn_d{mlp_dim}_l{n_mlp_layers}_seed{seed}.pth")
    start_epoch, train_m, valid_m, best_auc, best_state = 0, [], [], 0.0, None

    if resume and os.path.exists(cp_path):
        cp = torch.load(cp_path, map_location=device)
        model.load_state_dict(cp['model_state_dict']); optimizer.load_state_dict(cp['optimizer_state_dict'])
        start_epoch = cp['epoch']+1; train_m = cp.get('train_metrics',[]); valid_m = cp.get('valid_metrics',[])
        best_auc = cp.get('best_val_auc', 0.0)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step(va)
        train_m.append({'epoch':epoch+1,'train_loss':tl,'train_auc':ta})
        valid_m.append({'epoch':epoch+1,'valid_loss':vl,'valid_auc':va})
        if va > best_auc: best_auc = va; best_state = copy.deepcopy(model.state_dict())
        em, es = epoch_time(t0, time.time())
        print(f"\nEpoch {epoch+1:02}/{num_epochs} | {em}m {es}s | Train: {tl:.4f}/{ta:.4f} | Val: {vl:.4f}/{va:.4f} (Best: {best_auc:.4f})")
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                     'train_metrics':train_m,'valid_metrics':valid_m,'best_val_auc':best_auc}, cp_path)

    if best_state: model.load_state_dict(best_state)
    test_loss, test_auc = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*60}\nTest Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}\nParams: {count_parameters(model)}\n{'='*60}")

    metrics = [{'epoch':i+1,'train_loss':train_m[i]['train_loss'],'train_auc':train_m[i]['train_auc'],
                'valid_loss':valid_m[i]['valid_loss'],'valid_auc':valid_m[i]['valid_auc'],
                'test_loss':test_loss,'test_auc':test_auc} for i in range(len(train_m))]
    pd.DataFrame(metrics).to_csv(f"ReLU_TCN_d{mlp_dim}_l{n_mlp_layers}_seed{seed}_metrics.csv", index=False)
    return test_loss, test_auc, model


if __name__ == "__main__":
    args = get_args()
    print(f"\n{'='*60}\nReLU-TCN: Classical Baseline (No Periodic Structure)\n{'='*60}\n")
    train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
        seed=args.seed, device=device, batch_size=32, sampling_freq=args.freq, sample_size=args.n_sample)
    kernel_size = args.kernel_size; dilation = args.dilation
    test_loss, test_auc, model = run_training(
        args.seed, args.mlp_dim, args.n_mlp_layers, input_dim, train_loader, val_loader,
        test_loader, kernel_size, dilation, args.n_frequencies, args.mlp_hidden_dim,
        args.num_epochs, args.lr, resume=args.resume, args=args)
    print(f"Final Test AUC: {test_auc:.4f}")
