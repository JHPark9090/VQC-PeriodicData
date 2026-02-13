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

try:
    from dataset_dispatcher import add_dataset_args, load_dataset
except ImportError:
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from dataset_dispatcher import add_dataset_args, load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


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
    add_dataset_args(parser)
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

def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0, task='classification'):
    model.train(); loss_sum = 0; all_l, all_o = [], []
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device).float()
        optimizer.zero_grad(); out = model(inputs)
        loss = criterion(out, labels); loss.backward()
        if clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step(); loss_sum += loss.item()
        all_l.append(labels.cpu().numpy()); all_o.append(out.detach().cpu().numpy())
    all_l, all_o = np.concatenate(all_l), np.concatenate(all_o)
    if task == 'classification':
        try: metric = roc_auc_score(all_l, all_o)
        except: metric = 0.5
    else:
        metric = np.sqrt(np.mean((all_l - all_o) ** 2))
    return loss_sum / len(dataloader), metric

def evaluate(model, dataloader, criterion, device, task='classification'):
    model.eval(); loss_sum = 0; all_l, all_o = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device).float()
            out = model(inputs); loss_sum += criterion(out, labels).item()
            all_l.append(labels.cpu().numpy()); all_o.append(out.cpu().numpy())
    all_l, all_o = np.concatenate(all_l), np.concatenate(all_o)
    if task == 'classification':
        try: metric = roc_auc_score(all_l, all_o)
        except: metric = 0.5
    else:
        metric = np.sqrt(np.mean((all_l - all_o) ** 2))
    return loss_sum / len(dataloader), metric

def run_training(seed, mlp_dim, n_mlp_layers, input_dim, train_loader, val_loader,
                 test_loader, kernel_size=12, dilation=3, n_frequencies=None,
                 mlp_hidden_dim=64, num_epochs=50, lr=0.001,
                 checkpoint_dir=None, resume=False, args=None,
                 task='classification'):
    set_all_seeds(seed)
    model = ReLU_TCN(mlp_dim, n_mlp_layers, input_dim, kernel_size, dilation,
                     n_frequencies, mlp_hidden_dim).to(device)

    # Task-specific criterion and scheduler
    if task == 'classification':
        criterion = nn.BCEWithLogitsLoss()
        sched_mode, metric_name = 'max', 'auc'
        best_metric, is_better = 0.0, lambda new, old: new > old
    else:
        criterion = nn.MSELoss()
        sched_mode, metric_name = 'min', 'rmse'
        best_metric, is_better = float('inf'), lambda new, old: new < old

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=sched_mode, factor=0.5, patience=5, verbose=True)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(RESULTS_DIR, 'ReLU_TCN_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    cp_path = os.path.join(checkpoint_dir, f"relu_tcn_d{mlp_dim}_l{n_mlp_layers}_seed{seed}.pth")
    start_epoch, train_m, valid_m, best_state = 0, [], [], None

    if resume and os.path.exists(cp_path):
        cp = torch.load(cp_path, map_location=device)
        model.load_state_dict(cp['model_state_dict']); optimizer.load_state_dict(cp['optimizer_state_dict'])
        start_epoch = cp['epoch']+1; train_m = cp.get('train_metrics',[]); valid_m = cp.get('valid_metrics',[])
        best_metric = cp.get('best_val_metric', best_metric)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        tl, tm_val = train_epoch(model, train_loader, optimizer, criterion, device, task=task)
        vl, vm_val = evaluate(model, val_loader, criterion, device, task=task)
        scheduler.step(vm_val)
        train_m.append({'epoch':epoch+1,'train_loss':tl,f'train_{metric_name}':tm_val})
        valid_m.append({'epoch':epoch+1,'valid_loss':vl,f'valid_{metric_name}':vm_val})
        if is_better(vm_val, best_metric): best_metric = vm_val; best_state = copy.deepcopy(model.state_dict())
        em, es = epoch_time(t0, time.time())
        print(f"\nEpoch {epoch+1:02}/{num_epochs} | {em}m {es}s | Train: {tl:.4f}/{tm_val:.4f} | Val: {vl:.4f}/{vm_val:.4f} (Best {metric_name}: {best_metric:.4f})")
        torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                     'train_metrics':train_m,'valid_metrics':valid_m,'best_val_metric':best_metric}, cp_path)

    if best_state: model.load_state_dict(best_state)
    test_loss, test_metric = evaluate(model, test_loader, criterion, device, task=task)
    print(f"\n{'='*60}\nTest Loss: {test_loss:.4f}, Test {metric_name.upper()}: {test_metric:.4f}\nParams: {count_parameters(model)}\n{'='*60}")

    metrics = [{'epoch':i+1,'train_loss':train_m[i]['train_loss'],f'train_{metric_name}':train_m[i][f'train_{metric_name}'],
                'valid_loss':valid_m[i]['valid_loss'],f'valid_{metric_name}':valid_m[i][f'valid_{metric_name}'],
                'test_loss':test_loss,f'test_{metric_name}':test_metric} for i in range(len(train_m))]
    metrics_dir = os.path.join(RESULTS_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    pd.DataFrame(metrics).to_csv(os.path.join(metrics_dir, f"ReLU_TCN_d{mlp_dim}_l{n_mlp_layers}_seed{seed}_metrics.csv"), index=False)
    return test_loss, test_metric, model


if __name__ == "__main__":
    args = get_args()
    print(f"\n{'='*60}\nReLU-TCN: Classical Baseline (No Periodic Structure)\n{'='*60}\n")

    # Load dataset via dispatcher
    train_loader, val_loader, test_loader, input_dim, task, scaler = load_dataset(args, device)
    print(f"Dataset: {args.dataset}, Task: {task}, Input dim: {input_dim}")

    kernel_size = args.kernel_size; dilation = args.dilation
    test_loss, test_metric, model = run_training(
        args.seed, args.mlp_dim, args.n_mlp_layers, input_dim, train_loader, val_loader,
        test_loader, kernel_size, dilation, args.n_frequencies, args.mlp_hidden_dim,
        args.num_epochs, args.lr, resume=args.resume, args=args, task=task)
    metric_name = 'AUC' if task == 'classification' else 'RMSE'
    print(f"Final Test {metric_name}: {test_metric:.4f}")
