"""
Mackey-Glass Dataset Generator

Generates time-series from the Mackey-Glass delay differential equation:
    dx/dt = β · x(t-τ) / (1 + x(t-τ)^n) - γ · x(t)

This is a standard benchmark in quantum reservoir computing (Fujii & Nakajima,
2017). The Mackey-Glass system exhibits quasi-periodic behavior for τ=17 and
fully chaotic behavior for τ≥30. It tests whether periodic structure in a
model helps with deterministic chaos that has underlying periodic components.

Standard parameters:
    τ=17 (quasi-periodic), β=0.2, γ=0.1, n=10
    Integration: Euler method, dt=1.0
    Warmup: 500 steps discarded (transient removal)

This module provides:
    - generate_mackey_glass_series(): Generate raw Mackey-Glass time-series
    - get_mackey_glass_data(): Simple interface for QLSTM (mirrors get_narma_data)
    - get_mackey_glass_dataloaders(): Full DataLoader interface for training
    - analyze_mackey_glass_spectrum(): Frequency analysis

Author: VQC-PeriodicData benchmark suite
Date: February 2026
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional

# Reuse sliding-window logic from narma_generator
try:
    from .narma_generator import create_narma_sequences
except ImportError:
    from narma_generator import create_narma_sequences


# =============================================================================
# MACKEY-GLASS SERIES GENERATION
# =============================================================================

def generate_mackey_glass_series(
    n_samples: int = 500,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_power: int = 10,
    dt: float = 1.0,
    warmup: int = 500,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Mackey-Glass time-series via Euler integration.

    dx/dt = β · x(t-τ) / (1 + x(t-τ)^n) - γ · x(t)

    Args:
        n_samples: Number of usable time steps after warmup
        tau: Delay parameter (17=quasi-periodic, 30=chaotic)
        beta: Production rate (default: 0.2)
        gamma: Decay rate (default: 0.1)
        n_power: Nonlinearity exponent (default: 10)
        dt: Integration time step (default: 1.0)
        warmup: Number of warmup steps to discard (default: 500)
        seed: Random seed for initial conditions

    Returns:
        t: Time indices [n_samples]
        y: Output series [n_samples]
    """
    if seed is not None:
        np.random.seed(seed)

    total_steps = n_samples + warmup + tau

    # Initialize with small random perturbation around 1.2
    x = np.zeros(total_steps)
    x[:tau + 1] = 1.2 + 0.1 * np.random.randn(tau + 1)

    # Euler integration
    for t_idx in range(tau, total_steps - 1):
        x_delayed = x[t_idx - tau]
        dxdt = beta * x_delayed / (1.0 + x_delayed ** n_power) - gamma * x[t_idx]
        x[t_idx + 1] = x[t_idx] + dt * dxdt

    # Discard warmup transient
    y = x[warmup + tau:][:n_samples]
    t = np.arange(n_samples, dtype=np.float64)

    return t, y


# =============================================================================
# SIMPLE INTERFACE FOR QLSTM (mirrors get_narma_data)
# =============================================================================

def get_mackey_glass_data(
    tau: int = 17,
    seq_len: int = 8,
    n_samples: int = 500,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple interface for getting Mackey-Glass data (compatible with QLSTM).

    Args:
        tau: Delay parameter (17=quasi-periodic, 30=chaotic)
        seq_len: Sequence length for input windows
        n_samples: Total number of usable time steps
        seed: Random seed

    Returns:
        x: Input sequences [n_sequences, seq_len]
        y: Target values [n_sequences]
    """
    # Generate Mackey-Glass series
    _, y_series = generate_mackey_glass_series(n_samples, tau=tau, seed=seed)

    # Normalize to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_normalized = scaler.fit_transform(y_series.reshape(-1, 1)).flatten()

    # Create sequences using shared sliding-window function
    x, y = create_narma_sequences(y_normalized, seq_len, output_format='lstm')

    # Convert to tensors
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    return x_tensor, y_tensor


# =============================================================================
# FULL DATALOADER INTERFACE
# =============================================================================

def get_mackey_glass_dataloaders(
    n_samples: int = 2000,
    tau: int = 17,
    seq_len: int = 20,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    output_format: str = 'tcn',
    normalize: str = 'minmax',
    normalize_range: Tuple[float, float] = (-1, 1),
    seed: Optional[int] = None,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, tuple, object, TensorDataset, int, int]:
    """
    Generate Mackey-Glass data and create DataLoaders for training.

    Args:
        n_samples: Total number of usable time steps
        tau: Delay parameter (17=quasi-periodic, 30=chaotic)
        seq_len: Length of input sequences
        batch_size: Batch size for DataLoaders
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set (rest is test)
        output_format: 'lstm', 'tcn', or 'raw'
        normalize: 'minmax', 'standard', or 'none'
        normalize_range: Range for minmax normalization
        seed: Random seed
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader, val_loader, test_loader, input_dim, scaler,
        full_dataset, train_size, val_size
    """
    print(f"Generating Mackey-Glass (τ={tau}) data...")

    # Generate series
    _, y_series = generate_mackey_glass_series(n_samples, tau=tau, seed=seed)

    # Normalize
    if normalize == 'minmax':
        scaler = MinMaxScaler(feature_range=normalize_range)
        y_normalized = scaler.fit_transform(y_series.reshape(-1, 1))
    elif normalize == 'standard':
        scaler = StandardScaler()
        y_normalized = scaler.fit_transform(y_series.reshape(-1, 1))
    else:
        scaler = None
        y_normalized = y_series.reshape(-1, 1)

    # Create sequences
    x, y = create_narma_sequences(y_normalized, seq_len, output_format=output_format)

    # Convert to tensors
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    # Create dataset
    full_dataset = TensorDataset(x_tensor, y_tensor)

    # Sequential split (important for time-series!)
    n_total = len(full_dataset)
    train_end = int(train_ratio * n_total)
    val_end = int((train_ratio + val_ratio) * n_total)

    train_dataset = Subset(full_dataset, list(range(train_end)))
    val_dataset = Subset(full_dataset, list(range(train_end, val_end)))
    test_dataset = Subset(full_dataset, list(range(val_end, n_total)))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle_train, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=True)

    # Determine input dimensions
    if output_format == 'tcn':
        input_dim = (batch_size, x_tensor.shape[1], x_tensor.shape[2])
    elif output_format == 'lstm':
        input_dim = (batch_size, x_tensor.shape[1])
    else:
        input_dim = x_tensor.shape

    print(f"Data generated successfully!")
    print(f"  Total sequences: {n_total}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  Input shape: {x_tensor.shape}")
    print(f"  Sequence length: {seq_len}")

    return (
        train_loader, val_loader, test_loader,
        input_dim, scaler, full_dataset,
        len(train_dataset), len(val_dataset)
    )


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def analyze_mackey_glass_spectrum(
    tau: int = 17,
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> dict:
    """
    Analyze the frequency spectrum of Mackey-Glass series.

    Args:
        tau: Delay parameter
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Dictionary with spectrum analysis
    """
    _, y = generate_mackey_glass_series(n_samples, tau=tau, seed=seed)

    # Compute FFT
    fft = np.fft.rfft(y)
    frequencies = np.fft.rfftfreq(len(y))
    magnitude = np.abs(fft)
    phase = np.angle(fft)

    # Find dominant frequencies
    sorted_indices = np.argsort(magnitude)[::-1]
    top_freqs = frequencies[sorted_indices[:5]]
    top_mags = magnitude[sorted_indices[:5]]

    return {
        'frequencies': frequencies,
        'magnitude': magnitude,
        'phase': phase,
        'dominant_frequencies': top_freqs,
        'dominant_magnitudes': top_mags,
        'mean': np.mean(y),
        'std': np.std(y),
        'min': np.min(y),
        'max': np.max(y),
        'tau': tau,
    }


# =============================================================================
# MAIN (DEMONSTRATION)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mackey-Glass Dataset Generator")
    print("=" * 60)

    # Basic generation
    print("\n1. Basic Mackey-Glass (τ=17) generation:")
    t, y = generate_mackey_glass_series(n_samples=500, tau=17, seed=42)
    print(f"   Time shape: {t.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Output range: [{y.min():.4f}, {y.max():.4f}]")

    # Simple interface
    print("\n2. Simple interface (for QLSTM):")
    x, y = get_mackey_glass_data(tau=17, seq_len=8, n_samples=500, seed=42)
    print(f"   x shape: {x.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"   y range: [{y.min():.4f}, {y.max():.4f}]")

    # Chaotic regime
    print("\n3. Chaotic regime (τ=30):")
    x30, y30 = get_mackey_glass_data(tau=30, seq_len=8, n_samples=500, seed=42)
    print(f"   x shape: {x30.shape}")
    print(f"   y shape: {y30.shape}")

    # Spectrum analysis
    print("\n4. Spectrum analysis (τ=17):")
    analysis = analyze_mackey_glass_spectrum(tau=17, n_samples=1000, seed=42)
    print(f"   Dominant frequencies: {analysis['dominant_frequencies'][:3]}")
    print(f"   Mean: {analysis['mean']:.4f}, Std: {analysis['std']:.4f}")

    print("\n" + "=" * 60)
    print("Mackey-Glass Generator Ready!")
    print("=" * 60)
