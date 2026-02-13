"""
Multi-Sine Dataset Generator

Generates a superposition of K sinusoids with incommensurate frequencies:
    y(t) = Σ_{k=1}^{K} a_k · sin(2π · f_k · t + φ_k) + ε(t)

This is the purest test of VQC's Fourier series advantage, since the target
function IS a Fourier series. If VQC has a native periodic advantage, it
should excel here.

Default configuration (K=5):
    Frequencies: [0.1, 0.23, 0.37, 0.51, 0.79] (incommensurate)
    Amplitudes:  [1.0, 0.8, 0.6, 0.4, 0.2]     (decreasing)
    Phases:      [0, π/4, π/3, π/6, π/2]

This module provides:
    - generate_multisine_series(): Generate raw multi-sine time-series
    - get_multisine_data(): Simple interface for QLSTM (mirrors get_narma_data)
    - get_multisine_dataloaders(): Full DataLoader interface for training
    - analyze_multisine_spectrum(): Frequency analysis

Author: VQC-PeriodicData benchmark suite
Date: February 2026
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, List

# Reuse sliding-window logic from narma_generator
try:
    from .narma_generator import create_narma_sequences
except ImportError:
    from narma_generator import create_narma_sequences


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_FREQUENCIES = [0.1, 0.23, 0.37, 0.51, 0.79]
DEFAULT_AMPLITUDES = [1.0, 0.8, 0.6, 0.4, 0.2]
DEFAULT_PHASES = [0, np.pi / 4, np.pi / 3, np.pi / 6, np.pi / 2]


# =============================================================================
# MULTI-SINE SERIES GENERATION
# =============================================================================

def generate_multisine_series(
    n_samples: int = 500,
    K: int = 5,
    frequencies: Optional[List[float]] = None,
    amplitudes: Optional[List[float]] = None,
    phases: Optional[List[float]] = None,
    noise_std: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate multi-sine time-series data.

    y(t) = Σ_{k=1}^{K} a_k · sin(2π · f_k · t + φ_k) + ε(t)

    Args:
        n_samples: Number of time steps to generate
        K: Number of sinusoidal components
        frequencies: List of K frequencies (default: incommensurate set)
        amplitudes: List of K amplitudes (default: decreasing)
        phases: List of K phases in radians (default: varied)
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility

    Returns:
        t: Time indices [n_samples]
        y: Output series [n_samples]
    """
    if seed is not None:
        np.random.seed(seed)

    # Use defaults if not provided
    if frequencies is None:
        frequencies = DEFAULT_FREQUENCIES[:K]
    if amplitudes is None:
        amplitudes = DEFAULT_AMPLITUDES[:K]
    if phases is None:
        phases = DEFAULT_PHASES[:K]

    assert len(frequencies) == K, f"Expected {K} frequencies, got {len(frequencies)}"
    assert len(amplitudes) == K, f"Expected {K} amplitudes, got {len(amplitudes)}"
    assert len(phases) == K, f"Expected {K} phases, got {len(phases)}"

    t = np.arange(n_samples, dtype=np.float64)

    # Sum of sinusoids
    y = np.zeros(n_samples)
    for k in range(K):
        y += amplitudes[k] * np.sin(2 * np.pi * frequencies[k] * t + phases[k])

    # Add noise
    if noise_std > 0:
        y += np.random.normal(0, noise_std, n_samples)

    return t, y


# =============================================================================
# SIMPLE INTERFACE FOR QLSTM (mirrors get_narma_data)
# =============================================================================

def get_multisine_data(
    K: int = 5,
    seq_len: int = 8,
    n_samples: int = 500,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple interface for getting multi-sine data (compatible with QLSTM).

    Args:
        K: Number of sinusoidal components
        seq_len: Sequence length for input windows
        n_samples: Total number of time steps to generate
        seed: Random seed

    Returns:
        x: Input sequences [n_sequences, seq_len]
        y: Target values [n_sequences]
    """
    # Generate multi-sine series
    _, y_series = generate_multisine_series(n_samples, K=K, seed=seed)

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

def get_multisine_dataloaders(
    n_samples: int = 2000,
    K: int = 5,
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
    Generate multi-sine data and create DataLoaders for training.

    Args:
        n_samples: Total number of time steps to generate
        K: Number of sinusoidal components
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
    print(f"Generating Multi-Sine (K={K}) data...")

    # Generate series
    _, y_series = generate_multisine_series(n_samples, K=K, seed=seed)

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

def analyze_multisine_spectrum(
    K: int = 5,
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> dict:
    """
    Analyze the frequency spectrum of multi-sine series.

    Args:
        K: Number of sinusoidal components
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Dictionary with spectrum analysis
    """
    _, y = generate_multisine_series(n_samples, K=K, seed=seed)

    # Compute FFT
    fft = np.fft.rfft(y)
    frequencies = np.fft.rfftfreq(len(y))
    magnitude = np.abs(fft)
    phase = np.angle(fft)

    # Find dominant frequencies
    sorted_indices = np.argsort(magnitude)[::-1]
    top_freqs = frequencies[sorted_indices[:K + 2]]
    top_mags = magnitude[sorted_indices[:K + 2]]

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
        'K': K,
        'true_frequencies': DEFAULT_FREQUENCIES[:K],
        'true_amplitudes': DEFAULT_AMPLITUDES[:K],
    }


# =============================================================================
# MAIN (DEMONSTRATION)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Sine Dataset Generator")
    print("=" * 60)

    # Basic generation
    print("\n1. Basic Multi-Sine (K=5) generation:")
    t, y = generate_multisine_series(n_samples=500, K=5, seed=42)
    print(f"   Time shape: {t.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Output range: [{y.min():.4f}, {y.max():.4f}]")

    # Simple interface
    print("\n2. Simple interface (for QLSTM):")
    x, y = get_multisine_data(K=5, seq_len=8, n_samples=500, seed=42)
    print(f"   x shape: {x.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"   y range: [{y.min():.4f}, {y.max():.4f}]")

    # Spectrum analysis
    print("\n3. Spectrum analysis:")
    analysis = analyze_multisine_spectrum(K=5, n_samples=1000, seed=42)
    print(f"   True frequencies: {analysis['true_frequencies']}")
    print(f"   Dominant frequencies: {analysis['dominant_frequencies'][:5]}")
    print(f"   Mean: {analysis['mean']:.4f}, Std: {analysis['std']:.4f}")

    print("\n" + "=" * 60)
    print("Multi-Sine Generator Ready!")
    print("=" * 60)
