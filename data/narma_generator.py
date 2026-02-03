"""
NARMA (Nonlinear AutoRegressive Moving Average) Dataset Generator

NARMA is a standard benchmark for time-series prediction tasks, especially
for testing recurrent neural networks and reservoir computing systems.

NARMA-n is defined as:
    y(t) = α·y(t-1) + β·y(t-1)·Σ(y(t-1-i) for i in 0..n-1) + γ·u(t-n)·u(t-1) + δ

where:
    - u(t) is the input (uniform random in [0, 0.5])
    - y(t) is the output
    - n is the order (memory length)
    - α, β, γ, δ are coefficients (standard: 0.3, 0.05, 1.5, 0.1)

Common orders:
    - NARMA-5:  Short-term memory task
    - NARMA-10: Standard benchmark (most common)
    - NARMA-30: Long-term memory task (challenging)

This module provides:
    - generate_narma_series(): Generate raw NARMA time-series
    - create_narma_sequences(): Transform to input-output pairs
    - get_narma_data(): Simple interface for QLSTM
    - get_narma_dataloaders(): Full DataLoader interface for training

Author: Based on HQTCN2_NARMA.py
Date: February 2026
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, List, Union


# =============================================================================
# NARMA SERIES GENERATION
# =============================================================================

def generate_narma_series(
    n_samples: int,
    order: int = 10,
    alpha: float = 0.3,
    beta: float = 0.05,
    gamma: float = 1.5,
    delta: float = 0.1,
    input_range: Tuple[float, float] = (0.0, 0.5),
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NARMA time-series data.

    NARMA-n equation:
        y(t) = α·y(t-1) + β·y(t-1)·Σ(y(t-i-1)) + γ·u(t-n)·u(t-1) + δ

    Args:
        n_samples: Number of samples to generate
        order: NARMA order (memory length), typically 5, 10, or 30
        alpha: Coefficient for y(t-1) term (default: 0.3)
        beta: Coefficient for nonlinear term (default: 0.05)
        gamma: Coefficient for input product term (default: 1.5)
        delta: Constant offset (default: 0.1)
        input_range: Range for uniform input distribution (default: (0, 0.5))
        seed: Random seed for reproducibility

    Returns:
        u: Input sequence [n_samples]
        y: Output sequence [n_samples]
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate input sequence
    u = np.random.uniform(input_range[0], input_range[1], n_samples)

    # Initialize output sequence
    y = np.zeros(n_samples)

    # Generate NARMA sequence
    for t in range(order, n_samples):
        # Term 1: α·y(t-1)
        term1 = alpha * y[t - 1]

        # Term 2: β·y(t-1)·Σ(y(t-i-1) for i in 0..n-1)
        # This is the nonlinear memory term
        sum_y = sum(y[t - i - 1] for i in range(order))
        term2 = beta * y[t - 1] * sum_y

        # Term 3: γ·u(t-n)·u(t-1)
        # This couples past and recent inputs
        term3 = gamma * u[t - order] * u[t - 1]

        # Term 4: δ (constant offset)
        term4 = delta

        y[t] = term1 + term2 + term3 + term4

    return u, y


def generate_narma_variants(
    n_samples: int,
    orders: List[int] = [5, 10, 30],
    seed: Optional[int] = None
) -> dict:
    """
    Generate multiple NARMA variants with different orders.

    Args:
        n_samples: Number of samples per variant
        orders: List of NARMA orders to generate
        seed: Random seed

    Returns:
        Dictionary with 'narma_5', 'narma_10', etc. keys
    """
    variants = {}

    for order in orders:
        u, y = generate_narma_series(n_samples, order=order, seed=seed)
        variants[f'narma_{order}'] = {
            'input': u,
            'output': y,
            'order': order
        }

    return variants


# =============================================================================
# SEQUENCE TRANSFORMATION
# =============================================================================

def create_narma_sequences(
    data: np.ndarray,
    seq_len: int,
    output_format: str = 'lstm'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform NARMA series into input-output sequence pairs.

    Args:
        data: NARMA output series [n_samples] or [n_samples, 1]
        seq_len: Length of input sequences
        output_format:
            - 'lstm': [batch, seq_len] for QLSTM
            - 'tcn':  [batch, channels, seq_len] for QTCN (channels=1)
            - 'raw':  [batch, seq_len, 1] for general use

    Returns:
        x: Input sequences
        y: Target values
    """
    # Ensure data is 1D
    if data.ndim > 1:
        data = data.flatten()

    n_samples = len(data)
    x_list = []
    y_list = []

    for i in range(n_samples - seq_len):
        x_list.append(data[i:i + seq_len])
        y_list.append(data[i + seq_len])

    x = np.array(x_list)
    y = np.array(y_list)

    # Format output
    if output_format == 'lstm':
        # [batch, seq_len] - standard for LSTM
        pass
    elif output_format == 'tcn':
        # [batch, channels, seq_len] - for Conv1d in TCN
        x = x[:, np.newaxis, :]  # Add channel dimension
    elif output_format == 'raw':
        # [batch, seq_len, 1] - explicit feature dimension
        x = x[:, :, np.newaxis]
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

    return x, y


# =============================================================================
# SIMPLE INTERFACE FOR QLSTM
# =============================================================================

def get_narma_data(
    n_0: int = 10,
    seq_len: int = 8,
    n_samples: int = 500,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple interface for getting NARMA data (compatible with QLSTM_v0.py).

    Args:
        n_0: NARMA order (default: 10)
        seq_len: Sequence length for input windows
        n_samples: Total number of samples to generate
        seed: Random seed

    Returns:
        x: Input sequences [n_sequences, seq_len]
        y: Target values [n_sequences]
    """
    # Generate NARMA series
    _, y_series = generate_narma_series(n_samples, order=n_0, seed=seed)

    # Normalize to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_normalized = scaler.fit_transform(y_series.reshape(-1, 1)).flatten()

    # Create sequences
    x, y = create_narma_sequences(y_normalized, seq_len, output_format='lstm')

    # Convert to tensors
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()

    return x_tensor, y_tensor


# =============================================================================
# FULL DATALOADER INTERFACE
# =============================================================================

def get_narma_dataloaders(
    n_samples: int = 2000,
    order: int = 10,
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
    Generate NARMA data and create DataLoaders for training.

    Args:
        n_samples: Total number of NARMA samples to generate
        order: NARMA order (5, 10, or 30)
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
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        input_dim: Input dimensions tuple
        scaler: Fitted scaler for inverse transform
        full_dataset: Complete TensorDataset
        train_size: Number of training samples
        val_size: Number of validation samples
    """
    print(f"Generating NARMA-{order} data...")

    # Generate NARMA series
    _, y_series = generate_narma_series(n_samples, order=order, seed=seed)

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

    train_indices = list(range(train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, n_total))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

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
        train_loader,
        val_loader,
        test_loader,
        input_dim,
        scaler,
        full_dataset,
        len(train_dataset),
        len(val_dataset)
    )


# =============================================================================
# SPECIALIZED LOADERS FOR FOURIER MODELS
# =============================================================================

def get_narma_for_fourier_qlstm(
    n_samples: int = 1000,
    order: int = 10,
    seq_len: int = 16,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Get NARMA data formatted for Fourier-QLSTM.

    Fourier-QLSTM expects:
        - Input: [batch, seq_len] (univariate)
        - Will apply FFT internally with window_size

    Args:
        n_samples: Total samples
        order: NARMA order
        seq_len: Sequence length (should be >= window_size for FFT)
        batch_size: Batch size
        train_ratio: Training fraction
        val_ratio: Validation fraction
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader, seq_len
    """
    train_loader, val_loader, test_loader, _, _, _, _, _ = get_narma_dataloaders(
        n_samples=n_samples,
        order=order,
        seq_len=seq_len,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        output_format='lstm',  # [batch, seq_len]
        normalize='minmax',
        seed=seed
    )

    return train_loader, val_loader, test_loader, seq_len


def get_narma_for_fourier_qtcn(
    n_samples: int = 1000,
    order: int = 10,
    seq_len: int = 20,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, tuple, object]:
    """
    Get NARMA data formatted for Fourier-QTCN.

    Fourier-QTCN expects:
        - Input: [batch, channels, seq_len] for Conv1d
        - channels = 1 for univariate

    Args:
        n_samples: Total samples
        order: NARMA order
        seq_len: Sequence length
        batch_size: Batch size
        train_ratio: Training fraction
        val_ratio: Validation fraction
        seed: Random seed

    Returns:
        train_loader, val_loader, test_loader, input_dim, scaler
    """
    train_loader, val_loader, test_loader, input_dim, scaler, _, _, _ = get_narma_dataloaders(
        n_samples=n_samples,
        order=order,
        seq_len=seq_len,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        output_format='tcn',  # [batch, channels, seq_len]
        normalize='minmax',
        seed=seed
    )

    return train_loader, val_loader, test_loader, input_dim, scaler


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def analyze_narma_spectrum(
    order: int = 10,
    n_samples: int = 1000,
    seed: Optional[int] = None
) -> dict:
    """
    Analyze the frequency spectrum of NARMA series.

    This is useful for understanding what frequencies VQC should learn.

    Args:
        order: NARMA order
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Dictionary with spectrum analysis
    """
    _, y = generate_narma_series(n_samples, order=order, seed=seed)

    # Remove transient
    y = y[order:]

    # Compute FFT
    fft = np.fft.rfft(y)
    frequencies = np.fft.rfftfreq(len(y))
    magnitude = np.abs(fft)
    phase = np.angle(fft)

    # Find dominant frequencies
    sorted_indices = np.argsort(magnitude)[::-1]
    top_5_freqs = frequencies[sorted_indices[:5]]
    top_5_mags = magnitude[sorted_indices[:5]]

    return {
        'frequencies': frequencies,
        'magnitude': magnitude,
        'phase': phase,
        'dominant_frequencies': top_5_freqs,
        'dominant_magnitudes': top_5_mags,
        'mean': np.mean(y),
        'std': np.std(y),
        'min': np.min(y),
        'max': np.max(y)
    }


def plot_narma_analysis(order: int = 10, n_samples: int = 1000, seed: Optional[int] = None):
    """
    Plot NARMA series and its frequency spectrum.

    Args:
        order: NARMA order
        n_samples: Number of samples
        seed: Random seed
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return

    _, y = generate_narma_series(n_samples, order=order, seed=seed)
    analysis = analyze_narma_spectrum(order, n_samples, seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time series
    axes[0, 0].plot(y[order:order+200])
    axes[0, 0].set_title(f'NARMA-{order} Time Series (first 200 samples)')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True)

    # Histogram
    axes[0, 1].hist(y[order:], bins=50, density=True, alpha=0.7)
    axes[0, 1].set_title(f'NARMA-{order} Value Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True)

    # Frequency spectrum (magnitude)
    axes[1, 0].plot(analysis['frequencies'][:100], analysis['magnitude'][:100])
    axes[1, 0].set_title(f'NARMA-{order} Frequency Spectrum')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].grid(True)

    # Autocorrelation
    from numpy import correlate
    autocorr = correlate(y[order:] - np.mean(y[order:]), y[order:] - np.mean(y[order:]), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]
    axes[1, 1].plot(autocorr[:100])
    axes[1, 1].set_title(f'NARMA-{order} Autocorrelation')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'narma_{order}_analysis.png', dpi=150)
    print(f"Saved analysis plot to narma_{order}_analysis.png")
    plt.show()


# =============================================================================
# MAIN (DEMONSTRATION)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NARMA Dataset Generator")
    print("=" * 60)

    # Demonstrate basic usage
    print("\n1. Basic NARMA-10 generation:")
    u, y = generate_narma_series(n_samples=500, order=10, seed=42)
    print(f"   Input shape: {u.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Output range: [{y.min():.4f}, {y.max():.4f}]")

    # Demonstrate get_narma_data (QLSTM interface)
    print("\n2. Simple interface (for QLSTM):")
    x, y = get_narma_data(n_0=10, seq_len=8, n_samples=500, seed=42)
    print(f"   x shape: {x.shape}")
    print(f"   y shape: {y.shape}")

    # Demonstrate DataLoader interface
    print("\n3. DataLoader interface (for training):")
    train_loader, val_loader, test_loader, input_dim, scaler, _, train_size, val_size = \
        get_narma_dataloaders(
            n_samples=1000,
            order=10,
            seq_len=20,
            batch_size=32,
            output_format='tcn',
            seed=42
        )
    print(f"   Input dimension: {input_dim}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Demonstrate Fourier-QLSTM interface
    print("\n4. Fourier-QLSTM interface:")
    train_loader, val_loader, test_loader, seq_len = \
        get_narma_for_fourier_qlstm(n_samples=500, order=10, seq_len=16, seed=42)
    for x_batch, y_batch in train_loader:
        print(f"   Batch x shape: {x_batch.shape}")
        print(f"   Batch y shape: {y_batch.shape}")
        break

    # Demonstrate spectrum analysis
    print("\n5. Spectrum analysis:")
    analysis = analyze_narma_spectrum(order=10, n_samples=1000, seed=42)
    print(f"   Dominant frequencies: {analysis['dominant_frequencies'][:3]}")
    print(f"   Mean: {analysis['mean']:.4f}, Std: {analysis['std']:.4f}")

    # Generate different NARMA orders
    print("\n6. Multiple NARMA orders:")
    variants = generate_narma_variants(n_samples=500, orders=[5, 10, 30], seed=42)
    for name, data in variants.items():
        print(f"   {name}: mean={np.mean(data['output']):.4f}, std={np.std(data['output']):.4f}")

    print("\n" + "=" * 60)
    print("NARMA Generator Ready!")
    print("=" * 60)
