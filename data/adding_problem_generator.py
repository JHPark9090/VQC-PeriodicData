"""
Adding Problem Dataset Generator

The Adding Problem (Hochreiter & Schmidhuber, 1997) is a long-range dependency
benchmark. Each sample is an independent sequence:

    Signal: s_i ~ U(0, 1) for i=1..T
    Mask:   m_i ∈ {0, 1}, exactly 2 positions set to 1
            (one in the first half, one in the second half)
    Target: y = s_{j1} + s_{j2} where j1, j2 are marked positions
    Input:  x = [s_1, ..., s_T, m_1, ..., m_T] (concatenated, length 2T)

This tests whether periodic gating helps with selective memory — the model
must learn to attend to exactly 2 out of T positions.

Note: This is a sequence-to-scalar task, NOT a sliding-window prediction.
The returned x has shape [n_samples, 2*T], and models should use
--window-size=T (half the actual input length).

This module provides:
    - generate_adding_sequences(): Generate raw adding problem data
    - get_adding_data(): Simple interface for QLSTM (mirrors get_narma_data)
    - get_adding_dataloaders(): Full DataLoader interface for training

Author: VQC-PeriodicData benchmark suite
Date: February 2026
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


# =============================================================================
# ADDING PROBLEM SEQUENCE GENERATION
# =============================================================================

def generate_adding_sequences(
    n_samples: int = 500,
    T: int = 50,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Adding Problem sequences.

    Each sample:
        - Signal s_i ~ U(0, 1) for i=1..T
        - Mask with exactly 2 ones: one in [0, T/2), one in [T/2, T)
        - Target y = s[pos1] + s[pos2]
        - Input x = concat(signal, mask) of length 2T

    Args:
        n_samples: Number of independent sequences to generate
        T: Sequence length (controls difficulty)
        seed: Random seed for reproducibility

    Returns:
        x: Input sequences [n_samples, 2*T] (signal concatenated with mask)
        y: Target sums [n_samples]
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random signals
    signals = np.random.uniform(0, 1, size=(n_samples, T))

    # Generate masks with exactly 2 marked positions per sample
    masks = np.zeros((n_samples, T))

    # First marked position: uniformly in first half [0, T/2)
    half = T // 2
    pos1 = np.random.randint(0, half, size=n_samples)

    # Second marked position: uniformly in second half [T/2, T)
    pos2 = np.random.randint(half, T, size=n_samples)

    for i in range(n_samples):
        masks[i, pos1[i]] = 1.0
        masks[i, pos2[i]] = 1.0

    # Compute targets: sum of signal values at marked positions
    targets = np.sum(signals * masks, axis=1)

    # Concatenate signal and mask: [n_samples, 2*T]
    x = np.concatenate([signals, masks], axis=1)

    return x, targets


# =============================================================================
# SIMPLE INTERFACE FOR QLSTM (mirrors get_narma_data)
# =============================================================================

def get_adding_data(
    T: int = 50,
    n_samples: int = 500,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple interface for getting Adding Problem data (compatible with QLSTM).

    Note: The returned x has shape [n_samples, 2*T] because signal and mask
    are concatenated. Models should treat the full 2*T as the sequence length.

    Args:
        T: Sequence length (half the actual input length)
        n_samples: Number of independent sequences
        seed: Random seed

    Returns:
        x: Input sequences [n_samples, 2*T]
        y: Target sums [n_samples], normalized to [-1, 1]
    """
    # Generate adding sequences
    x, y = generate_adding_sequences(n_samples, T=T, seed=seed)

    # Normalize targets to [-1, 1]
    # Targets are in [0, 2] (sum of two U(0,1) values)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Convert to tensors
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y_normalized).float()

    return x_tensor, y_tensor


# =============================================================================
# FULL DATALOADER INTERFACE
# =============================================================================

def get_adding_dataloaders(
    n_samples: int = 2000,
    T: int = 50,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    normalize_range: Tuple[float, float] = (-1, 1),
    seed: Optional[int] = None,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, tuple, object, TensorDataset, int, int]:
    """
    Generate Adding Problem data and create DataLoaders for training.

    Args:
        n_samples: Total number of sequences
        T: Sequence length per sample
        batch_size: Batch size for DataLoaders
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set (rest is test)
        normalize_range: Range for target normalization
        seed: Random seed
        shuffle_train: Whether to shuffle training data

    Returns:
        train_loader, val_loader, test_loader, input_dim, scaler,
        full_dataset, train_size, val_size
    """
    print(f"Generating Adding Problem (T={T}) data...")

    # Generate sequences
    x, y = generate_adding_sequences(n_samples, T=T, seed=seed)

    # Normalize targets
    scaler = MinMaxScaler(feature_range=normalize_range)
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Convert to tensors
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y_normalized).float()

    # Create dataset
    full_dataset = TensorDataset(x_tensor, y_tensor)

    # Random split is fine here (sequences are independent)
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

    input_dim = (batch_size, 2 * T)

    print(f"Data generated successfully!")
    print(f"  Total sequences: {n_total}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  Input shape: {x_tensor.shape} (signal + mask)")
    print(f"  T={T}, input length=2*T={2*T}")

    return (
        train_loader, val_loader, test_loader,
        input_dim, scaler, full_dataset,
        len(train_dataset), len(val_dataset)
    )


# =============================================================================
# MAIN (DEMONSTRATION)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Adding Problem Dataset Generator")
    print("=" * 60)

    # Basic generation
    print("\n1. Basic Adding Problem (T=50) generation:")
    x, y = generate_adding_sequences(n_samples=500, T=50, seed=42)
    print(f"   x shape: {x.shape}  (signal + mask concatenated)")
    print(f"   y shape: {y.shape}")
    print(f"   y range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"   y mean:  {y.mean():.4f} (expected ~1.0)")

    # Verify mask structure
    signals = x[:, :50]
    masks = x[:, 50:]
    print(f"\n   Mask sum per sample (should be 2): {masks.sum(axis=1)[:5]}")
    print(f"   First sample marked positions: {np.where(masks[0] == 1)[0]}")
    print(f"   First sample target: {y[0]:.4f}")
    recomputed = np.sum(signals[0] * masks[0])
    print(f"   Recomputed target:   {recomputed:.4f}")

    # Simple interface
    print("\n2. Simple interface (for QLSTM):")
    x_t, y_t = get_adding_data(T=50, n_samples=500, seed=42)
    print(f"   x shape: {x_t.shape}")
    print(f"   y shape: {y_t.shape}")
    print(f"   y range: [{y_t.min():.4f}, {y_t.max():.4f}]")

    # Different T values
    print("\n3. Difficulty scaling:")
    for T in [10, 50, 100, 200]:
        x_t, y_t = get_adding_data(T=T, n_samples=100, seed=42)
        print(f"   T={T:3d}: x shape={x_t.shape}, y range=[{y_t.min():.2f}, {y_t.max():.2f}]")

    print("\n" + "=" * 60)
    print("Adding Problem Generator Ready!")
    print("=" * 60)
