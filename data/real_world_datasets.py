"""
Real-world forecasting benchmark datasets for TCN model evaluation.

Datasets:
    - ETTh1: Electricity Transformer Temperature (7 features, 17,420 hourly steps)
    - Weather: 21 meteorological features, 52,696 steps at 10-min resolution
    - ECL: Electricity Consumption Load (321 clients, ~26,304 hourly steps)

These benchmarks are standard in top-tier time-series forecasting papers
(Autoformer, PatchTST, iTransformer, TimesNet).

Task: Next-step univariate target prediction using all channels as input.
Input shape: (batch, n_channels, seq_len) -> target: (batch,)

Author: Real-world benchmark loader for VQC-PeriodicData
Date: February 2026
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional
from pathlib import Path
import os
import urllib.request


# =============================================================================
# DATASET METADATA
# =============================================================================

DATASET_INFO = {
    'etth1': {
        'target': 'OT',
        'n_features': 7,
        'freq': 'hourly',
        'description': 'Electricity Transformer Temperature (hourly)',
        'url': 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv',
        'split': 'informer',  # 12/4/4 months
    },
    'weather': {
        'target': 'OT',
        'n_features': 21,
        'freq': '10min',
        'description': 'Weather (21 meteorological features, 10-min)',
        'url': None,  # Google Drive — manual download
        'split': 'ratio',  # 0.7/0.1/0.2
    },
    'ecl': {
        'target': 'MT_001',
        'n_features': 321,
        'freq': 'hourly',
        'description': 'Electricity Consumption Load (321 clients, hourly)',
        'url': None,  # Google Drive — manual download
        'split': 'ratio',  # 0.7/0.1/0.2
    },
}


# =============================================================================
# SEQUENCE CREATION
# =============================================================================

def create_multivariate_sequences(
    data: np.ndarray,
    target: np.ndarray,
    seq_len: int,
    pred_len: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for multivariate forecasting.

    Args:
        data: Input features, shape (timesteps, n_features)
        target: Target values, shape (timesteps,)
        seq_len: Look-back window length
        pred_len: Prediction horizon (1 = next-step)

    Returns:
        x: shape (n_sequences, n_features, seq_len) — TCN format
        y: shape (n_sequences,) — scalar target at pred_len steps ahead
    """
    n_total = len(data) - seq_len - pred_len + 1
    if n_total <= 0:
        raise ValueError(
            f"Not enough data: {len(data)} steps for seq_len={seq_len}, "
            f"pred_len={pred_len}. Need at least {seq_len + pred_len} steps."
        )

    x = np.zeros((n_total, data.shape[1], seq_len), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.float32)

    for i in range(n_total):
        # x: (n_features, seq_len) — channels-first for Conv1d/TCN
        x[i] = data[i:i + seq_len].T
        # y: scalar target at the prediction horizon
        y[i] = target[i + seq_len + pred_len - 1]

    return x, y


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def _download_etth1(data_dir: str) -> str:
    """Download ETTh1 CSV from GitHub if not present."""
    data_dir = Path(data_dir)
    save_path = data_dir / 'ETTh1.csv'

    if save_path.exists():
        return str(save_path)

    data_dir.mkdir(parents=True, exist_ok=True)
    url = DATASET_INFO['etth1']['url']
    print(f"Downloading ETTh1 from {url}...")

    try:
        urllib.request.urlretrieve(url, str(save_path))
        print(f"Saved to {save_path}")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download ETTh1: {e}\n"
            f"Please manually download from:\n"
            f"  {url}\n"
            f"And save to: {save_path}"
        )

    return str(save_path)


def _load_csv(dataset_name: str, data_path: Optional[str] = None) -> pd.DataFrame:
    """Load dataset CSV with auto-download for ETTh1."""
    base_dir = Path(__file__).parent.parent / 'data'

    if dataset_name == 'etth1':
        if data_path is None:
            data_path = _download_etth1(str(base_dir / 'etth1'))
        df = pd.read_csv(data_path)
        # Drop the date column
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        return df

    elif dataset_name == 'weather':
        if data_path is None:
            default_path = base_dir / 'weather' / 'weather.csv'
            if default_path.exists():
                data_path = str(default_path)
            else:
                raise FileNotFoundError(
                    f"Weather dataset not found at {default_path}\n"
                    f"Please download from the Autoformer repository:\n"
                    f"  https://drive.google.com/drive/folders/1ohGYWWfm4i9LC71pE29fhz4Y_UZcQYMZ\n"
                    f"Save 'weather.csv' to: {default_path.parent}/"
                )
        df = pd.read_csv(data_path)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        return df

    elif dataset_name == 'ecl':
        if data_path is None:
            default_path = base_dir / 'ecl' / 'electricity.csv'
            if default_path.exists():
                data_path = str(default_path)
            else:
                raise FileNotFoundError(
                    f"ECL dataset not found at {default_path}\n"
                    f"Please download from the Autoformer repository:\n"
                    f"  https://drive.google.com/drive/folders/1ohGYWWfm4i9LC71pE29fhz4Y_UZcQYMZ\n"
                    f"Save 'electricity.csv' to: {default_path.parent}/"
                )
        df = pd.read_csv(data_path)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])
        return df

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: etth1, weather, ecl")


def _get_split_indices(
    dataset_name: str,
    n_total: int
) -> Tuple[int, int]:
    """Return (train_end, val_end) indices for the dataset split."""
    if dataset_name == 'etth1':
        # Informer standard split: 12/4/4 months of hourly data
        train_end = 12 * 30 * 24  # 8640
        val_end = train_end + 4 * 30 * 24  # 8640 + 2880 = 11520
        # Clamp to actual data size
        train_end = min(train_end, int(0.6 * n_total))
        val_end = min(val_end, int(0.8 * n_total))
    else:
        # Standard 0.7/0.1/0.2 split for weather and ecl
        train_end = int(0.7 * n_total)
        val_end = int(0.8 * n_total)

    return train_end, val_end


# =============================================================================
# MAIN DATALOADER FUNCTION
# =============================================================================

def get_forecasting_dataloaders(
    dataset_name: str,
    data_path: Optional[str] = None,
    seq_len: int = 96,
    pred_len: int = 1,
    target_col: Optional[str] = None,
    n_channels: Optional[int] = None,
    batch_size: int = 32,
    normalize: str = 'standard',
    seed: Optional[int] = None,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, tuple, object, TensorDataset, int, int]:
    """
    Load a real-world forecasting dataset and create DataLoaders.

    Follows the same 8-tuple return pattern as get_narma_dataloaders().

    Args:
        dataset_name: 'etth1', 'weather', or 'ecl'
        data_path: Path to CSV file (auto-download for etth1 if None)
        seq_len: Look-back window length (default: 96)
        pred_len: Prediction horizon (default: 1 = next-step)
        target_col: Target column name (default per dataset)
        n_channels: Subset of input channels (None=all, int for first N)
        batch_size: Batch size for DataLoaders
        normalize: 'standard' (z-score), 'minmax', or 'none'
        seed: Random seed (for reproducibility of shuffling only)
        shuffle_train: Whether to shuffle training DataLoader

    Returns:
        train_loader, val_loader, test_loader, input_dim, scaler,
        full_dataset, train_size, val_size
    """
    dataset_name = dataset_name.lower()
    info = DATASET_INFO.get(dataset_name)
    if info is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: {list(DATASET_INFO.keys())}")

    print(f"\n{'='*60}")
    print(f"Loading {info['description']}")
    print(f"{'='*60}")

    # Load CSV
    df = _load_csv(dataset_name, data_path)
    print(f"Raw data shape: {df.shape} ({df.shape[0]} timesteps, {df.shape[1]} features)")

    # Determine target column
    if target_col is None:
        target_col = info['target']
    if target_col not in df.columns:
        # For ECL/Weather, target might be the last column named 'OT' or first col
        print(f"Warning: target '{target_col}' not found. Using last column: {df.columns[-1]}")
        target_col = df.columns[-1]

    # Channel subsetting (useful for ECL with 321 channels)
    if n_channels is not None and n_channels < df.shape[1]:
        # Always include target column
        cols = list(df.columns[:n_channels])
        if target_col not in cols:
            cols[-1] = target_col
        df = df[cols]
        print(f"Subsetted to {len(cols)} channels (including target '{target_col}')")

    # Split raw data BEFORE normalization (fit scaler on train only)
    n_total = len(df)
    train_end, val_end = _get_split_indices(dataset_name, n_total)

    print(f"Split: train[0:{train_end}], val[{train_end}:{val_end}], test[{val_end}:{n_total}]")

    # Extract features and target
    feature_cols = [c for c in df.columns if c != target_col] + [target_col]
    # Ensure target is included as a feature (all channels as input)
    data_array = df[feature_cols].values.astype(np.float32)
    target_idx = feature_cols.index(target_col)

    # Split
    train_data = data_array[:train_end]
    val_data = data_array[train_end:val_end]
    test_data = data_array[val_end:]

    # Normalize: fit on train only
    if normalize == 'standard':
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)
    elif normalize == 'minmax':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)
        test_data = scaler.transform(test_data)
    else:
        scaler = None

    # Create sequences for each split
    train_x, train_y = create_multivariate_sequences(
        train_data, train_data[:, target_idx], seq_len, pred_len
    )
    val_x, val_y = create_multivariate_sequences(
        val_data, val_data[:, target_idx], seq_len, pred_len
    )
    test_x, test_y = create_multivariate_sequences(
        test_data, test_data[:, target_idx], seq_len, pred_len
    )

    print(f"Sequences: train={len(train_x)}, val={len(val_x)}, test={len(test_x)}")
    print(f"Input shape: (batch, {train_x.shape[1]}, {train_x.shape[2]})")
    print(f"Target: '{target_col}' (next-step prediction, pred_len={pred_len})")

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.from_numpy(train_x), torch.from_numpy(train_y)
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_x), torch.from_numpy(val_y)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_x), torch.from_numpy(test_y)
    )

    # Full dataset for compatibility
    all_x = np.concatenate([train_x, val_x, test_x], axis=0)
    all_y = np.concatenate([train_y, val_y, test_y], axis=0)
    full_dataset = TensorDataset(
        torch.from_numpy(all_x), torch.from_numpy(all_y)
    )

    # Create DataLoaders
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=shuffle_train, drop_last=True, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=True
    )

    # input_dim: (batch_size, n_channels, seq_len)
    input_dim = (batch_size, train_x.shape[1], train_x.shape[2])

    print(f"DataLoaders created (batch_size={batch_size})")
    print(f"input_dim: {input_dim}")
    print(f"{'='*60}\n")

    return (
        train_loader,
        val_loader,
        test_loader,
        input_dim,
        scaler,
        full_dataset,
        len(train_dataset),
        len(val_dataset),
    )


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================

def get_etth1_dataloaders(**kwargs) -> Tuple:
    """Convenience wrapper for ETTh1 dataset."""
    return get_forecasting_dataloaders('etth1', **kwargs)


def get_weather_dataloaders(**kwargs) -> Tuple:
    """Convenience wrapper for Weather dataset."""
    return get_forecasting_dataloaders('weather', **kwargs)


def get_ecl_dataloaders(**kwargs) -> Tuple:
    """Convenience wrapper for ECL dataset. Defaults to 20 channels."""
    kwargs.setdefault('n_channels', 20)
    return get_forecasting_dataloaders('ecl', **kwargs)


# =============================================================================
# MAIN: Standalone demo
# =============================================================================

if __name__ == '__main__':
    print("Real-World Forecasting Dataset Loader Demo")
    print("=" * 60)

    # Demo: Load ETTh1 (auto-downloads)
    try:
        (train_loader, val_loader, test_loader, input_dim,
         scaler, full_dataset, train_size, val_size) = get_etth1_dataloaders(
            seq_len=96, batch_size=32, seed=2025
        )

        print(f"\nETTh1 loaded successfully!")
        print(f"  input_dim: {input_dim}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Train samples: {train_size}")
        print(f"  Val samples: {val_size}")

        # Quick sanity check
        for x, y in train_loader:
            print(f"\n  Sample batch:")
            print(f"    x shape: {x.shape}")
            print(f"    y shape: {y.shape}")
            print(f"    x range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"    y range: [{y.min():.3f}, {y.max():.3f}]")
            break

    except Exception as e:
        print(f"ETTh1 loading failed: {e}")

    print("\nDone.")
