"""
Dataset dispatcher for TCN models.

Provides shared argparse arguments and a unified load_dataset() function
so that each TCN model can support EEG, NARMA, ETTh1, Weather, and ECL
without duplicating 40+ lines of dataset loading logic.

Usage in model files:
    from dataset_dispatcher import add_dataset_args, load_dataset

    def get_args():
        parser = argparse.ArgumentParser(...)
        # ... model-specific args ...
        add_dataset_args(parser)
        return parser.parse_args()

    if __name__ == "__main__":
        args = get_args()
        train_loader, val_loader, test_loader, input_dim, task, scaler = load_dataset(args, device)

Author: Dataset dispatcher for VQC-PeriodicData TCN models
Date: February 2026
"""

import os
import sys
import torch

# =============================================================================
# TASK MAPPING
# =============================================================================

DATASET_TASK_MAP = {
    'eeg': 'classification',
    'narma': 'regression',
    'etth1': 'regression',
    'weather': 'regression',
    'ecl': 'regression',
}


def add_dataset_args(parser):
    """
    Add dataset-related arguments to an argparse parser.

    These are shared across all TCN models. Model-specific args
    (like --mlp-dim, --n-qubits) remain in each model file.
    """
    parser.add_argument(
        "--dataset", type=str, default="eeg",
        choices=list(DATASET_TASK_MAP.keys()),
        help="Dataset to use (default: eeg)"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        choices=["classification", "regression"],
        help="Override task type (default: auto from dataset)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=96,
        help="Sequence length for forecasting datasets (default: 96)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to dataset CSV (auto for etth1)"
    )
    parser.add_argument(
        "--target-col", type=str, default=None,
        help="Target column name (default per dataset)"
    )
    parser.add_argument(
        "--n-channels", type=int, default=None,
        help="Subset of input channels (default: all; useful for ECL)"
    )


def load_dataset(args, device):
    """
    Dispatch to the appropriate data loader based on args.dataset.

    Returns:
        train_loader, val_loader, test_loader, input_dim, task, scaler

    Where:
        - input_dim: tuple (batch_size, n_channels, seq_len)
        - task: 'classification' or 'regression'
        - scaler: fitted scaler (or None for EEG)
    """
    dataset = args.dataset.lower()
    task = args.task if args.task else DATASET_TASK_MAP.get(dataset, 'classification')
    batch_size = getattr(args, 'batch_size', 32)

    if dataset == 'eeg':
        # Import EEG loader
        try:
            from Load_PhysioNet_EEG import load_eeg_ts_revised
        except ImportError:
            sys.path.insert(0, os.path.dirname(__file__))
            from Load_PhysioNet_EEG import load_eeg_ts_revised

        freq = getattr(args, 'freq', 80)
        n_sample = getattr(args, 'n_sample', 50)
        train_loader, val_loader, test_loader, input_dim = load_eeg_ts_revised(
            seed=args.seed, device=device, batch_size=batch_size,
            sampling_freq=freq, sample_size=n_sample
        )
        return train_loader, val_loader, test_loader, input_dim, task, None

    elif dataset == 'narma':
        # Import NARMA loader
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.narma_generator import get_narma_dataloaders

        result = get_narma_dataloaders(
            n_samples=2000,
            order=10,
            seq_len=getattr(args, 'seq_len', 20),
            batch_size=batch_size,
            output_format='tcn',
            normalize='standard',
            seed=args.seed
        )
        train_loader, val_loader, test_loader, input_dim, scaler = (
            result[0], result[1], result[2], result[3], result[4]
        )
        return train_loader, val_loader, test_loader, input_dim, task, scaler

    elif dataset in ('etth1', 'weather', 'ecl'):
        # Import real-world dataset loader
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from data.real_world_datasets import get_forecasting_dataloaders

        n_channels = getattr(args, 'n_channels', None)
        # Default ECL to 20 channels if not specified
        if dataset == 'ecl' and n_channels is None:
            n_channels = 20

        result = get_forecasting_dataloaders(
            dataset_name=dataset,
            data_path=getattr(args, 'data_path', None),
            seq_len=getattr(args, 'seq_len', 96),
            pred_len=1,
            target_col=getattr(args, 'target_col', None),
            n_channels=n_channels,
            batch_size=batch_size,
            normalize='standard',
            seed=args.seed,
        )
        train_loader, val_loader, test_loader, input_dim, scaler = (
            result[0], result[1], result[2], result[3], result[4]
        )
        return train_loader, val_loader, test_loader, input_dim, task, scaler

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASET_TASK_MAP.keys())}")
