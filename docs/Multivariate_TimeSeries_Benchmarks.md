# Multivariate Time-Series Benchmark Datasets for TCN Models

This document describes the real-world multivariate forecasting benchmarks added to the VQC-PeriodicData project. These datasets complement the existing synthetic benchmarks (NARMA, Multi-Sine, Mackey-Glass) and EEG classification task by providing standardized, widely-cited forecasting problems for evaluating quantum vs. classical TCN architectures.

---

## Motivation

Prior experiments in this project used only PhysioNet EEG (classification) and NARMA (regression). While these are informative, top-tier time-series venues (NeurIPS, ICML, ICLR) expect results on established multivariate forecasting benchmarks. The three datasets below are used in virtually every recent forecasting paper:

| Paper | Venue | ETTh1 | Weather | ECL |
|-------|-------|:-----:|:-------:|:---:|
| Autoformer (Wu et al., 2021) | NeurIPS 2021 | x | x | x |
| PatchTST (Nie et al., 2023) | ICLR 2023 | x | x | x |
| iTransformer (Liu et al., 2024) | ICLR 2024 | x | x | x |
| TimesNet (Wu et al., 2023) | ICLR 2023 | x | x | x |

---

## Dataset Overview

### 1. ETTh1 (Electricity Transformer Temperature - Hourly)

**Source:** [ETDataset](https://github.com/zhouhaoyi/ETDataset) (Zhou et al., Informer, AAAI 2021)

| Property | Value |
|----------|-------|
| Features | 7 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) |
| Target | OT (oil temperature) |
| Timesteps | 17,420 |
| Resolution | Hourly |
| Duration | ~2 years (July 2016 - June 2018) |
| Download | **Automatic** (from GitHub) |

**Feature descriptions:**
- **HUFL** - High UseFul Load
- **HULL** - High UseLess Load
- **MUFL** - Middle UseFul Load
- **MULL** - Middle UseLess Load
- **LUFL** - Low UseFul Load
- **LULL** - Low UseLess Load
- **OT** - Oil Temperature (prediction target)

**Train/Val/Test split (Informer standard):**
```
Train:  months 1-12  →  8,640 steps  (0:8640)
Val:    months 13-16 →  2,880 steps  (8640:11520)
Test:   months 17-20 →  5,900 steps  (11520:17420)
```

**Why ETTh1 matters for VQC research:**
- Strong diurnal (24h) and weekly (168h) periodic components in transformer loading patterns
- These periodic structures should align with VQC's native Fourier series output
- Small enough (7 features) to run efficiently with quantum simulation
- Universally used as the primary benchmark in forecasting papers

### 2. Weather

**Source:** [Autoformer repository](https://drive.google.com/drive/folders/1ohGYWWfm4i9LC71pE29fhz4Y_UZcQYMZ) (Wu et al., 2021)

| Property | Value |
|----------|-------|
| Features | 21 meteorological variables |
| Target | OT (last column) |
| Timesteps | 52,696 |
| Resolution | 10 minutes |
| Duration | ~1 year (2020) |
| Download | **Manual** (Google Drive) |

**Feature descriptions include:** air temperature, humidity, pressure, wind speed/direction, radiation, precipitation, and other meteorological observations from the Weather Station of the Max Planck Institute for Biogeochemistry in Jena, Germany.

**Train/Val/Test split (ratio-based):**
```
Train:  70%  →  36,887 steps
Val:    10%  →   5,270 steps
Test:   20%  →  10,539 steps
```

**Why Weather matters for VQC research:**
- Strong multi-scale periodicity: diurnal (144 steps), seasonal
- Higher dimensionality (21 features) tests scalability
- Fine temporal resolution (10-min) captures intra-day patterns
- Mix of periodic (temperature, radiation) and aperiodic (precipitation, wind gusts) features

### 3. ECL (Electricity Consumption Load)

**Source:** [Autoformer repository](https://drive.google.com/drive/folders/1ohGYWWfm4i9LC71pE29fhz4Y_UZcQYMZ) (Wu et al., 2021)

| Property | Value |
|----------|-------|
| Features | 321 electricity clients |
| Target | MT_001 (first client) |
| Timesteps | ~26,304 |
| Resolution | Hourly |
| Duration | ~3 years |
| Download | **Manual** (Google Drive) |

**Train/Val/Test split (ratio-based):**
```
Train:  70%  →  18,413 steps
Val:    10%  →   2,630 steps
Test:   20%  →   5,261 steps
```

**Default channel subsetting:** Because 321 channels is prohibitively large for quantum simulation, ECL defaults to `--n-channels=20` (first 20 clients including the target). This is configurable.

**Why ECL matters for VQC research:**
- Strongest periodic structure of all three datasets (clear diurnal + weekly patterns in electricity consumption)
- High dimensionality tests whether quantum circuits can extract useful features from many correlated channels
- Cross-client correlations provide rich structure for quantum entanglement-based processing

---

## Task Definition

All three datasets use the same task formulation, consistent with the existing NARMA regression setup:

**Next-step univariate prediction:**
```
Input:  x[t-96 : t]     shape (batch, n_channels, 96)   — all channels
Target: y[t]             shape (batch,)                  — single target column
```

- **Look-back window:** 96 steps (default, configurable via `--seq-len`)
- **Prediction horizon:** 1 step (next-step prediction, `pred_len=1`)
- **Loss function:** MSE (Mean Squared Error)
- **Evaluation metric:** RMSE (Root Mean Squared Error)
- **Normalization:** Z-score (StandardScaler), fit on training data only

---

## Data Pipeline

```
Raw CSV (timesteps x features)
    |
    ├── Drop date column
    ├── Optional: channel subsetting (--n-channels)
    |
    ├── Split: train / val / test (chronological, no shuffle)
    |
    ├── Normalize: StandardScaler fit on train, transform all
    |
    ├── Sliding window: create_multivariate_sequences()
    |   Input:  (timesteps, n_features)
    |   Output: x (n_seq, n_features, seq_len)  ← channels-first for Conv1d/TCN
    |           y (n_seq,)                       ← scalar target
    |
    └── DataLoader: batch_size=32, train shuffled, val/test sequential
```

---

## Usage

### Quick Start

```bash
cd /pscratch/sd/j/junghoon/VQC-PeriodicData

# ETTh1 (auto-downloads on first run)
python models/ReLU_TCN_EEG.py --dataset=etth1 --num-epochs=50

# Weather (requires manual download first)
python models/ReLU_TCN_EEG.py --dataset=weather --num-epochs=50

# ECL with 20 channels (requires manual download first)
python models/ReLU_TCN_EEG.py --dataset=ecl --n-channels=20 --num-epochs=50
```

### All 6 TCN Models

```bash
# Classical baselines
python models/ReLU_TCN_EEG.py     --dataset=etth1 --num-epochs=50
python models/Tanh_TCN_EEG.py     --dataset=etth1 --num-epochs=50
python models/Snake_TCN_EEG.py    --dataset=etth1 --num-epochs=50
python models/SIREN_TCN_EEG.py    --dataset=etth1 --num-epochs=50

# Quantum models
python models/FourierQTCN_EEG.py  --dataset=etth1 --num-epochs=50
python models/HQTCN2_EEG.py       --dataset=etth1 --num-epochs=50
```

### Dataset-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `eeg` | Dataset name: `eeg`, `narma`, `etth1`, `weather`, `ecl` |
| `--seq-len` | `96` | Look-back window length |
| `--batch-size` | `32` | Batch size |
| `--data-path` | auto | Path to CSV file |
| `--target-col` | per-dataset | Target column name |
| `--n-channels` | all (20 for ECL) | Number of input channels to use |
| `--task` | auto | Override: `classification` or `regression` |

### Backward Compatibility

The default `--dataset=eeg` preserves the original EEG classification behavior:

```bash
# These are equivalent to the original commands
python models/ReLU_TCN_EEG.py --freq=80 --n-sample=50
python models/ReLU_TCN_EEG.py --dataset=eeg --freq=80 --n-sample=50
```

### SLURM Batch Script Template

```bash
#!/bin/bash
#SBATCH -A m4138_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/VQC-PeriodicData

# Run all 6 models on ETTh1
for model in ReLU_TCN_EEG Tanh_TCN_EEG Snake_TCN_EEG SIREN_TCN_EEG FourierQTCN_EEG HQTCN2_EEG; do
    echo "===== Running ${model} on ETTh1 ====="
    python models/${model}.py --dataset=etth1 --num-epochs=50 --seed=2025
done
```

---

## Manual Download Instructions

### Weather Dataset

1. Go to: https://drive.google.com/drive/folders/1ohGYWWfm4i9LC71pE29fhz4Y_UZcQYMZ
2. Download `weather.csv`
3. Save to: `data/weather/weather.csv`

### ECL Dataset

1. Go to: https://drive.google.com/drive/folders/1ohGYWWfm4i9LC71pE29fhz4Y_UZcQYMZ
2. Download `electricity.csv`
3. Save to: `data/ecl/electricity.csv`

Expected directory structure after setup:
```
VQC-PeriodicData/
├── data/
│   ├── etth1/
│   │   └── ETTh1.csv          ← auto-downloaded
│   ├── weather/
│   │   └── weather.csv        ← manual download
│   ├── ecl/
│   │   └── electricity.csv    ← manual download
│   ├── real_world_datasets.py
│   └── narma_generator.py
```

---

## Architecture

### File Structure

| File | Purpose |
|------|---------|
| `data/real_world_datasets.py` | Dataset loading, normalization, sequence creation |
| `models/dataset_dispatcher.py` | Shared `--dataset` CLI args and `load_dataset()` dispatcher |
| `data/__init__.py` | Package exports |

### Dispatcher Pattern

All 6 TCN model files use a shared dispatcher to avoid duplicating dataset logic:

```python
# In each model's __main__:
from dataset_dispatcher import add_dataset_args, load_dataset

args = get_args()  # includes add_dataset_args(parser)
train_loader, val_loader, test_loader, input_dim, task, scaler = load_dataset(args, device)

# task = 'classification' for eeg, 'regression' for everything else
# Criterion, metric, and scheduler mode switch automatically based on task
```

### Task-Aware Training

The training loop automatically adapts based on the task:

| Component | Classification (EEG) | Regression (ETTh1/Weather/ECL/NARMA) |
|-----------|---------------------|--------------------------------------|
| Criterion | `BCEWithLogitsLoss` | `MSELoss` |
| Metric | ROC-AUC (higher = better) | RMSE (lower = better) |
| Scheduler | `mode='max'` | `mode='min'` |
| Best model | highest AUC | lowest RMSE |

---

## Model Comparison on ETTh1

Verification results from 1-epoch smoke tests (seed=2025, kernel_size=12, dilation=3):

| Model | Type | Params | Val RMSE | Test RMSE | Time/Epoch |
|-------|------|-------:|:--------:|:---------:|:----------:|
| ReLU-TCN | Classical baseline | 3,544 | 0.5693 | 1.0269 | ~1 min |
| Tanh-TCN | Classical baseline | 3,544 | 0.5730 | 1.3075 | ~1 min |
| Snake-TCN | Periodic (learnable a) | 3,640 | 0.5865 | 1.2556 | ~1 min |
| SIREN-TCN | Fourier baseline (sin) | 3,546 | 0.6768 | 1.5928 | ~1 min |
| FourierQTCN | Quantum + FFT | - | - | (verified training) | ~90 min |
| HQTCN2 | Quantum + FC | - | - | (verified training) | ~85 min |

**Note:** These are 1-epoch results for verification only. Full 50-epoch training is needed for meaningful comparison. Quantum models are ~90x slower than classical due to quantum circuit simulation overhead.

---

## Implementation Details

### Normalization

Z-score normalization (StandardScaler) is applied per-feature, fit on the training split only:

```python
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)    # fit + transform
val_data   = scaler.transform(val_data)           # transform only
test_data  = scaler.transform(test_data)           # transform only
```

This prevents data leakage from validation/test sets into training statistics.

### Sequence Creation

Sliding windows are created **after** splitting and normalizing, ensuring no look-ahead:

```python
def create_multivariate_sequences(data, target, seq_len, pred_len=1):
    """
    data:   (timesteps, n_features)
    target: (timesteps,)

    Returns:
        x: (n_sequences, n_features, seq_len)   # channels-first for TCN
        y: (n_sequences,)                         # next-step target
    """
```

### Return Format

The data loader returns an 8-tuple matching the existing `get_narma_dataloaders()` pattern:

```python
(train_loader, val_loader, test_loader, input_dim, scaler,
 full_dataset, train_size, val_size)
```

The dispatcher wraps this into a 6-tuple for model consumption:

```python
(train_loader, val_loader, test_loader, input_dim, task, scaler)
```

### Known Constraints

- **Kernel size / dilation:** With default `kernel_size=12, dilation=3`, the effective receptive field requires at least 34 time steps (`dilation * (kernel_size - 1) + 1 = 34`). All three datasets with `seq_len=96` satisfy this. For shorter sequences, reduce kernel_size or dilation via CLI.
- **ECL memory:** Full 321-channel ECL with seq_len=96 generates large tensors. Use `--n-channels=20` (default) or lower to stay within GPU memory.
- **Quantum circuit dtype:** PennyLane's `default.qubit` with backprop requires float64. The quantum models (FourierQTCN, HQTCN2) handle this by casting to double at the circuit boundary and back to float32 for the loss.
- **scipy.constants:** On this environment (scipy 1.10.1), `import scipy.constants` must appear before `import pennylane`. Both quantum model files include this workaround.

---

## Programmatic API

For use outside of the CLI model scripts:

```python
from data.real_world_datasets import (
    get_etth1_dataloaders,
    get_weather_dataloaders,
    get_ecl_dataloaders,
    get_forecasting_dataloaders,
    create_multivariate_sequences,
)

# ETTh1 (auto-downloads)
train_loader, val_loader, test_loader, input_dim, scaler, \
    full_dataset, train_size, val_size = get_etth1_dataloaders(
        seq_len=96, batch_size=32, seed=2025
    )

# Generic interface
result = get_forecasting_dataloaders(
    dataset_name='weather',
    seq_len=96,
    batch_size=32,
    normalize='standard',
    seed=2025,
)
```

---

## References

- Zhou, H., Zhang, S., Peng, J., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI 2021*.
- Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. *NeurIPS 2021*.
- Wu, H., Hu, T., Liu, Y., et al. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. *ICLR 2023*.
- Nie, Y., Nguyen, N.H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*.
- Liu, Y., Hu, T., Zhang, H., et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR 2024*.
- Schuld, M., Sweke, R., & Meyer, J.J. (2021). Effect of data encoding on the expressive power of variational quantum machine learning models. *Physical Review A*, 103(3), 032430.
- Ziyin, L., Hartwig, T., & Ueda, M. (2020). Neural networks fail to learn periodic functions and how to fix it. *NeurIPS 2020*.
