# Fourier-Based Quantum Temporal Convolutional Network (F-QTCN)

A minimal, theoretically-grounded implementation of Quantum TCN that fully utilizes VQC's periodic advantages for EEG classification.

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture](#architecture)
4. [Key Components](#key-components)
5. [Comparison with Original QTCN](#comparison-with-original-qtcn)
6. [Usage](#usage)
7. [Hyperparameters](#hyperparameters)
8. [Output Files](#output-files)
9. [References](#references)

---

## Overview

### The Problem

The original QTCN (Hybrid Quantum TCN) does **not** utilize VQC's periodic advantages because:

1. **No frequency analysis**: Raw time-domain data is fed directly into VQC
2. **Generic encoding**: Uses standard `AngleEmbedding` without frequency matching
3. **Information-destroying aggregation**: `torch.mean()` destroys periodic structure
4. **FC layer bottleneck**: Fully-connected dimension reduction loses frequency information

### The Solution

Fourier-Based QTCN addresses all these issues with minimal classical computation:

| Issue | Original QTCN | Fourier QTCN Solution |
|-------|---------------|----------------------|
| No frequency analysis | Raw time-domain input | FFT-based frequency extraction |
| Generic encoding | Standard RY gates | RY + frequency-scaled RX encoding |
| Blind freq_scale init | N/A | FFT-seeded initialization (data-informed) |
| Mean aggregation | `torch.mean()` | Learnable weighted aggregation |
| FC bottleneck | Multi-layer FC | Single linear projection |

---

## Theoretical Foundation

### VQC as Fourier Series

From Schuld et al. (2021), a VQC with angle encoding computes:

$$f(x) = \sum_{\omega \in \Omega} c_\omega e^{i\omega x}$$

where:
- $\Omega$ is the frequency spectrum determined by encoding Hamiltonians
- $c_\omega$ are trainable Fourier coefficients (determined by circuit parameters)

**Key insight**: VQCs are naturally Fourier series generators. To leverage this:
1. Extract frequency content from input (via FFT)
2. Match input frequencies to VQC's frequency spectrum (via learnable scaling)

### Why FFT Instead of Bandpass Filtering?

| Method | Information Loss | Computational Cost | VQC Alignment |
|--------|------------------|-------------------|---------------|
| Bandpass Filtering | **High** (discards out-of-band) | O(n) per band | Indirect |
| FFT | **None** (invertible) | O(n log n) | **Direct** (Fourier ↔ Fourier) |

FFT extracts Fourier coefficients directly, which naturally align with VQC's Fourier series output.

### Learnable Frequency Rescaling

Similar to Snake activation's learnable parameter `a`:

$$\text{Snake}_a(x) = x + \frac{1}{a}\sin^2(ax)$$

We introduce learnable `freq_scale` parameters:

$$\text{Encoding}: \quad RY(\theta_i) \cdot RX(\alpha_i \cdot \theta_i)$$

where $\alpha_i$ is learned per qubit, allowing the VQC to adapt its frequency spectrum to match the data's periodic content.

---

## Architecture

```
Input EEG [batch, channels, time]
    │
    ▼
┌─────────────────────────────────┐
│   Sliding Window Extraction     │
│   (kernel_size, dilation)       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   FFT-based Frequency           │  ◄── LOSSLESS extraction
│   Extraction                    │      Magnitude + Phase
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Simple Linear Projection      │  ◄── Minimal classical computation
│   (fft_features → n_qubits)     │      Single layer, no hidden layers
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Frequency-Matched Quantum     │  ◄── KEY: Learnable freq_scale
│   Encoding (RY + scaled RX)     │      Aligns VQC spectrum to data
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Quantum Conv-Pool Layers      │  ◄── U3 + Ising gates
│   (circuit_depth iterations)    │      Mid-circuit measurements
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   Learnable Weighted            │  ◄── Preserves periodicity
│   Aggregation                   │      (unlike torch.mean)
└─────────────────────────────────┘
    │
    ▼
Output Prediction [batch]
```

---

## Key Components

### 1. FFT-based Frequency Extraction

```python
def _extract_fft_features(self, window):
    # Apply FFT (lossless transformation)
    fft_result = torch.fft.rfft(window, dim=-1)

    # Select first n_frequencies (lower frequencies contain most EEG energy)
    fft_selected = fft_result[:, :, :self.n_frequencies]

    # Extract magnitude (amplitude) and phase (timing)
    magnitude = torch.log1p(torch.abs(fft_selected))  # Log scale for dynamic range
    phase = torch.angle(fft_selected) / np.pi         # Normalize to [-1, 1]

    return torch.stack([magnitude, phase], dim=-1).reshape(batch_size, -1)
```

**Why magnitude and phase?**
- **Magnitude**: Captures the strength of each frequency component
- **Phase**: Captures the temporal alignment of oscillations
- Together they provide complete frequency-domain information

### 2. Learnable Frequency Rescaling with FFT-Seeded Initialization

The `freq_scale` parameter is **essential** for VQC's periodic advantage:
- Maps input frequency content to VQC's natural frequency spectrum
- Learned during training to optimize for the specific dataset
- Analogous to Snake activation's learnable `a` parameter

**Initialization methods** (controlled by `--freq-init`):

| Method | Flag | Description |
|--------|------|-------------|
| **FFT-seeded** (default) | `--freq-init=fft` | Analyzes training data via FFT to find dominant frequencies, uses ratio-based scaling to seed `freq_scale` |
| **Linspace** (legacy) | `--freq-init=linspace` | Generic `linspace(0.5, 3.0)` initialization (original behavior) |

**FFT-seeded initialization** (hybrid of Methods 3 + 4 from the encoding guide):

```python
# Analyzes training data power spectrum, finds top-N frequency bins,
# converts to ratio-based scaling factors clamped to [0.5, 5.0]
freq_scale_init = analyze_training_frequencies(train_loader, n_qubits)

# Parameter remains learnable — FFT just provides a better starting point
self.freq_scale = nn.Parameter(freq_scale_init.clone().float())
```

**Why FFT-seeded?**
- Avoids local minima from blind initialization
- Preserves actual frequency relationships in the data (ratio-based)
- Different datasets produce different initializations (e.g., EEG vs ETTh1)
- Parameter remains fully learnable via backpropagation

**Linspace fallback** (backward compatible):

```python
# Original generic initialization
self.freq_scale = nn.Parameter(torch.linspace(0.5, 3.0, n_qubits))
```

### 3. Frequency-Matched Encoding

```python
def _circuit(self, features):
    for i, wire in enumerate(wires):
        # Standard amplitude encoding
        qml.RY(features[i], wires=wire)

        # FREQUENCY-MATCHED encoding (KEY modification)
        qml.RX(self.freq_scale[i] * features[i], wires=wire)
```

The combination of RY and frequency-scaled RX:
1. RY encodes the amplitude information
2. RX with `freq_scale` matches VQC's frequency to data's frequency content
3. Together they enable full utilization of VQC's Fourier structure

### 4. Quantum Convolutional Layer

```python
def _apply_convolution(self, weights, wires):
    for parity in [0, 1]:
        for idx, w in enumerate(wires):
            if idx % 2 == parity and idx < n_wires - 1:
                # Single-qubit rotations
                qml.U3(*weights[idx, :3], wires=w)
                qml.U3(*weights[idx + 1, 3:6], wires=next_w)

                # Two-qubit Ising interactions
                qml.IsingZZ(weights[idx, 6], wires=[w, next_w])
                qml.IsingYY(weights[idx, 7], wires=[w, next_w])
                qml.IsingXX(weights[idx, 8], wires=[w, next_w])

                # Post-interaction rotations
                qml.U3(*weights[idx, 9:12], wires=w)
                qml.U3(*weights[idx + 1, 12:15], wires=next_w)
```

### 5. Learnable Weighted Aggregation

```python
# Instead of torch.mean() which destroys periodicity
weights = F.softmax(self.agg_weights[:n_windows], dim=0)
output = (outputs * weights.unsqueeze(0)).sum(dim=1)
```

**Why not mean?**
- `torch.mean()` is equivalent to a low-pass filter
- Destroys high-frequency periodic information
- Learnable weights preserve the periodic structure

---

## Comparison with Original QTCN

```
╔══════════════════════════════════════════════════════════════════════╗
║            ORIGINAL QTCN vs FOURIER-BASED QTCN                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ORIGINAL QTCN:                                                      ║
║  ├─ Encoding: Generic AngleEmbedding (RY only)                       ║
║  ├─ Frequency Matching: None                                         ║
║  ├─ Preprocessing: FC layer (information loss)                       ║
║  ├─ Aggregation: Simple mean (destroys periodicity)                  ║
║  └─ VQC Periodic Advantage: NOT UTILIZED                             ║
║                                                                      ║
║  FOURIER-BASED QTCN:                                                 ║
║  ├─ Encoding: RY (amplitude) + RX (frequency-scaled)                 ║
║  ├─ Frequency Matching: Learnable freq_scale (like Snake's 'a')      ║
║  ├─ Preprocessing: FFT (lossless) + Linear projection                ║
║  ├─ Aggregation: Learnable weighted sum                              ║
║  └─ VQC Periodic Advantage: FULLY UTILIZED                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Detailed Comparison

| Aspect | Original QTCN | Fourier QTCN |
|--------|---------------|--------------|
| **Frequency Extraction** | None (raw time-domain) | FFT-based (complete spectrum) |
| **Information Loss** | High (FC layer compression) | None (FFT is invertible) |
| **Frequency Matching** | None | Learnable `freq_scale` per qubit |
| **freq_scale Init** | N/A | FFT-seeded (data-informed) or linspace (generic) |
| **Encoding Gates** | RY only | RY + frequency-scaled RX |
| **Temporal Aggregation** | `torch.mean()` | Learnable weighted sum |
| **Classical Overhead** | Medium (multi-layer FC) | Low (single linear layer) |
| **VQC Periodic Advantage** | Not utilized | Fully utilized |
| **Theoretical Grounding** | Ad-hoc design | Based on Schuld et al. (2021) |

---

## Usage

### Basic Usage (FFT-seeded init, default)

```bash
cd /pscratch/sd/j/junghoon/VQC-PeriodicData

python models/FourierQTCN_EEG.py \
    --dataset=eeg \
    --n-qubits=8 \
    --circuit-depth=2 \
    --freq=80 \
    --n-sample=50 \
    --num-epochs=50 \
    --lr=0.001 \
    --seed=2025
```

This uses `--freq-init=fft` by default, which analyzes the training data to seed `freq_scale`.

### With Generic Linspace Init (legacy behavior)

```bash
python models/FourierQTCN_EEG.py \
    --dataset=eeg \
    --n-qubits=8 \
    --circuit-depth=2 \
    --freq-init=linspace \
    --num-epochs=50
```

### With Custom FFT Frequencies

```bash
python models/FourierQTCN_EEG.py \
    --dataset=etth1 \
    --n-qubits=8 \
    --circuit-depth=2 \
    --n-frequencies=16 \
    --kernel-size=12 \
    --dilation=3 \
    --freq-init=fft \
    --num-epochs=100
```

### Resume from Checkpoint

```bash
python models/FourierQTCN_EEG.py \
    --dataset=eeg \
    --n-qubits=8 \
    --circuit-depth=2 \
    --resume \
    --num-epochs=100
```

### SLURM Submission Example

```bash
#!/bin/bash
#SBATCH --job-name=FourierQTCN
#SBATCH --account=m4138_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32

export PYTHONNOUSERSITE=1
module load conda
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

python models/FourierQTCN_EEG.py \
    --dataset=eeg \
    --n-qubits=8 \
    --circuit-depth=2 \
    --freq=80 \
    --n-sample=50 \
    --freq-init=fft \
    --num-epochs=100 \
    --seed=2025
```

---

## Hyperparameters

### Quantum Circuit Parameters

| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `n_qubits` | 8 | Number of qubits | 4, 6, 8, 12 |
| `circuit_depth` | 2 | Number of conv-pool layers | 1-4 |
| `n_frequencies` | `n_qubits` | FFT frequencies to extract | n_qubits to 2*n_qubits |
| `freq_init` | `fft` | freq_scale initialization method | `fft` (data-informed) or `linspace` (generic) |

### Temporal Parameters

| Parameter | Default | Description | Notes |
|-----------|---------|-------------|-------|
| `kernel_size` | 12 | Sliding window size | Larger = more context |
| `dilation` | 3 | Dilation factor | Controls receptive field |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 0.001 | Learning rate |
| `num_epochs` | 50 | Training epochs |
| `seed` | 2025 | Random seed |

### Data Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `freq` | 80 | Sampling frequency (Hz) |
| `n_sample` | 50 | Number of samples |

---

## Output Files

### During Training

1. **Checkpoint file**: `FourierQTCN_checkpoints/fourier_qtcn_q{n_qubits}_d{circuit_depth}_seed{seed}.pth`
   - Contains: model state, optimizer state, metrics history, best validation AUC

### After Training

2. **Metrics CSV**: `FourierQTCN_q{n_qubits}_d{circuit_depth}_seed{seed}_metrics.csv`
   - Columns: epoch, train_loss, train_auc, valid_loss, valid_auc, test_loss, test_auc

3. **Frequency Scales CSV**: `FourierQTCN_q{n_qubits}_d{circuit_depth}_seed{seed}_freqscales.csv`
   - Columns: qubit, freq_scale
   - Shows learned frequency scaling factors for each qubit

### Interpreting Frequency Scales

After training, examine the learned `freq_scale` values:

```python
import pandas as pd
df = pd.read_csv('FourierQTCN_q8_d2_seed2025_freqscales.csv')
print(df)
```

| qubit | freq_scale | Interpretation |
|-------|------------|----------------|
| 0 | 0.52 | Tuned to lower frequencies (Delta/Theta) |
| 1 | 0.89 | Tuned to Alpha band |
| 2 | 1.45 | Tuned to Beta band |
| ... | ... | ... |
| 7 | 2.87 | Tuned to higher frequencies (Gamma) |

---

## File Location

**Implementation**: `/pscratch/sd/j/junghoon/VQC-PeriodicData/models/FourierQTCN_EEG.py`

**Related Documentation**:
- `VQC_Periodic_Data_Encoding_Guide.md` - Encoding strategies for VQCs
- `QTCN_Periodic_Advantage_Analysis.md` - Analysis of original QTCN's limitations
- `VQC_vs_ReLU_Periodic_Data_Comparison.md` - Theoretical comparison with classical methods

---

## References

1. **Schuld, M., Sweke, R., & Meyer, J. J.** (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A*, 103(3), 032430.
   - Foundation for understanding VQC as Fourier series

2. **Ziyin, L., Hartwig, T., & Ueda, M.** (2020). Neural networks fail to learn periodic functions and how to fix it. *Advances in Neural Information Processing Systems*, 33.
   - Snake activation and periodic function learning

3. **Pérez-Salinas, A., et al.** (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.
   - Data re-uploading and frequency spectrum expansion

---

## Summary

Fourier-Based QTCN is designed with one goal: **fully utilize VQC's periodic advantages** with **minimal classical computation**.

The key innovations are:
1. **FFT for lossless frequency extraction** (not bandpass filtering)
2. **Learnable frequency rescaling** with **FFT-seeded initialization** (data-informed starting point, like Snake's `a` parameter)
3. **Frequency-matched quantum encoding** (RY + scaled RX)
4. **Learnable weighted aggregation** (not mean)

This design is grounded in the theoretical understanding that VQCs are Fourier series generators (Schuld et al., 2021), and the input preprocessing should preserve and align frequency information rather than destroying it.
