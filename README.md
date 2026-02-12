# VQC-PeriodicData: Leveraging Quantum Periodic Advantages for Time-Series

A research project exploring how to fully utilize Variational Quantum Circuits' (VQC) inherent periodic structure for time-series analysis, with applications to EEG classification and NARMA prediction.

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
   - [Can VQCs Learn Periodic Data Better?](#can-vqcs-learn-periodic-data-better)
   - [VQC as Fourier Series](#vqc-as-fourier-series)
   - [When Do VQCs Outperform Classical Models?](#when-do-vqcs-outperform-classical-models)
3. [Limitations of Existing Models](#limitations-of-existing-models)
   - [QTCN Limitations](#qtcn-limitations)
   - [QLSTM Limitations](#qlstm-limitations)
4. [Our Solutions](#our-solutions)
   - [FourierQTCN](#fourierqtcn)
   - [FourierQLSTM](#fourierqlstm)
5. [Experiments](#experiments)
   - [PhysioNet EEG Classification](#physionet-eeg-classification)
   - [NARMA Time-Series Prediction](#narma-time-series-prediction)
6. [Installation & Usage](#installation--usage)
7. [Project Structure](#project-structure)
8. [References](#references)

---

## Overview

### The Central Question

> **Can Variational Quantum Circuits (VQCs) outperform classical neural networks for periodic data?**

This project investigates this question and provides:
- Theoretical analysis of VQC's periodic capabilities
- Identification of why existing quantum models (QTCN, QLSTM) fail to leverage these capabilities
- Novel architectures (FourierQTCN, FourierQLSTM) that fully utilize VQC's periodic advantages
- Experimental validation on EEG and NARMA benchmarks

### Key Findings

| Finding | Implication |
|---------|-------------|
| VQCs compute Fourier series | Natural fit for periodic/oscillatory data |
| Original QTCN ignores frequency structure | Periodic advantage not utilized |
| Original QLSTM destroys periodicity | sigmoid/tanh after VQC eliminates periodic structure |
| FFT + frequency-matched encoding | Enables full periodic advantage |
| SIREN also computes Fourier series | Strongest classical challenger — isolates quantum vs classical Fourier |

---

## Theoretical Background

### Can VQCs Learn Periodic Data Better?

This is the foundational question of this project. We investigate:

1. **What is VQC's mathematical structure?**
   - VQCs with angle encoding compute truncated Fourier series
   - The frequency spectrum is determined by encoding Hamiltonians

2. **Does this give advantages for periodic data?**
   - Yes, for specific tasks (extrapolation, generalization)
   - Not automatically - requires proper encoding and architecture

3. **How do VQCs compare to classical alternatives?**
   - vs ReLU/tanh: VQC better for periodic extrapolation
   - vs Snake activation: Comparable for interpolation, VQC better for extrapolation

### VQC as Fourier Series

From **Schuld et al. (2021)**, a VQC with angle encoding computes:

$$f(x) = \sum_{\omega \in \Omega} c_\omega e^{i\omega x}$$

where:
- **Ω** = frequency spectrum, determined by the encoding Hamiltonian
- **c_ω** = trainable Fourier coefficients, determined by circuit parameters

#### Frequency Spectrum Examples

| Encoding | Hamiltonian | Frequency Spectrum |
|----------|-------------|-------------------|
| Single RY(x) | σ_y | {-1, 0, 1} |
| RY(x) repeated r times | r·σ_y | {-r, ..., 0, ..., r} |
| RY(x) + RZ(x) | σ_y + σ_z | Richer spectrum |

#### Key Insight

**VQCs are not generic function approximators - they are Fourier series generators with trainable coefficients.**

This means:
- VQCs naturally express periodic functions
- The frequency spectrum must match the data's frequency content
- Proper encoding is essential to leverage this structure

### When Do VQCs Outperform Classical Models?

#### Proven Advantages

| Task | VQC Advantage | Reason |
|------|---------------|--------|
| **Periodic Extrapolation** | Strong | VQC maintains periodicity outside training interval |
| **Sample Efficiency** | Moderate | Fourier structure as inductive bias |
| **Approximation of Periodic Functions** | O(1/n) error | Optimal for Fourier-representable functions |

#### Comparison with Classical Activations

| Activation | Periodic Interpolation | Periodic Extrapolation |
|------------|----------------------|------------------------|
| **ReLU** | Poor (piecewise linear) | Fails (linear extrapolation) |
| **tanh** | Moderate | Fails (saturates to constant) |
| **Snake** | Good (learnable frequency) | Moderate |
| **SIREN** | **Good** (native Fourier) | **Good** (maintains periodicity) |
| **VQC** | Good | **Best** (maintains periodicity) |

#### What's NOT Proven

- Faster convergence (optimization landscape unclear)
- Better local minima (barren plateaus possible)
- Universal superiority (depends on task structure)

---

## Limitations of Existing Models

### QTCN Limitations

The original Quantum Temporal Convolutional Network (HQTCN) fails to utilize VQC's periodic advantages.

**File Analyzed**: `models/HQTCN2_EEG.py`

#### Issue 1: No Frequency Extraction

```python
# Original QTCN: Raw time-domain input
window = x[:, :, indices].reshape(batch_size, -1)
reduced_window = self.fc(window)  # FC layer loses frequency info
```

**Problem**: Raw time-domain values fed to VQC without frequency analysis.

#### Issue 2: Generic Encoding

```python
# Original QTCN: Standard angle embedding
qml.AngleEmbedding(features, wires=wires, rotation='Y')
```

**Problem**: No frequency matching - VQC's spectrum not aligned with data.

#### Issue 3: Mean Aggregation

```python
# Original QTCN: Simple averaging
output = torch.mean(torch.stack(output, dim=1), dim=1)
```

**Problem**: `torch.mean()` is a low-pass filter that destroys high-frequency periodic information.

#### Summary: QTCN Periodic Advantage

| Component | Utilizes Periodic Advantage? |
|-----------|------------------------------|
| Input Processing | No (raw time-domain) |
| Encoding | No (generic RY) |
| Aggregation | No (mean destroys periodicity) |
| **Overall** | **Not Utilized** |

---

### QLSTM Limitations

The original Quantum LSTM has a **critical flaw**: sigmoid/tanh activations completely destroy VQC's periodic structure.

**File Analyzed**: `models/QLSTM_v0.py`

#### The Critical Problem: Sigmoid After VQC

```python
# Original QLSTM gates
i_t = torch.sigmoid(self.input_gate(combined))   # DESTROYS PERIODICITY
f_t = torch.sigmoid(self.forget_gate(combined))  # DESTROYS PERIODICITY
g_t = torch.tanh(self.cell_gate(combined))       # DESTROYS PERIODICITY
o_t = torch.sigmoid(self.output_gate(combined))  # DESTROYS PERIODICITY
```

#### Why This Is Catastrophic

VQC outputs a Fourier series (periodic):
$$f_{VQC}(x) = \sum_{\omega} c_\omega e^{i\omega x}$$

Sigmoid transforms it (destroys periodicity):
$$\sigma(f_{VQC}(x)) = \frac{1}{1 + e^{-f_{VQC}(x)}}$$

| Property | VQC Output | After Sigmoid |
|----------|------------|---------------|
| Range | Oscillating | Bounded (0,1) |
| Periodicity | **Periodic** | **Not periodic** |
| Frequency content | Rich | Destroyed |

#### Summary: QLSTM Periodic Advantage

| Component | Utilizes Periodic Advantage? |
|-----------|------------------------------|
| Input Processing | No (raw values) |
| Encoding | No (generic RY) |
| Gate Activation | **Destroys** (sigmoid/tanh) |
| Memory | No (classical cell state) |
| **Overall** | **Actively Destroyed** |

---

## Our Solutions

### Design Principles

1. **FFT Preprocessing**: Extract frequency content (lossless)
2. **Frequency-Matched Encoding**: RY + learnable freq_scale × RX
3. **Preserve Periodicity**: No sigmoid/tanh after VQC
4. **Minimal Classical Computation**: Avoid overshadowing quantum advantage

---

### FourierQTCN

**File**: `models/FourierQTCN_EEG.py`

#### Architecture

```
Input EEG [batch, channels, time]
    │
    ▼
┌─────────────────────────────────┐
│  Sliding Window Extraction      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  FFT Frequency Extraction       │  ← LOSSLESS
│  magnitude + phase              │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Linear Projection              │  ← Minimal classical
│  (fft_features → n_qubits)      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Frequency-Matched Encoding     │  ← KEY INNOVATION
│  RY(x) + RX(freq_scale × x)     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Quantum Conv-Pool Layers       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Learnable Weighted Aggregation │  ← Not mean()
└─────────────────────────────────┘
    │
    ▼
Output Prediction
```

#### Key Innovations

| Component | Original QTCN | FourierQTCN |
|-----------|---------------|-------------|
| Preprocessing | FC layer | FFT (lossless) |
| Encoding | RY only | RY + freq_scale × RX |
| Frequency Matching | None | Learnable per qubit |
| Aggregation | mean() | Learnable weights |

#### Usage

```python
from models.FourierQTCN_EEG import FourierQTCN

model = FourierQTCN(
    n_qubits=8,
    circuit_depth=2,
    input_dim=(batch, channels, time),
    kernel_size=12,
    dilation=3,
    n_frequencies=8
)
```

---

### FourierQLSTM

**File**: `models/FourierQLSTM.py`

#### Architecture

```
Input x_t [batch, window_size]
    │
    ▼
┌─────────────────────────────────┐
│  FFT Preprocessing              │  ← Always applied
│  magnitude + phase              │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Frequency-Matched VQC Gates    │
│  RY(x) + RX(freq_scale × x)     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Rescaled Gating                │  ← NOT sigmoid!
│  gate = (VQC + 1) / 2           │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Frequency-Domain Cell State    │  ← (magnitude, phase)
│  c_mag, c_phase                 │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Periodic Output                │
│  h = o × mag × cos(phase × π)   │
└─────────────────────────────────┘
```

#### Key Innovations

| Component | Original QLSTM | FourierQLSTM |
|-----------|----------------|--------------|
| Preprocessing | None | FFT |
| Encoding | RY only | RY + freq_scale × RX |
| Gate Activation | sigmoid/tanh | (VQC + 1) / 2 |
| Cell State | Single tensor | (magnitude, phase) |
| Periodicity | **Destroyed** | **Preserved** |

#### Usage

```python
from models.FourierQLSTM import FourierQLSTM

model = FourierQLSTM(
    input_size=1,
    hidden_size=4,
    n_qubits=6,
    vqc_depth=2,
    window_size=8
)

outputs, (c_mag, c_phase) = model(x_sequence)
```

---

### SIREN Baselines (Classical Fourier Challenger)

SIREN (Sitzmann et al., NeurIPS 2020) uses `sin(w0 * (Wx + b))` as activation, making it a **direct classical analogue** of VQC's Fourier series. Both methods fundamentally compute Fourier series with trainable coefficients. Adding SIREN as a baseline isolates: **does quantum Fourier give advantage over classical Fourier?**

#### SIREN-LSTM

**File**: `models/SIREN_LSTM.py`

Mirrors FourierQLSTM exactly — same FFT preprocessing, same frequency-domain memory, same rescaled gating — with 4 SIRENGate instances replacing 4 FrequencyMatchedVQC instances.

```python
from models.SIREN_LSTM import SIREN_LSTM

model = SIREN_LSTM(
    input_size=1,
    hidden_size=4,
    n_qubits=6,       # SIREN input dim (matched to VQC)
    vqc_depth=2,       # SIREN depth (matched to VQC)
    window_size=8,
    w0=30.0,           # SIREN frequency (learnable)
    learnable_w0=True
)
```

#### SIREN-TCN

**File**: `models/SIREN_TCN_EEG.py`

Mirrors FourierQTCN exactly — same sliding window, same FFT extraction, same weighted aggregation — with SIRENBlock replacing quantum circuit.

```python
from models.SIREN_TCN_EEG import SIREN_TCN

model = SIREN_TCN(
    siren_dim=8,           # Equivalent to n_qubits
    n_siren_layers=2,      # Equivalent to circuit_depth
    input_dim=(batch, channels, time),
    kernel_size=12,
    dilation=3,
    w0=30.0,
    learnable_w0=True
)
```

#### Key Comparison: VQC vs SIREN

| Aspect | VQC (FourierQ*) | SIREN |
|--------|-----------------|-------|
| Core computation | Quantum Fourier series | Classical Fourier series |
| Frequency control | Encoding Hamiltonian + freq_scale | Learnable w0 |
| Batch processing | Per-sample (circuit) | Full batch (native) |
| Parameter efficiency | High (exponential Hilbert space) | Standard (polynomial) |
| Training speed | Slower (simulation) | Faster (GPU-native) |

See `docs/VQC_vs_SIREN_Comparison.md` for detailed theoretical and architectural analysis.

---

## Experiments

### PhysioNet EEG Classification

#### Dataset
- **Source**: PhysioNet Motor Imagery Dataset
- **Channels**: 64 EEG channels
- **Sampling Rate**: 160 Hz (downsampled to 80 Hz)
- **Task**: Binary classification (left vs right motor imagery)

#### EEG Frequency Bands

| Band | Frequency | Relevance |
|------|-----------|-----------|
| Delta | 0.5-4 Hz | Deep sleep |
| Theta | 4-8 Hz | Drowsiness |
| Alpha | 8-13 Hz | Relaxed state |
| Beta | 13-30 Hz | Active thinking, motor planning |
| Gamma | 30-100 Hz | Higher cognition |

#### Why EEG is Ideal for VQC

1. **Intrinsically periodic**: Brain oscillations are rhythmic
2. **Multiple frequency bands**: Rich spectral content
3. **Temporal structure**: Time-series with periodic components

#### Experiment Configuration

```bash
cd VQC-PeriodicData
python models/FourierQTCN_EEG.py \
    --n-qubits=8 \
    --circuit-depth=2 \
    --freq=80 \
    --n-sample=50 \
    --num-epochs=50 \
    --kernel-size=12 \
    --dilation=3 \
    --seed=2025
```

#### Expected Comparisons

| Model | Periodic Advantage | Expected AUC |
|-------|-------------------|--------------|
| Classical TCN | None | Baseline |
| Original QTCN | Not utilized | ~Baseline |
| SIREN-TCN | Classical Fourier | >Baseline |
| FourierQTCN | **Quantum Fourier** | >Baseline |

#### Metrics
- **Primary**: ROC-AUC
- **Secondary**: Accuracy, Loss convergence
- **Analysis**: Learned freq_scale values per qubit

---

### NARMA Time-Series Prediction

#### Dataset
- **Source**: Generated via `data/narma_generator.py`
- **Task**: NARMA-n prediction (n = 5, 10, 30)
- **Type**: Nonlinear autoregressive moving average

#### NARMA Equation

$$y(t) = 0.3 \cdot y(t-1) + 0.05 \cdot y(t-1) \sum_{i=0}^{n-1} y(t-i-1) + 1.5 \cdot u(t-n) \cdot u(t-1) + 0.1$$

#### Why NARMA Tests Periodic Learning

1. **Nonlinear dynamics**: Tests function approximation
2. **Memory requirements**: Tests temporal modeling
3. **Quasi-periodic patterns**: Exhibits oscillatory behavior

#### Experiment Configuration

```python
from data.narma_generator import get_narma_for_fourier_qlstm
from models.FourierQLSTM import FourierQLSTM

# Data
train_loader, val_loader, test_loader, seq_len = get_narma_for_fourier_qlstm(
    n_samples=1000,
    order=10,
    seq_len=16,
    batch_size=32,
    seed=2025
)

# Model
model = FourierQLSTM(
    input_size=1,
    hidden_size=4,
    n_qubits=6,
    vqc_depth=2,
    window_size=8
)
```

#### Expected Comparisons

| Model | Periodic Advantage | Expected MSE |
|-------|-------------------|--------------|
| Classical LSTM | None | Baseline |
| Original QLSTM | Destroyed | ~Baseline or worse |
| SIREN-LSTM | Classical Fourier | <Baseline |
| FourierQLSTM | **Quantum Fourier** | <Baseline |

#### Metrics
- **Primary**: MSE (Mean Squared Error)
- **Secondary**: MAE, convergence speed
- **Analysis**: Learned freq_scale, phase evolution

---

## Installation & Usage

### Requirements

```bash
# Core dependencies
pip install torch pennylane numpy scipy scikit-learn

# For EEG experiments
pip install mne

# For visualization
pip install matplotlib pandas tqdm
```

### Quick Start

```python
# 1. Generate NARMA data
from data.narma_generator import get_narma_data
x, y = get_narma_data(n_0=10, seq_len=8, n_samples=500)

# 2. Create FourierQLSTM model
from models.FourierQLSTM import FourierQLSTM
model = FourierQLSTM(
    input_size=1,
    hidden_size=4,
    n_qubits=6,
    vqc_depth=2,
    window_size=8
)

# 3. Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    outputs, _ = model(x)
    loss = F.mse_loss(outputs[:, -1, :], y)
    loss.backward()
    optimizer.step()
```

### Running Experiments

```bash
cd VQC-PeriodicData

# PhysioNet EEG with FourierQTCN
python models/FourierQTCN_EEG.py --n-qubits=8 --num-epochs=50

# PhysioNet EEG with SIREN-TCN (classical Fourier baseline)
python models/SIREN_TCN_EEG.py --siren-dim=8 --n-siren-layers=2 --num-epochs=50

# NARMA with FourierQLSTM
python models/FourierQLSTM.py

# NARMA with SIREN-LSTM (classical Fourier baseline)
python models/SIREN_LSTM.py --n-qubits=6 --vqc-depth=2
```

---

## Project Structure

```
VQC-PeriodicData/
│
├── README.md                                  # This file
│
├── data/                                      # Data generators
│   ├── __init__.py
│   └── narma_generator.py                     # NARMA dataset generator
│
├── models/                                    # Model implementations
│   ├── FourierQLSTM.py                        # Fourier-QLSTM (periodic-aware)
│   ├── FourierQTCN_EEG.py                     # Fourier-QTCN (periodic-aware)
│   ├── SIREN_LSTM.py                          # SIREN-LSTM (classical Fourier baseline)
│   ├── SIREN_TCN_EEG.py                       # SIREN-TCN (classical Fourier baseline)
│   ├── PeriodicAwareQTCN_EEG.py               # Alternative periodic QTCN
│   ├── QLSTM_v0.py                            # Original QLSTM (for comparison)
│   ├── HQTCN2_EEG.py                          # Original QTCN (for comparison)
│   ├── HQTCN2_NARMA.py                        # Original QTCN for NARMA
│   └── Load_PhysioNet_EEG.py                  # EEG data loader
│
└── docs/                                      # Documentation & Analysis
    ├── FourierQLSTM_README.md                 # Fourier-QLSTM documentation
    ├── FourierQTCN_README.md                  # Fourier-QTCN documentation
    ├── QLSTM_Periodic_Advantage_Analysis.md   # Why original QLSTM fails
    ├── QTCN_Periodic_Advantage_Analysis.md    # Why original QTCN fails
    ├── VQC_Periodic_Data_Encoding_Guide.md    # Encoding strategies
    ├── VQC_Periodic_Data_Performance_Analysis.md
    ├── VQC_vs_ReLU_Periodic_Data_Comparison.md
    ├── VQC_Universal_Extrapolation_Analysis.md
    ├── VQC_vs_Snake_Practical_Comparison.md
    ├── VQC_vs_SIREN_Comparison.md             # VQC vs SIREN (classical Fourier)
    └── Hybrid_VQC_Snake_Architecture_Analysis.md
```

---

## References

### Primary Sources

1. **Schuld, M., Sweke, R., & Meyer, J. J.** (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A*, 103(3), 032430.
   - **Key insight**: VQC computes Fourier series; frequency spectrum determined by encoding

2. **Ziyin, L., Hartwig, T., & Ueda, M.** (2020). Neural networks fail to learn periodic functions and how to fix it. *Advances in Neural Information Processing Systems*, 33.
   - **Key insight**: Snake activation for periodic learning; VQC comparison

3. **Pérez-Salinas, A., et al.** (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.
   - **Key insight**: Repeated encoding expands frequency spectrum

4. **Sitzmann, V., Martel, J. N., Bergman, A. W., Lindell, D. B., & Wetzstein, G.** (2020). Implicit Neural Representations with Periodic Activation Functions. *Advances in Neural Information Processing Systems*, 33.
   - **Key insight**: sin(w0·(Wx+b)) as universal periodic activation; classical Fourier baseline for VQC

### Additional Resources

- PhysioNet Motor Imagery Dataset: https://physionet.org/content/eegmmidb/
- PennyLane Documentation: https://pennylane.ai/
- NARMA Benchmark: Atiya & Parlos (2000)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{vqc-periodic-data,
  title={VQC-PeriodicData: Leveraging Quantum Periodic Advantages for Time-Series},
  author={[Your Name]},
  year={2026},
  howpublished={\url{https://github.com/[your-repo]}}
}
```

---

## License

[Specify your license]

---

## Acknowledgments

This research utilizes:
- PennyLane quantum computing framework
- PhysioNet EEG datasets
- NERSC computing resources

---

*Last Updated: February 2026*
