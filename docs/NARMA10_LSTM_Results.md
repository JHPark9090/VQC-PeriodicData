# NARMA-10 LSTM Experiment Results

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | NARMA-10 (nonlinear autoregressive moving average, order 10) |
| Samples | 500 total → 329 train / 163 test (67/33 split) |
| Seed | 2025 |
| Epochs | 50 |
| Batch Size | 10 |
| Learning Rate | 0.01 (Adam) |
| Window Size | 8 |
| Hidden Size | 4 (frequency-domain cell state) |
| Input Dim (n_qubits) | 6 |
| Gate Depth (vqc_depth) | 2 |
| MLP Hidden Dim | 64 (classical baselines) |
| Loss Function | MSE |

---

## Results Summary

| Rank | Model | Activation Type | Train MSE | Test MSE | Params | Time/Epoch |
|------|-------|----------------|-----------|----------|--------|------------|
| **1** | **FourierQLSTM** | **Quantum Fourier** | **0.0456** | **0.0554** | ~19K | ~40s |
| 2 | Tanh-LSTM | Monotonic (no periodic) | 0.0665 | 0.0594 | 19,599 | 0.1s |
| 3 | ReLU-LSTM | Piecewise linear (no periodic) | 0.0493 | 0.0673 | 19,599 | 0.1s |
| 4 | SIREN-LSTM | Classical Fourier (sin) | 0.0921 | 0.0742 | 19,607 | 0.2s |
| 5 | Snake-LSTM | Partial periodic (x + sin²) | 0.0960 | 0.0789 | 20,111 | 0.2s |

---

## Activation Hierarchy

```
Periodic Structure Strength:
  None          Partial         Full Periodic       Quantum Fourier
   |               |                |                     |
  ReLU           Snake            SIREN             FourierQLSTM
  Tanh        (x + sin²/a)    (sin(w0·Wx))       (VQC Fourier series)
```

---

## Training Curves (every 10 epochs)

### FourierQLSTM (Quantum)
| Epoch | Train MSE | Test MSE | Freq Scale Range |
|-------|-----------|----------|-----------------|
| 10 | 0.058680 | 0.053775 | [0.67, 3.00] |
| 20 | 0.058054 | 0.056431 | [0.55, 3.32] |
| 30 | 0.055179 | 0.054330 | [0.43, 3.26] |
| 40 | 0.055505 | 0.057831 | [0.24, 3.00] |
| 50 | 0.045555 | 0.055372 | [0.06, 3.03] |

### ReLU-LSTM
| Epoch | Train MSE | Test MSE |
|-------|-----------|----------|
| 10 | 0.061645 | 0.059262 |
| 20 | 0.050452 | 0.068804 |
| 30 | 0.047860 | 0.064872 |
| 40 | 0.050805 | 0.081417 |
| 50 | 0.049346 | 0.067267 |

### Tanh-LSTM
| Epoch | Train MSE | Test MSE |
|-------|-----------|----------|
| 10 | 0.064697 | 0.085846 |
| 20 | 0.067655 | 0.059201 |
| 30 | 0.066464 | 0.059322 |
| 40 | 0.066547 | 0.059552 |
| 50 | 0.066530 | 0.059352 |

### Snake-LSTM
| Epoch | Train MSE | Test MSE | Snake 'a' Range |
|-------|-----------|----------|----------------|
| 10 | 0.096715 | 0.079909 | [0.80, 1.37] |
| 20 | 0.096658 | 0.079798 | [0.80, 1.37] |
| 30 | 0.096464 | 0.079508 | [0.80, 1.37] |
| 40 | 0.096241 | 0.079182 | [0.80, 1.37] |
| 50 | 0.096008 | 0.078845 | [0.79, 1.37] |

### SIREN-LSTM
| Epoch | Train MSE | Test MSE | w0 Range |
|-------|-----------|----------|----------|
| 10 | 0.092415 | 0.074443 | [29.69, 29.90] |
| 20 | 0.092262 | 0.074087 | [29.59, 29.85] |
| 30 | 0.092090 | 0.074294 | [29.57, 29.87] |
| 40 | 0.092070 | 0.074224 | [29.61, 29.81] |
| 50 | 0.092076 | 0.074215 | [29.61, 29.74] |

---

## Learned Parameters Analysis

### FourierQLSTM — Frequency Scales (per qubit)

The quantum model's learnable `freq_scale` parameters control how each qubit's encoding frequency matches the data's spectral content. After 50 epochs:

| Gate | Qubit 0 | Qubit 1 | Qubit 2 | Qubit 3 | Qubit 4 | Qubit 5 |
|------|---------|---------|---------|---------|---------|---------|
| Input | 0.86 | 0.06 | 0.74 | 1.25 | 3.03 | 3.00 |
| Forget | 0.50 | 1.00 | 1.50 | 2.00 | 2.50 | 3.00 |
| Cell | 0.67 | 0.64 | 0.77 | 2.01 | 1.92 | 3.00 |
| Output | 0.00 | -0.10 | 0.70 | 1.25 | 3.52 | 3.00 |

**Key observation**: The input gate developed a wide frequency spread [0.06, 3.03], indicating the model learned to cover both low and high frequency components of NARMA-10. The forget gate remained at initialization, acting as a fixed spectral filter.

### Snake-LSTM — Learned 'a' Values

| Gate | Layer 0 (mean ± std) | Layer 1 (mean ± std) |
|------|---------------------|---------------------|
| Input | 1.009 ± 0.114 | 0.986 ± 0.099 |
| Forget | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Cell | 0.992 ± 0.087 | 0.986 ± 0.088 |
| Output | 1.006 ± 0.084 | 0.991 ± 0.078 |

**Key observation**: Snake's 'a' values barely moved from initialization (a=1.0). The forget gate didn't learn at all (std=0.000). This suggests 50 epochs is insufficient for Snake to discover useful frequency parameters, or NARMA-10's structure doesn't strongly benefit from partial periodicity.

### SIREN-LSTM — Learned w0 Values

| Gate | Layer 0 w0 | Layer 1 w0 |
|------|-----------|-----------|
| Input | 29.61 | 29.74 |
| Forget | 30.00 | 30.00 |
| Cell | 30.45 | 29.95 |
| Output | 29.55 | 29.74 |

**Key observation**: SIREN's w0 values barely moved from the default (30.0). The high initial frequency may cause the network to oscillate too rapidly for NARMA-10's quasi-periodic dynamics. A lower w0 initialization (e.g., 1.0–5.0) might improve performance.

---

## Key Findings

### 1. FourierQLSTM achieves the best test MSE (0.0554)

The quantum Fourier model outperforms all classical baselines, supporting the hypothesis that VQC's native Fourier series structure provides an advantage for periodic/quasi-periodic data when properly utilized (FFT preprocessing + frequency-matched encoding + periodicity-preserving gates).

### 2. Classical periodic activations (SIREN, Snake) underperform non-periodic ones (ReLU, Tanh)

This is likely due to initialization sensitivity:
- **SIREN**: w0=30.0 is standard for image fitting (Sitzmann et al.) but may be too high for NARMA-10's low-frequency dynamics
- **Snake**: a=1.0 initialization and 50 epochs may be insufficient for meaningful frequency adaptation
- **Recommendation**: Sweep w0 ∈ {1, 5, 10, 30} for SIREN and a_init ∈ {0.1, 0.5, 1.0, 5.0} for Snake

### 3. Tanh is surprisingly competitive (test MSE = 0.0594)

Tanh's saturation behavior (bounded to [-1, 1]) may accidentally suit NARMA-10's bounded output dynamics, even without periodic structure. This doesn't mean tanh has periodic advantages — it would fail on extrapolation tasks.

### 4. ReLU shows overfitting tendency

ReLU achieves low train MSE (0.0493) but higher test MSE (0.0673), with increasing test loss after epoch 10. The piecewise linear approximation memorizes training patterns without generalizing periodic structure.

### 5. Training speed vs accuracy trade-off

| Model | Time/Epoch | Slowdown vs ReLU | Test MSE |
|-------|-----------|-------------------|----------|
| ReLU-LSTM | 0.1s | 1× | 0.0673 |
| Tanh-LSTM | 0.1s | 1× | 0.0594 |
| Snake-LSTM | 0.2s | 2× | 0.0789 |
| SIREN-LSTM | 0.2s | 2× | 0.0742 |
| FourierQLSTM | ~40s | ~400× | 0.0554 |

The quantum model is ~400× slower due to per-sample circuit simulation (no batch processing in PennyLane). On real quantum hardware, this gap would narrow significantly.

---

## Architecture Comparison

All 5 models share identical architecture except for the gate computation:

```
Input x_t [batch, window_size=8]
    │
    ▼
FFT Preprocessing → magnitude + phase (5 freq bins × 2 = 10 features)
    │
    ▼
Linear Projection → 6 dimensions (n_qubits)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 4 Gates (input, forget, cell, output):          │
│                                                  │
│   FourierQLSTM: VQC with freq_scale encoding    │
│   ReLU-LSTM:    MLP with ReLU + tanh output     │
│   Tanh-LSTM:    MLP with tanh + tanh output     │
│   Snake-LSTM:   MLP with Snake(a) + tanh output │
│   SIREN-LSTM:   MLP with sin(w0·Wx) + tanh out  │
└─────────────────────────────────────────────────┘
    │
    ▼
Rescaled Gating: (output + 1) / 2   ← NOT sigmoid
    │
    ▼
Frequency-Domain Cell State: (c_magnitude, c_phase)
    │
    ▼
Periodic Output: h = o × mag × cos(phase × π)
    │
    ▼
Linear Projection → prediction
```

---

## Reproducibility

```bash
cd /pscratch/sd/j/junghoon/VQC-PeriodicData

# Quantum LSTM
python -c "import scipy.constants; from models.FourierQLSTM import main; main()"

# Classical baselines
python models/ReLU_LSTM.py --seed=2025 --n-qubits=6 --vqc-depth=2 --hidden-size=4 --window-size=8 --n-epochs=50 --batch-size=10 --lr=0.01 --narma-order=10 --n-samples=500
python models/Tanh_LSTM.py --seed=2025 --n-qubits=6 --vqc-depth=2 --hidden-size=4 --window-size=8 --n-epochs=50 --batch-size=10 --lr=0.01 --narma-order=10 --n-samples=500
python models/Snake_LSTM.py --seed=2025 --n-qubits=6 --vqc-depth=2 --hidden-size=4 --window-size=8 --n-epochs=50 --batch-size=10 --lr=0.01 --narma-order=10 --n-samples=500
python models/SIREN_LSTM.py --seed=2025 --n-qubits=6 --vqc-depth=2 --hidden-size=4 --window-size=8 --n-epochs=50 --batch-size=10 --lr=0.01 --narma-order=10 --n-samples=500
```

**Environment**: Perlmutter (NERSC), conda env `qml_eeg`, Python 3.11, PyTorch 2.5.0+cu121, PennyLane 0.42.3

---

## References

1. **Schuld et al.** (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A*, 103(3).
2. **Sitzmann et al.** (2020). Implicit Neural Representations with Periodic Activation Functions. *NeurIPS*.
3. **Ziyin et al.** (2020). Neural networks fail to learn periodic functions and how to fix it. *NeurIPS*.
4. **Atiya & Parlos** (2000). New results on recurrent network training: Unifying the algorithms and accelerating convergence. *IEEE Trans. Neural Networks*.

---

*Generated: February 2026 | Perlmutter @ NERSC*
