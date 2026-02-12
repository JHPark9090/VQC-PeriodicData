# VQC vs SIREN: The Definitive Classical Fourier Baseline

## Why SIREN Is the Key Comparison for VQC's Periodic Advantage

**Author**: Analysis based on Sitzmann et al. (NeurIPS 2020) and Schuld et al. (Physical Review A, 2021)

**Date**: February 2026

---

## Executive Summary

SIREN (Sinusoidal Representation Networks) uses `sin(w0 * (Wx + b))` as its activation function, making it a **direct classical analogue of VQC's Fourier series computation**. Both methods fundamentally compute Fourier series with trainable coefficients. This makes SIREN the strongest and most scientifically relevant classical baseline for testing quantum Fourier advantage.

| Property | VQC | SIREN |
|----------|-----|-------|
| Core computation | Fourier series | Fourier series |
| Output structure | Σ c_ω e^{iωx} | Composition of sin(w0·(Wx+b)) |
| Frequency control | Encoding Hamiltonian | Learnable w0 |
| Coefficient control | Variational parameters | Network weights |
| Output range | [-1, 1] (PauliZ) | Unbounded (or tanh-bounded) |
| Batch processing | Per-sample (circuit) | Full batch (matrix ops) |

**Key question this comparison answers**: Does the quantum mechanism for generating Fourier series provide any advantage over the classical mechanism (SIREN)?

---

## Table of Contents

1. [Mathematical Comparison](#1-mathematical-comparison)
2. [Architectural Comparison](#2-architectural-comparison)
3. [Parameter Count Comparison](#3-parameter-count-comparison)
4. [What Each Outcome Means](#4-what-each-outcome-means)
5. [Training Efficiency Comparison](#5-training-efficiency-comparison)
6. [Experimental Design](#6-experimental-design)
7. [References](#7-references)

---

## 1. Mathematical Comparison

### 1.1 VQC as Fourier Series

From Schuld et al. (2021), a VQC with angle encoding computes:

$$f_{VQC}(x) = \sum_{\omega \in \Omega} c_\omega e^{i\omega x}$$

where:
- **Ω** = frequency spectrum determined by encoding Hamiltonian
- **c_ω** = trainable Fourier coefficients determined by variational parameters

The frequency spectrum is fixed by the circuit design (encoding gates), while coefficients are optimized during training.

### 1.2 SIREN as Fourier Series

A single SIREN layer computes:

$$h(x) = \sin(\omega_0 (Wx + b))$$

A multi-layer SIREN computes a composition of sinusoids:

$$f_{SIREN}(x) = W_L \cdot \sin(\omega_0^{(L-1)} (W_{L-1} \cdots \sin(\omega_0^{(1)} (W_1 x + b_1)) \cdots + b_{L-1})) + b_L$$

From the product-to-sum identities for trigonometric functions, this composition generates a sum of sinusoids at various frequencies — i.e., a Fourier series:

$$f_{SIREN}(x) = \sum_{k} a_k \sin(\omega_k x + \phi_k)$$

### 1.3 Key Mathematical Differences

| Aspect | VQC | SIREN |
|--------|-----|-------|
| **Frequency spectrum** | Discrete, integer-valued, bounded by ±r (r = encoding repetitions) | Continuous, determined by w0 and weight products |
| **Coefficient generation** | Quantum interference of amplitudes | Classical matrix multiplication |
| **Entanglement** | Creates correlations between frequency modes | No quantum correlations |
| **Expressivity scaling** | Frequencies scale with qubit count/encoding reps | Frequencies scale with depth and w0 |
| **Frequency resolution** | Fixed by Hamiltonian eigenvalues | Adaptive via learnable w0 |

### 1.4 The Critical Difference: Entanglement

VQC's unique property is **quantum entanglement between qubits**, which creates correlations between Fourier coefficients that cannot be efficiently represented classically for large systems. Specifically:

- **VQC**: Fourier coefficients c_ω are generated jointly through entangling gates, creating complex dependencies
- **SIREN**: Coefficients are generated through matrix multiplication chains, with dependencies limited by network width

For small qubit counts (4-16, typical in current experiments), this entanglement advantage is not expected to manifest because classical networks can represent the same coefficient correlations with sufficient width.

---

## 2. Architectural Comparison

### 2.1 LSTM Comparison: FourierQLSTM vs SIREN-LSTM

Both models share identical architecture except for the gate computation:

```
FourierQLSTM                          SIREN-LSTM
============                          ==========

Input x [batch, window]               Input x [batch, window]
    |                                     |
    v                                     v
FFT Preprocessing                     FFT Preprocessing         <- IDENTICAL
(magnitude + phase)                   (magnitude + phase)
    |                                     |
    v                                     v
Linear Projection                     Linear Projection          <- IDENTICAL
(fft_features -> n_qubits)            (fft_features -> siren_dim)
    |                                     |
    v                                     v
tanh * pi normalization               tanh * pi normalization    <- IDENTICAL
    |                                     |
    v                                     v
+-----------------------------+    +-----------------------------+
| 4x FrequencyMatchedVQC     |    | 4x SIRENGate               |  <- DIFFERENT
|  RY(x) + RX(freq_scale*x)  |    |  sin(w0 * (Wx + b))        |
|  Entangling CNOT layers     |    |  Multi-layer SIREN          |
|  Variational RY rotations   |    |  tanh output bounding       |
|  PauliZ measurement         |    |                             |
+-----------------------------+    +-----------------------------+
    |                                     |
    v                                     v
Rescaled gating: (out+1)/2           Rescaled gating: (out+1)/2  <- IDENTICAL
    |                                     |
    v                                     v
Frequency-domain cell state           Frequency-domain cell state <- IDENTICAL
(c_mag, c_phase)                      (c_mag, c_phase)
    |                                     |
    v                                     v
Output projection                     Output projection          <- IDENTICAL
```

### 2.2 TCN Comparison: FourierQTCN vs SIREN-TCN

```
FourierQTCN                           SIREN-TCN
===========                           =========

Input EEG [batch, ch, time]           Input EEG [batch, ch, time]
    |                                     |
    v                                     v
Sliding Window Extraction             Sliding Window Extraction   <- IDENTICAL
    |                                     |
    v                                     v
FFT Frequency Extraction              FFT Frequency Extraction    <- IDENTICAL
(magnitude + phase)                   (magnitude + phase)
    |                                     |
    v                                     v
Linear Projection                     Linear Projection           <- IDENTICAL
(fft_dim -> n_qubits)                 (fft_dim -> siren_dim)
    |                                     |
    v                                     v
tanh * pi normalization               tanh * pi normalization     <- IDENTICAL
    |                                     |
    v                                     v
+-----------------------------+    +-----------------------------+
| Quantum Conv-Pool Circuit   |    | SIREN Block                 |  <- DIFFERENT
|  Freq-matched RY+RX enc.   |    |  sin(w0 * (Wx + b)) layers  |
|  U3 + Ising gate conv.     |    |  Progressive dim reduction   |
|  Mid-circuit measurement    |    |  Linear output              |
|  PauliZ final measurement  |    |                             |
+-----------------------------+    +-----------------------------+
    |                                     |
    v                                     v
Learnable Weighted Aggregation        Learnable Weighted Aggregation  <- IDENTICAL
    |                                     |
    v                                     v
Output                                Output
```

### 2.3 What This Isolation Tests

By keeping everything identical except the Fourier-generating mechanism:

- **If VQC > SIREN**: Quantum Fourier mechanism provides genuine advantage (entanglement, interference)
- **If VQC ≈ SIREN**: The advantage comes from the Fourier structure, not from quantum mechanics
- **If VQC < SIREN**: Classical Fourier (SIREN) is more efficient for these problem sizes

---

## 3. Parameter Count Comparison

### 3.1 LSTM Gate Parameter Count

**FrequencyMatchedVQC** (per gate):
- freq_scale: n_qubits
- var_params: vqc_depth × n_qubits
- **Total**: n_qubits × (1 + vqc_depth)

Example (n_qubits=6, vqc_depth=2): 6 × 3 = **18 parameters**

**SIRENGate** (per gate, hidden_dim=64, n_layers=2):
- Layer 1: (n_qubits × 64 + 64) + 1 (w0) = **449**
- Layer 2: (64 × 64 + 64) + 1 (w0) = **4,161**
- Output: 64 × output_dim + output_dim

Example (input=6, output=4, hidden=64, 2 layers): ~**4,638 parameters**

### 3.2 Interpretation

SIREN has significantly more parameters than VQC for the same architectural role. This is intentional and reflects a key aspect of the comparison:

| Aspect | VQC | SIREN |
|--------|-----|-------|
| Parameters per gate | ~18 | ~4,638 |
| Expressivity source | Quantum state space (2^n) | Parameter count |
| Parameter efficiency | High (exponential state space) | Standard (polynomial) |

**If VQC matches SIREN performance with fewer parameters**, this suggests the quantum state space provides a form of parameter efficiency — the exponential Hilbert space encodes more information per parameter.

### 3.3 Fair Comparison Strategy

To make the comparison fair:
1. **Match architecture** (identical everything except VQC/SIREN) — this is what we do
2. **Report parameter counts** — let the reader assess efficiency
3. **Also compare at matched parameter count** — reduce SIREN hidden dim to match VQC params

---

## 4. What Each Outcome Means

### 4.1 Outcome A: VQC > SIREN

**Interpretation**: Quantum Fourier generation provides genuine advantage.

**Possible explanations**:
- Entanglement creates beneficial correlations between Fourier coefficients
- Quantum interference enables more efficient coefficient optimization
- The quantum feature space captures relevant patterns that SIREN misses

**Significance**: Strong evidence for practical quantum advantage in Fourier-based tasks.

### 4.2 Outcome B: VQC ≈ SIREN

**Interpretation**: The advantage comes from Fourier structure, not quantum mechanics.

**Possible explanations**:
- Both methods compute equivalent Fourier series at these scales
- The entanglement advantage doesn't manifest for small qubit counts
- The Fourier inductive bias (shared by both) is the key factor

**Significance**: VQC advantage over ReLU/tanh is explained by periodic structure, not by quantum effects. SIREN is equally effective as a classical alternative.

### 4.3 Outcome C: VQC < SIREN

**Interpretation**: Classical Fourier (SIREN) is more efficient.

**Possible explanations**:
- SIREN's continuous frequency control (w0) is more flexible than VQC's discrete spectrum
- SIREN's batch processing enables better optimization (more gradient updates per wall-clock time)
- VQC's barren plateaus or measurement noise degrade performance
- SIREN has more parameters, providing better approximation

**Significance**: The quantum Fourier mechanism incurs overhead without benefit at current scales.

---

## 5. Training Efficiency Comparison

### 5.1 Computational Cost

| Aspect | VQC | SIREN | Ratio |
|--------|-----|-------|-------|
| **Forward pass (per sample)** | O(n_qubits × depth) circuit simulation | O(width × depth) matmul | SIREN ~10-100x faster |
| **Forward pass (batch)** | N × single-sample | Single batched matmul | SIREN ~N× faster |
| **Backward pass** | Parameter-shift or backprop through sim | Standard backprop | SIREN ~2× faster |
| **GPU utilization** | Limited (simulation) | Full | SIREN much better |

### 5.2 The Batch Processing Advantage

The most significant practical difference:

```python
# VQC: Must loop over samples (PennyLane limitation for most backends)
for i in range(batch_size):
    out = self.circuit(x[i])   # One sample at a time

# SIREN: Native batch processing
out = torch.sin(self.w0 * self.linear(x))  # Full batch at once
```

This means SIREN is **orders of magnitude faster** per epoch for the same architecture.

### 5.3 What This Means for the Comparison

If VQC and SIREN achieve similar final performance, the practical recommendation is clear: **use SIREN** for its speed advantage. VQC would only be preferred if it achieves meaningfully better accuracy that justifies the computational overhead.

---

## 6. Experimental Design

### 6.1 NARMA Prediction (LSTM)

| Setting | FourierQLSTM | SIREN-LSTM |
|---------|-------------|------------|
| Data | NARMA-10 | NARMA-10 |
| Window size | 8 | 8 |
| Hidden size | 4 | 4 |
| Input dim | 6 (qubits) | 6 (siren_dim) |
| Depth | 2 (vqc_depth) | 2 (n_layers) |
| FFT preprocessing | Yes | Yes |
| Frequency-domain memory | Yes | Yes |
| Gating | (out+1)/2 | (out+1)/2 |
| Metric | MSE | MSE |

### 6.2 EEG Classification (TCN)

| Setting | FourierQTCN | SIREN-TCN |
|---------|-------------|-----------|
| Data | PhysioNet EEG | PhysioNet EEG |
| Kernel size | 12 | 12 |
| Dilation | 3 | 3 |
| Input dim | 8 (qubits) | 8 (siren_dim) |
| Depth | 2 (circuit_depth) | 2 (n_layers) |
| FFT preprocessing | Yes | Yes |
| Aggregation | Learnable weights | Learnable weights |
| Loss | BCEWithLogitsLoss | BCEWithLogitsLoss |
| Metric | ROC-AUC | ROC-AUC |

### 6.3 Running the Experiments

```bash
cd VQC-PeriodicData

# NARMA with FourierQLSTM
python models/FourierQLSTM.py

# NARMA with SIREN-LSTM (matched config)
python models/SIREN_LSTM.py --n-qubits=6 --vqc-depth=2 --hidden-size=4 --window-size=8

# EEG with FourierQTCN
python models/FourierQTCN_EEG.py --n-qubits=8 --circuit-depth=2

# EEG with SIREN-TCN (matched config)
python models/SIREN_TCN_EEG.py --siren-dim=8 --n-siren-layers=2
```

---

## 7. References

1. **Sitzmann, V., Martel, J. N., Bergman, A. W., Lindell, D. B., & Wetzstein, G.** (2020). Implicit Neural Representations with Periodic Activation Functions. *Advances in Neural Information Processing Systems*, 33.
   - Key insight: sin(w0·(Wx+b)) as universal periodic activation

2. **Schuld, M., Sweke, R., & Meyer, J. J.** (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A*, 103(3), 032430.
   - Key insight: VQC computes Fourier series; frequency spectrum determined by encoding

3. **Ziyin, L., Hartwig, T., & Ueda, M.** (2020). Neural networks fail to learn periodic functions and how to fix it. *Advances in Neural Information Processing Systems*, 33.
   - Key insight: Standard activations (ReLU, tanh) fail on periodic data; Snake activation helps

4. **Pérez-Salinas, A., et al.** (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.
   - Key insight: Repeated encoding expands VQC frequency spectrum

---

*Document prepared for the VQC-PeriodicData project comparing quantum and classical Fourier methods for time-series analysis.*
