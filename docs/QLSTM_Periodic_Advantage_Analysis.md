# QLSTM Periodic Advantage Analysis

Analysis of whether `QLSTM_v0.py` utilizes VQC's periodic advantages for time-series prediction.

**File Analyzed**: `/pscratch/sd/j/junghoon/VQC-PeriodicData/QLSTM_v0.py`

**Date**: February 2026

---

## Executive Summary

**Does QLSTM_v0.py utilize VQC's periodic advantages?**

**NO** - The model treats VQCs as generic function approximators (drop-in replacements for linear layers), not as Fourier series generators. Furthermore, it **actively destroys** any periodic structure through post-VQC sigmoid/tanh activations.

---

## Table of Contents

1. [Model Architecture Overview](#model-architecture-overview)
2. [VQC Component Analysis](#vqc-component-analysis)
3. [QLSTM Cell Analysis](#qlstm-cell-analysis)
4. [Critical Issues](#critical-issues)
5. [The Sigmoid/Tanh Problem](#the-sigmoidtanh-problem)
6. [Impact on Time-Series Tasks](#impact-on-time-series-tasks)
7. [Comparison with Periodic-Aware Design](#comparison-with-periodic-aware-design)
8. [Recommendations](#recommendations)

---

## Model Architecture Overview

The QLSTM model follows this structure:

```
Input Time-Series [batch, seq_len, input_size]
    │
    ▼
┌─────────────────────────────────────────────┐
│   CustomLSTM                                │
│   ├── For each timestep t:                  │
│   │   ├── Concatenate: [x_t, h_prev]        │
│   │   ├── VQC → sigmoid (input gate)        │
│   │   ├── VQC → sigmoid (forget gate)       │
│   │   ├── VQC → tanh (cell gate)            │
│   │   ├── VQC → sigmoid (output gate)       │
│   │   ├── Cell state update                 │
│   │   └── Hidden state update               │
│   └── Collect outputs                       │
└─────────────────────────────────────────────┘
    │
    ▼
Output Prediction [batch, seq_len, output_size]
```

---

## VQC Component Analysis

### VQC Circuit Definition (lines 182-205)

```python
def q_function(x, q_weights, n_class):
    """ The variational quantum circuit. """
    n_dep = q_weights.shape[0]
    n_qub = q_weights.shape[1]

    # Initialize to |+⟩ state
    H_layer(n_qub)

    # Embed features in the quantum node
    RY_layer(x)  # ← ISSUE 1: Generic encoding, no frequency matching

    # Sequence of trainable variational layers
    for k in range(n_dep):
        entangling_layer(n_qub)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]
    return exp_vals
```

### Issues with VQC Design

| Component | Current Implementation | Issue |
|-----------|----------------------|-------|
| **Initialization** | `H_layer(n_qub)` | Standard, acceptable |
| **Data Encoding** | `RY_layer(x)` | Generic RY only, no frequency scaling |
| **Entanglement** | `CNOT` layers | Standard, acceptable |
| **Variational** | `RY_layer(q_weights[k])` | Only RY rotations |
| **Measurement** | `PauliZ` expectation | Standard, acceptable |

### Missing Elements for Periodic Advantage

1. **No learnable frequency scaling** (like `freq_scale` or Snake's `a`)
2. **No frequency-matched encoding** (RY + scaled RX)
3. **No FFT preprocessing** to extract frequency components
4. **Single-axis encoding** (RY only) limits Fourier spectrum

---

## QLSTM Cell Analysis

### CustomQLSTMCell Structure (lines 265-300)

```python
class CustomQLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vqc_depth):
        super(CustomQLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # VQCs replace Linear layers for LSTM gates
        self.input_gate = VQC(vqc_depth, n_qubits=input_size + hidden_size, n_class=hidden_size)
        self.forget_gate = VQC(vqc_depth, n_qubits=input_size + hidden_size, n_class=hidden_size)
        self.cell_gate = VQC(vqc_depth, n_qubits=input_size + hidden_size, n_class=hidden_size)
        self.output_gate = VQC(vqc_depth, n_qubits=input_size + hidden_size, n_class=hidden_size)

        self.output_post_processing = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)

        # CRITICAL ISSUE: sigmoid/tanh AFTER VQC destroys periodic structure!
        i_t = torch.sigmoid(self.input_gate(combined))  # ← Destroys periodicity
        f_t = torch.sigmoid(self.forget_gate(combined)) # ← Destroys periodicity
        g_t = torch.tanh(self.cell_gate(combined))      # ← Destroys periodicity
        o_t = torch.sigmoid(self.output_gate(combined)) # ← Destroys periodicity

        # Cell state update (classical LSTM mechanism)
        c_t = f_t * c_prev + i_t * g_t

        # Hidden state update
        h_t = o_t * torch.tanh(c_t)  # ← Another tanh!

        # Final output
        out = self.output_post_processing(h_t)

        return out, h_t, c_t
```

### The Design Philosophy Problem

The QLSTM takes a **"drop-in replacement"** approach:

```
Classical LSTM:     Linear → sigmoid/tanh
QLSTM:              VQC → sigmoid/tanh  (just replace Linear with VQC)
```

This approach **fundamentally misunderstands** VQC's advantages:

- VQCs are not just "better linear layers"
- VQCs are **Fourier series generators**
- Applying sigmoid/tanh after VQC output **destroys** the Fourier structure

---

## Critical Issues

### Issue 1: Generic RY Encoding (No Frequency Matching)

```python
def RY_layer(w):
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)
```

**Problem**: Raw values encoded without frequency scaling.

**What's missing**:
```python
# Frequency-matched encoding (not in current code)
for i, wire in enumerate(wires):
    qml.RY(features[i], wires=wire)           # Amplitude
    qml.RX(freq_scale[i] * features[i], wires=wire)  # Frequency-matched
```

### Issue 2: No Frequency Extraction

**Current**: Raw concatenated input `[x_t, h_prev]` fed directly to VQC.

**Problem**: Time-series periodic structure not extracted.

**What's missing**:
```python
# FFT-based frequency extraction (not in current code)
fft_result = torch.fft.rfft(window, dim=-1)
magnitude = torch.log1p(torch.abs(fft_result))
phase = torch.angle(fft_result) / np.pi
```

### Issue 3: Sigmoid/Tanh After VQC (Most Critical)

**Current code**:
```python
i_t = torch.sigmoid(self.input_gate(combined))
f_t = torch.sigmoid(self.forget_gate(combined))
g_t = torch.tanh(self.cell_gate(combined))
o_t = torch.sigmoid(self.output_gate(combined))
```

**This completely destroys VQC's periodic advantage!**

### Issue 4: No Learnable Frequency Scaling

**Current**: Fixed RY encoding.

**What's missing**:
```python
self.freq_scale = nn.Parameter(torch.linspace(0.5, 3.0, n_qubits))
```

### Issue 5: Classical LSTM Memory Mechanism

The cell state update uses non-periodic operations:
```python
c_t = f_t * c_prev + i_t * g_t  # Element-wise multiply and add
h_t = o_t * torch.tanh(c_t)     # tanh destroys periodicity
```

---

## The Sigmoid/Tanh Problem

### Mathematical Analysis

VQC outputs a Fourier series:

$$f_{VQC}(x) = \sum_{\omega \in \Omega} c_\omega e^{i\omega x}$$

This is a **periodic function** with rich frequency content.

But then sigmoid is applied:

$$\sigma(f_{VQC}(x)) = \frac{1}{1 + e^{-f_{VQC}(x)}}$$

### Why Sigmoid Destroys Periodicity

| Property | VQC Output | After Sigmoid |
|----------|------------|---------------|
| **Range** | $(-\infty, +\infty)$ or $[-1, 1]$ | $(0, 1)$ |
| **Periodicity** | Periodic (Fourier series) | **Not periodic** |
| **Monotonicity** | Oscillating | Monotonic (in transformation) |
| **Information** | Full frequency spectrum | Compressed, distorted |

### Visual Representation

```
VQC Output (Periodic):
    ∧     ∧     ∧
   / \   / \   / \
──/───\─/───\─/───\──  (oscillates around 0)
       \/    \/

After Sigmoid (Non-Periodic):
         ___________
        /
       /
──────/              (bounded, monotonic transformation)
```

### Comparison: This is WORSE than Original QTCN

| Model | Post-VQC Processing | Periodicity Preserved? |
|-------|---------------------|----------------------|
| **Original QTCN** | `torch.mean()` | Partially (low-pass filter) |
| **QLSTM_v0** | `sigmoid()` / `tanh()` | **NO** (completely destroyed) |
| **Fourier QTCN** | Learnable weighted sum | **YES** |

---

## Impact on Time-Series Tasks

### NARMA Task (Used in This Code)

```python
x, y = get_narma_data(n_0 = 10, seq_len = 4)
```

NARMA (Nonlinear AutoRegressive Moving Average) is defined as:

$$y(t) = \alpha y(t-1) + \beta y(t-1) \sum_{i=0}^{n-1} y(t-1-i) + \gamma u(t-n) u(t-1) + \delta$$

**Characteristics**:
- Nonlinear temporal dependencies
- Memory requirements (depends on past n values)
- **Not explicitly periodic**, but has temporal structure

**VQC's potential advantage**: Could capture nonlinear patterns through Fourier decomposition.

**Current limitation**: Sigmoid/tanh destroys any learned periodic representations.

### For Explicitly Periodic Time-Series

For tasks with periodic signals (e.g., EEG rhythms, seasonal data):

| Task Type | VQC Potential | Current QLSTM Performance |
|-----------|---------------|---------------------------|
| Periodic signals | High (Fourier match) | **Wasted** (sigmoid destroys) |
| Quasi-periodic | Moderate | Wasted |
| Nonlinear dynamics | Moderate | Partially utilized |

---

## Comparison with Periodic-Aware Design

### Current QLSTM vs. Periodic-Aware Design

| Aspect | Current QLSTM | Periodic-Aware Design |
|--------|---------------|----------------------|
| **Input Processing** | Raw `torch.cat((x, h_prev))` | FFT frequency extraction |
| **Encoding** | `RY_layer(x)` only | `RY(x)` + `RX(freq_scale * x)` |
| **Post-VQC Activation** | `sigmoid()` / `tanh()` | Direct output or periodic activation |
| **Frequency Matching** | None | Learnable `freq_scale` per qubit |
| **Gate Design** | Classical sigmoid gates | Redesigned for periodic signals |
| **Memory Mechanism** | Classical cell state | Periodic-preserving updates |
| **VQC as** | Drop-in linear replacement | Fourier series generator |

### What Would Need to Change

```python
# Current (destroys periodicity)
i_t = torch.sigmoid(self.input_gate(combined))

# Periodic-aware (preserves periodicity)
# Option 1: Direct VQC output (already in [-1, 1] for PauliZ)
i_t = (self.input_gate(combined) + 1) / 2  # Rescale to [0, 1]

# Option 2: Periodic activation
i_t = 0.5 + 0.5 * torch.sin(self.input_gate(combined))
```

---

## Recommendations

### For Utilizing VQC's Periodic Advantages in QLSTM

#### 1. Add FFT-Based Preprocessing

```python
def preprocess_input(self, x, h_prev):
    # Extract frequency components from input
    fft_x = torch.fft.rfft(x, dim=-1)
    magnitude = torch.log1p(torch.abs(fft_x))
    phase = torch.angle(fft_x) / np.pi

    # Combine with hidden state
    freq_features = torch.cat([magnitude, phase, h_prev], dim=-1)
    return freq_features
```

#### 2. Add Learnable Frequency Scaling

```python
class PeriodicVQC(nn.Module):
    def __init__(self, ...):
        # Learnable frequency scale (like Snake's 'a')
        self.freq_scale = nn.Parameter(torch.linspace(0.5, 3.0, n_qubits))

    def encode(self, x):
        for i, wire in enumerate(range(self.n_qubits)):
            qml.RY(x[i], wires=wire)  # Amplitude
            qml.RX(self.freq_scale[i] * x[i], wires=wire)  # Frequency-matched
```

#### 3. Replace Sigmoid/Tanh with Periodic-Preserving Activations

```python
def forward(self, x, hidden):
    combined = self.preprocess_input(x, h_prev)

    # Option A: Rescale VQC output directly
    i_t = (self.input_gate(combined) + 1) / 2  # [−1,1] → [0,1]

    # Option B: Use periodic activation
    i_t = 0.5 + 0.5 * torch.sin(self.input_gate(combined))

    # Option C: Learnable periodic gate
    i_t = self.gate_activation(self.input_gate(combined))
```

#### 4. Redesign Cell State Updates

```python
# Current: c_t = f_t * c_prev + i_t * g_t (non-periodic)

# Periodic-aware: Use phase-based updates
phase_update = self.phase_gate(combined)  # Returns phase adjustment
c_t = c_prev * torch.cos(phase_update) + g_t * torch.sin(phase_update)
```

---

## Summary Table

| Criterion | Status | Notes |
|-----------|--------|-------|
| Frequency extraction from input | **Missing** | Raw values used |
| Learnable frequency scaling | **Missing** | Fixed RY encoding |
| Frequency-matched encoding | **Missing** | RY only, no RX scaling |
| Periodic-preserving post-processing | **Failed** | Sigmoid/tanh destroys |
| VQC treated as Fourier generator | **No** | Treated as linear replacement |
| **Overall Periodic Advantage** | **NOT UTILIZED** | Actively destroyed |

---

## Conclusion

The QLSTM_v0.py implementation:

1. **Does not extract** frequency information from input
2. **Does not match** VQC's frequency spectrum to data
3. **Actively destroys** VQC's periodic output through sigmoid/tanh activations
4. Treats VQCs as **drop-in replacements** for linear layers, missing their fundamental nature as Fourier series generators

For univariate or low-dimensional multivariate time-series with periodic structure, this model **wastes VQC's unique capabilities** and may perform no better than (or worse than) a classical LSTM.

---

## Related Documents

- `FourierQTCN_README.md` - Implementation that properly utilizes VQC's periodic advantages
- `VQC_Periodic_Data_Encoding_Guide.md` - Guide on proper encoding for periodic data
- `QTCN_Periodic_Advantage_Analysis.md` - Similar analysis for QTCN model
- `VQC_vs_ReLU_Periodic_Data_Comparison.md` - Theoretical comparison with classical methods
