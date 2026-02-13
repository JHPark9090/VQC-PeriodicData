# Fourier-QLSTM: Quantum LSTM with Full Periodic Advantage

A redesigned Quantum LSTM that fully utilizes VQC's periodic advantages through frequency-domain processing and periodic-preserving operations.

## Table of Contents

1. [Overview](#overview)
2. [The Problem with Original QLSTM](#the-problem-with-original-qlstm)
3. [Theoretical Foundation](#theoretical-foundation)
4. [Architecture](#architecture)
5. [Key Innovations](#key-innovations)
6. [Comparison with Original QLSTM](#comparison-with-original-qlstm)
7. [Usage](#usage)
8. [Hyperparameters](#hyperparameters)
9. [Implementation Details](#implementation-details)
10. [References](#references)

---

## Overview

Fourier-QLSTM is a complete redesign of Quantum LSTM that addresses the fundamental issue: **the original QLSTM destroys VQC's periodic structure through sigmoid/tanh activations**.

### Design Philosophy

```
Original QLSTM:  VQC as "better Linear layer" → sigmoid → output
                                                   ↑
                                          DESTROYS PERIODICITY

Fourier-QLSTM:   FFT → VQC as "Fourier generator" → rescale → output
                  ↑              ↑                      ↑
              LOSSLESS    FREQUENCY-MATCHED      PRESERVES PERIODICITY
```

### Key Features

| Feature | Description |
|---------|-------------|
| **FFT Preprocessing** | Lossless frequency extraction from input |
| **Frequency-Matched Encoding** | RY + learnable freq_scale × RX |
| **FFT-Seeded Initialization** | Data-informed freq_scale starting point via power spectrum analysis |
| **Rescaled Gating** | (VQC + 1) / 2 instead of sigmoid |
| **Frequency-Domain Memory** | Cell state as (magnitude, phase) |
| **Phase Accumulation** | Additive phase updates for periodic signals |

---

## The Problem with Original QLSTM

The original QLSTM (QLSTM_v0.py) has a critical flaw:

```python
# Original QLSTM gates
i_t = torch.sigmoid(self.input_gate(combined))   # ← DESTROYS PERIODICITY
f_t = torch.sigmoid(self.forget_gate(combined))  # ← DESTROYS PERIODICITY
g_t = torch.tanh(self.cell_gate(combined))       # ← DESTROYS PERIODICITY
o_t = torch.sigmoid(self.output_gate(combined))  # ← DESTROYS PERIODICITY
```

### Why Sigmoid/Tanh Destroys Periodicity

VQC outputs a Fourier series:

$$f_{VQC}(x) = \sum_{\omega \in \Omega} c_\omega e^{i\omega x}$$

This is **periodic** with rich frequency content. But sigmoid transforms it:

$$\sigma(f_{VQC}(x)) = \frac{1}{1 + e^{-f_{VQC}(x)}}$$

| Property | VQC Output | After Sigmoid |
|----------|------------|---------------|
| Range | Oscillating | Bounded (0, 1) |
| Periodicity | **Periodic** | **Not periodic** |
| Frequency content | Rich spectrum | Distorted/lost |

### Other Issues

1. **No frequency extraction**: Raw time-domain values fed to VQC
2. **Generic encoding**: RY only, no frequency matching
3. **Classical memory**: Cell state doesn't leverage periodic structure

---

## Theoretical Foundation

### VQC as Fourier Series Generator

From Schuld et al. (2021), a VQC with angle encoding computes:

$$f(x) = \sum_{\omega \in \Omega} c_\omega e^{i\omega x}$$

where:
- $\Omega$ = frequency spectrum (determined by encoding Hamiltonians)
- $c_\omega$ = trainable Fourier coefficients (circuit parameters)

### Frequency-Matched Encoding

To leverage VQC's periodic nature, we need:

1. **Extract frequency content** from input (via FFT)
2. **Match VQC's spectrum** to data's frequencies (via learnable scaling)

Our encoding:
$$\text{Encoding}(x_i) = RY(x_i) \cdot RX(\alpha_i \cdot x_i)$$

where $\alpha_i$ is learnable per qubit (like Snake's `a` parameter).

### Frequency-Domain Memory

Instead of classical cell state $c_t \in \mathbb{R}^n$, we use:

$$c_t = (c_{mag}, c_{phase}) \in \mathbb{R}^n \times [-\pi, \pi]^n$$

This represents memory as **Fourier coefficients**, naturally aligned with VQC's output.

### Phase Accumulation

For periodic signals, phase accumulates over time:

$$c_{phase}^{(t+1)} = c_{phase}^{(t)} + i_t \cdot \angle(g_t)$$

This is the natural way periodic signals evolve (think: rotating phasors).

---

## Architecture

### Overall Structure

```
Input x_t [batch, window_size]
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: FFT Preprocessing (LOSSLESS)                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ x_t → rfft → magnitude (log scale) + phase (normalized) │    │
│  │ Output: [batch, n_frequencies × 2]                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Linear Projection (Minimal Classical)                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ FFT features → n_qubits dimension                        │    │
│  │ Hidden state (mag, phase) → n_qubits dimension           │    │
│  │ Combined = input_proj + hidden_proj                      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Frequency-Matched VQC Gates                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Encoding: H → RY(x) → RX(freq_scale × x)                 │    │
│  │ Entangling: CNOT ladder                                  │    │
│  │ Variational: RY(θ) layers                                │    │
│  │ Measurement: PauliZ expectations                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────┐ │
│  │ Input Gate   │ │ Forget Gate  │ │ Cell Gate    │ │ Output  │ │
│  │ VQC_i        │ │ VQC_f        │ │ VQC_g        │ │ VQC_o   │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └─────────┘ │
│         │                │                │              │      │
│         ▼                ▼                ▼              ▼      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Rescale: (VQC + 1) / 2  ← PRESERVES PERIODICITY          │   │
│  │ (Instead of sigmoid which destroys periodicity)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Frequency-Domain Cell State Update                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Magnitude: c_mag = f_t × c_mag_prev + i_t × |g_t|        │    │
│  │ Phase:     c_phase = c_phase_prev + i_t × angle(g_t)     │    │
│  │            c_phase = wrap_to_range(c_phase)              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Periodic-Preserving Output                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ h_t = o_t × c_mag × cos(c_phase × π)                     │    │
│  │ output = linear_projection(h_t)                          │    │
│  │                                                          │    │
│  │ (Instead of h_t = o_t × tanh(c_t) which loses phase)     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Output [batch, output_size]
Hidden State: (c_mag, c_phase) for next timestep
```

### Quantum Circuit Structure

```
        ┌───┐ ┌──────────┐ ┌─────────────────────┐
q_0: ───┤ H ├─┤ RY(x_0)  ├─┤ RX(α_0 × x_0)       ├──■────────────────┤ RY(θ_0) ├─── ⟨Z⟩
        └───┘ └──────────┘ └─────────────────────┘  │                 └─────────┘
        ┌───┐ ┌──────────┐ ┌─────────────────────┐ ┌┴┐
q_1: ───┤ H ├─┤ RY(x_1)  ├─┤ RX(α_1 × x_1)       ├─┤X├──■─────────────┤ RY(θ_1) ├─── ⟨Z⟩
        └───┘ └──────────┘ └─────────────────────┘ └─┘  │             └─────────┘
        ┌───┐ ┌──────────┐ ┌─────────────────────┐     ┌┴┐
q_2: ───┤ H ├─┤ RY(x_2)  ├─┤ RX(α_2 × x_2)       ├─────┤X├──■─────────┤ RY(θ_2) ├─── ⟨Z⟩
        └───┘ └──────────┘ └─────────────────────┘     └─┘  │         └─────────┘
        ...                                                 ...

        ↑                   ↑                          ↑         ↑
    Superposition     Frequency-Matched           Entangling  Variational
                        Encoding
```

---

## Key Innovations

### 1. FFT Preprocessing (Lossless)

```python
def _extract_fft_features(self, x):
    # FFT along time dimension
    fft_result = torch.fft.rfft(x, dim=1)

    # Extract magnitude (log scale for dynamic range)
    magnitude = torch.log1p(torch.abs(fft_result))

    # Extract phase (normalized to [-1, 1])
    phase = torch.angle(fft_result) / np.pi

    return torch.cat([magnitude, phase], dim=1)
```

**Why FFT?**
- **Lossless**: FFT is invertible, no information loss
- **Fourier alignment**: VQC outputs Fourier series, FFT extracts Fourier coefficients
- **Natural match**: Both input and VQC speak the same "language"

### 2. Frequency-Matched Encoding with FFT-Seeded Initialization

```python
def _circuit_fn(self, inputs):
    for i in range(self.n_qubits):
        qml.Hadamard(wires=i)
        qml.RY(inputs[i], wires=i)                      # Amplitude encoding
        qml.RX(self.freq_scale[i] * inputs[i], wires=i) # Frequency-matched
```

**Why learnable freq_scale?**
- Maps input frequencies to VQC's natural frequency spectrum
- Similar to Snake activation's learnable `a` parameter
- Each qubit specializes in different frequency range

**FFT-seeded initialization** (`--freq-init=fft`, default):
- Analyzes training data power spectrum to find dominant frequencies
- Converts top-N frequency bin indices to ratio-based scaling factors
- Clamps to [0.5, 5.0] range, sorted ascending (lower qubits = lower frequencies)
- Parameter remains learnable — FFT provides a data-informed starting point
- Falls back to `linspace(0.5, 3.0)` with `--freq-init=linspace`

```python
# FFT-seeded: different datasets → different initializations
# NARMA:  freq_scale = [1.0, 2.0, 3.0, 4.0, 4.5, 5.0]
# ETTh1:  freq_scale = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]
# vs linspace: freq_scale = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  (always the same)
```

### 3. Rescaled Gating (No Sigmoid!)

```python
def _rescale_gate(self, x):
    # VQC output ∈ [-1, 1] (from PauliZ measurement)
    # Rescale to [0, 1] for gating
    return (x + 1) / 2
```

**Why not sigmoid?**

| Operation | Periodicity | Mathematical Effect |
|-----------|-------------|---------------------|
| `sigmoid(x)` | **Destroyed** | Monotonic squashing |
| `(x + 1) / 2` | **Preserved** | Linear rescaling |

The rescaling is a **linear transformation**, which preserves the Fourier structure of VQC output.

### 4. Frequency-Domain Memory

```python
# Cell state as (magnitude, phase)
c_mag_new = f_t * c_mag + i_t * torch.abs(g_t)
c_phase_new = c_phase + i_t * angle(g_t)
c_phase_new = wrap_to_range(c_phase_new)  # Keep bounded
```

**Why frequency-domain?**
- Aligns with VQC's Fourier output
- Phase accumulation is natural for periodic signals
- Magnitude captures "strength" of each frequency

### 5. Periodic-Preserving Output

```python
# Instead of: h_t = o_t * tanh(c_t)
# We use:
h_t = o_t * c_mag * torch.cos(c_phase * np.pi)
```

**Why cos instead of tanh?**
- `cos` is periodic, preserves phase information
- Combines magnitude and phase naturally
- Output maintains periodic structure

---

## Comparison with Original QLSTM

```
╔══════════════════════════════════════════════════════════════════════════╗
║              OLD QLSTM vs FOURIER-QLSTM                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  OLD QLSTM (QLSTM_v0.py):                                                ║
║  ├─ Input: Raw time-domain values                                        ║
║  ├─ Encoding: Generic RY only                                            ║
║  ├─ Gates: sigmoid(VQC) / tanh(VQC) ← DESTROYS PERIODICITY               ║
║  ├─ Memory: Classical (c_t, h_t)                                         ║
║  └─ VQC Periodic Advantage: NOT UTILIZED                                 ║
║                                                                          ║
║  FOURIER-QLSTM:                                                          ║
║  ├─ Input: FFT → (magnitude, phase)                                      ║
║  ├─ Encoding: RY(x) + RX(freq_scale × x)                                 ║
║  ├─ Gates: rescale(VQC) = (VQC + 1) / 2 ← PRESERVES PERIODICITY          ║
║  ├─ Memory: Frequency-domain (c_magnitude, c_phase)                      ║
║  └─ VQC Periodic Advantage: FULLY UTILIZED                               ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Detailed Comparison Table

| Aspect | Original QLSTM | Fourier-QLSTM |
|--------|----------------|---------------|
| **Input Processing** | Raw values | FFT (lossless) |
| **Frequency Extraction** | None | Magnitude + Phase |
| **Encoding Gates** | RY only | RY + freq_scale × RX |
| **Frequency Matching** | None | Learnable per qubit |
| **freq_scale Init** | N/A | FFT-seeded (data-informed) or linspace (generic) |
| **Gate Activation** | sigmoid / tanh | (VQC + 1) / 2 |
| **Periodicity After Gate** | **Destroyed** | **Preserved** |
| **Cell State Type** | Single tensor | (magnitude, phase) |
| **Phase Information** | Lost | Preserved & accumulated |
| **Hidden State Update** | o × tanh(c) | o × mag × cos(phase) |
| **VQC Treated As** | Linear replacement | Fourier generator |
| **Periodic Advantage** | **Wasted** | **Utilized** |

---

## Usage

### Command-Line Usage (FFT-seeded init, default)

```bash
cd /pscratch/sd/j/junghoon/VQC-PeriodicData

# NARMA with FFT-seeded freq_scale (default)
PYTHONNOUSERSITE=1 python models/FourierQLSTM.py \
    --dataset=narma \
    --n-qubits=6 \
    --vqc-depth=2 \
    --n-epochs=50

# With generic linspace init (legacy behavior)
PYTHONNOUSERSITE=1 python models/FourierQLSTM.py \
    --dataset=narma \
    --freq-init=linspace \
    --n-epochs=50

# Multisine dataset with FFT-seeded init
PYTHONNOUSERSITE=1 python models/FourierQLSTM.py \
    --dataset=multisine \
    --freq-init=fft \
    --n-qubits=8 \
    --n-epochs=100
```

### Python API — Basic Usage

```python
from FourierQLSTM import FourierQLSTM, analyze_training_frequencies

# Create model (generic init)
model = FourierQLSTM(
    input_size=1,       # Univariate time-series
    hidden_size=4,      # Frequency-domain hidden dimension
    n_qubits=6,         # Number of qubits
    vqc_depth=2,        # VQC circuit depth
    output_size=1,      # Prediction dimension
    window_size=8       # FFT window size
)

# Forward pass
# x: [batch, seq_len] or [batch, seq_len, input_size]
outputs, (c_mag, c_phase) = model(x)

# outputs: [batch, n_windows, output_size]
# Use outputs[:, -1, :] for final prediction
```

### With FFT-Seeded Initialization

```python
from FourierQLSTM import FourierQLSTM, analyze_training_frequencies

# Analyze training data to seed freq_scale
freq_scale_init = analyze_training_frequencies(x_train, n_qubits=6)
# Prints: FFT-seeded freq_scale: [1.0, 2.0, 3.0, 4.0, 4.5, 5.0]

# Create model with data-informed initialization
model = FourierQLSTM(
    input_size=1,
    hidden_size=4,
    n_qubits=6,
    vqc_depth=2,
    window_size=8,
    freq_scale_init=freq_scale_init  # FFT-seeded
).double()
```

### With Custom Configuration

```python
model = FourierQLSTM(
    input_size=1,
    hidden_size=8,        # Larger frequency-domain hidden
    n_qubits=8,           # More qubits
    vqc_depth=3,          # Deeper circuit
    output_size=1,
    window_size=16,       # Larger FFT window
    n_frequencies=8       # Custom number of frequencies
)
```

### Training Example

```python
import torch
from torch.optim import Adam

# Data
x_train = torch.randn(100, 20)  # [batch, seq_len]
y_train = torch.randn(100, 1)   # [batch, output_size]

# FFT-seeded initialization
freq_scale_init = analyze_training_frequencies(x_train, n_qubits=6)

# Model
model = FourierQLSTM(
    input_size=1,
    hidden_size=4,
    n_qubits=6,
    vqc_depth=2,
    window_size=8,
    freq_scale_init=freq_scale_init
).double()

# Training
optimizer = Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()

    outputs, _ = model(x_train.double())
    predictions = outputs[:, -1, :]  # Last window output

    loss = F.mse_loss(predictions, y_train.double())
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
```

### Accessing Learned Frequency Scales

```python
# After training, inspect learned frequency scales
for gate_name in ['input_gate', 'forget_gate', 'cell_gate', 'output_gate']:
    gate = getattr(model.cell, gate_name)
    freq_scales = gate.get_freq_scales()
    print(f"{gate_name}: {freq_scales}")
```

---

## Hyperparameters

### Model Parameters

| Parameter | Default | Description | Recommended Range |
|-----------|---------|-------------|-------------------|
| `input_size` | 1 | Input dimension | 1 (univariate) to 10 |
| `hidden_size` | 4 | Frequency-domain hidden size | 4, 8, 16 |
| `n_qubits` | 6 | Number of qubits | 4, 6, 8, 12 |
| `vqc_depth` | 2 | VQC circuit depth | 1, 2, 3 |
| `output_size` | 1 | Output dimension | Task-dependent |
| `window_size` | 8 | FFT window size | 8, 16, 32 |
| `n_frequencies` | window_size//2+1 | FFT frequencies | ≤ window_size//2+1 |
| `freq_init` | `fft` | freq_scale initialization method | `fft` (data-informed) or `linspace` (generic) |

### Training Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Learning rate | 0.01 - 0.001 | Start higher, reduce if unstable |
| Batch size | 10 - 32 | Smaller for quantum simulation |
| Epochs | 50 - 200 | Monitor convergence |
| Optimizer | Adam, RMSprop | Both work well |

### Choosing Window Size

| Data Characteristics | Recommended Window Size |
|---------------------|------------------------|
| High-frequency content | 16 - 32 |
| Low-frequency content | 8 - 16 |
| Short sequences | 4 - 8 |
| Long sequences | 16 - 32 |

---

## Implementation Details

### FFT Feature Extraction

```python
def _extract_fft_features(self, x):
    """
    Extract frequency-domain features using FFT.

    Input:  [batch, window_size] or [batch, window_size, input_size]
    Output: [batch, n_frequencies × 2 × input_size]
    """
    # Real FFT (positive frequencies only)
    fft_result = torch.fft.rfft(x, dim=1)

    # Magnitude: log1p for numerical stability and dynamic range
    magnitude = torch.log1p(torch.abs(fft_result))

    # Phase: normalized to [-1, 1]
    phase = torch.angle(fft_result) / np.pi

    # Concatenate and flatten
    features = torch.cat([magnitude, phase], dim=1)
    return features.reshape(batch_size, -1)
```

### Frequency-Matched VQC

```python
def _circuit_fn(self, inputs):
    """
    Quantum circuit with frequency-matched encoding.
    """
    for i in range(self.n_qubits):
        # Initialize in superposition
        qml.Hadamard(wires=i)

        # Amplitude encoding (standard)
        qml.RY(inputs[i], wires=i)

        # Frequency-matched encoding (KEY innovation)
        # freq_scale is learned during training
        qml.RX(self.freq_scale[i] * inputs[i], wires=i)

    # Entangling + variational layers
    for layer in range(self.vqc_depth):
        # CNOT ladder
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Parameterized rotations
        for i in range(self.n_qubits):
            qml.RY(self.var_params[layer, i], wires=i)

    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(self.n_outputs)]
```

### Cell State Update

```python
def forward(self, x, hidden):
    # ... (FFT preprocessing and projection)

    # VQC gates with rescaling (not sigmoid!)
    i_t = self._rescale_gate(self.input_gate(combined))   # [0, 1]
    f_t = self._rescale_gate(self.forget_gate(combined))  # [0, 1]
    g_t = self.cell_gate(combined)                        # [-1, 1]
    o_t = self._rescale_gate(self.output_gate(combined))  # [0, 1]

    # Frequency-domain cell state update
    c_mag_new = f_t * c_mag + i_t * torch.abs(g_t)

    # Phase accumulation (natural for periodic signals)
    c_phase_new = c_phase + i_t * torch.atan2(
        torch.sin(g_t * np.pi),
        torch.cos(g_t * np.pi)
    ) / np.pi

    # Wrap phase to [-1, 1]
    c_phase_new = torch.remainder(c_phase_new + 1, 2) - 1

    # Periodic-preserving output
    h_t = o_t * c_mag_new * torch.cos(c_phase_new * np.pi)

    return output, (c_mag_new, c_phase_new)
```

---

## File Location

**Implementation**: `/pscratch/sd/j/junghoon/VQC-PeriodicData/models/FourierQLSTM.py`

**Related Documentation**:
- `QLSTM_Periodic_Advantage_Analysis.md` - Analysis of original QLSTM's limitations
- `FourierQTCN_README.md` - Similar redesign for QTCN
- `VQC_Periodic_Data_Encoding_Guide.md` - Encoding strategies for VQCs

---

## References

1. **Schuld, M., Sweke, R., & Meyer, J. J.** (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A*, 103(3), 032430.
   - Foundation for understanding VQC as Fourier series

2. **Ziyin, L., Hartwig, T., & Ueda, M.** (2020). Neural networks fail to learn periodic functions and how to fix it. *Advances in Neural Information Processing Systems*, 33.
   - Snake activation and periodic function learning

3. **Chen, S. Y. C., et al.** (2022). Quantum Long Short-Term Memory. *IEEE International Conference on Quantum Computing and Engineering*.
   - Original QLSTM architecture

---

## Summary

Fourier-QLSTM addresses all the issues with the original QLSTM:

| Issue | Original QLSTM | Fourier-QLSTM Solution |
|-------|----------------|------------------------|
| Sigmoid destroys periodicity | `sigmoid(VQC)` | `(VQC + 1) / 2` |
| No frequency extraction | Raw values | FFT preprocessing |
| Generic encoding | RY only | RY + freq_scale × RX |
| Blind parameter init | N/A | FFT-seeded freq_scale from data power spectrum |
| Classical memory | Single tensor | (magnitude, phase) |
| Phase information lost | tanh(c_t) | cos(c_phase × π) |

The result is a Quantum LSTM that **fully utilizes VQC's unique capability as a Fourier series generator**, rather than treating it as a drop-in replacement for linear layers.
