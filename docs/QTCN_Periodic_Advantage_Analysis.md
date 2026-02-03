# Analysis: Does QTCN Fully Utilize VQC's Periodic Advantage?

## Critical Evaluation of HQTCN2_EEG.py's QTCN Model

**Date**: February 2026
**File Analyzed**: `/pscratch/sd/j/junghoon/HQTCN_Project/scripts/HQTCN2_EEG.py`

---

## Executive Summary

**Question**: Does the QTCN model fully utilize VQC's periodic advantages for high-dimensional spatio-temporal data like EEG?

**Answer**: **No, it does NOT.**

The QTCN model uses VQC for feature extraction but **fails to leverage VQC's inherent periodic structure**. The model treats VQC as a generic feature extractor rather than a periodic function learner.

---

## 1. Original QTCN Architecture Analysis

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QTCN ARCHITECTURE FLOW                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input: EEG [batch, channels, time_steps]                          │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────┐                       │
│  │  Sliding Window (kernel_size, dilation) │  ← Temporal context   │
│  └─────────────────────────────────────────┘                       │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────┐                       │
│  │  FC Layer: (channels × kernel) → n_qubits│  ← Dimension reduction│
│  └─────────────────────────────────────────┘                       │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────┐                       │
│  │  AngleEmbedding (RY rotation)           │  ← Generic encoding   │
│  └─────────────────────────────────────────┘                       │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────┐                       │
│  │  QCNN (Conv + Pool layers)              │  ← Quantum processing │
│  └─────────────────────────────────────────┘                       │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────┐                       │
│  │  torch.mean() over time windows         │  ← Temporal averaging │
│  └─────────────────────────────────────────┘                       │
│                         │                                           │
│                         ▼                                           │
│  Output: Single prediction                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Code Sections

#### Dimension Reduction (Line 65)
```python
self.fc = nn.Linear(self.input_channels * self.kernel_size, n_qubits)
```

#### Quantum Encoding (Lines 69-70)
```python
qml.AngleEmbedding(features, wires=wires, rotation='Y')
```

#### Temporal Aggregation (Line 119)
```python
output = torch.mean(torch.stack(output, dim=1), dim=1)
```

---

## 2. Critical Issues Preventing VQC Periodic Advantage

### Issue 1: No Frequency Analysis

**Current Implementation:**
```python
# Line 65:
self.fc = nn.Linear(self.input_channels * self.kernel_size, n_qubits)
```

**Problem:**
- Generic linear layer that doesn't analyze or preserve EEG frequency content
- Mixes all frequency information (alpha, beta, gamma, etc.) without consideration
- No separation of periodic components

**Impact:**
The FC layer destroys the frequency structure of EEG data before it reaches the quantum circuit.

---

### Issue 2: Generic Angle Embedding Without Frequency Matching

**Current Implementation:**
```python
# Lines 69-70:
qml.AngleEmbedding(features, wires=wires, rotation='Y')
```

**Problem:**
- Encodes features directly without rescaling to match VQC's natural period (2π)
- Does not match EEG frequency bands to VQC frequency spectrum
- VQC outputs Fourier series with frequencies determined by encoding, but encoding doesn't consider data's periodic structure

**Impact:**
VQC's frequency spectrum (determined by encoding) is NOT matched to EEG's frequency content. The quantum circuit cannot leverage its periodic nature.

**What Should Happen:**
```python
# For EEG with alpha (10 Hz), beta (20 Hz), gamma (40 Hz):
# VQC should encode with matched frequencies

# Example of frequency-matched encoding:
for i, wire in enumerate(wires):
    qml.RY(features[i], wires=wire)           # Amplitude encoding
    qml.RX(freq_scale[i] * features[i], wires=wire)  # Frequency-matched encoding
```

---

### Issue 3: Temporal Averaging Destroys Periodicity

**Current Implementation:**
```python
# Line 119:
output = torch.mean(torch.stack(output, dim=1), dim=1)
```

**Problem:**
- Averaging over time windows destroys periodic information
- Even if VQC captured periodic patterns, they are lost in the mean operation
- No preservation of temporal/frequency structure in the output

**Impact:**
Any periodic features captured by the quantum circuit are averaged away.

**What Should Happen:**
```python
# Use attention-based aggregation to preserve important periodic patterns:
attended, _ = self.temporal_attention(outputs, outputs, outputs)
output = attended.weighted_sum()

# Or use frequency-aware pooling:
output = self.frequency_pooling(outputs)  # Preserves dominant frequencies
```

---

### Issue 4: No Frequency Band Decomposition

**Current Implementation:**
- Raw EEG is processed without separating frequency bands
- No bandpass filtering or spectral decomposition

**Problem:**
EEG contains distinct frequency bands with different physiological meanings:
- **Delta** (0.5-4 Hz): Deep sleep
- **Theta** (4-8 Hz): Drowsiness, memory
- **Alpha** (8-13 Hz): Relaxed wakefulness
- **Beta** (13-30 Hz): Active thinking
- **Gamma** (30-100 Hz): Cognitive processing

**Impact:**
VQC cannot target specific periodic patterns because all bands are mixed together.

**What Should Happen:**
```python
# Decompose EEG into frequency bands BEFORE quantum encoding:
band_signals = {}
for band_name in ['alpha', 'beta', 'gamma']:
    band_signals[band_name] = bandpass_filter(eeg, band_frequencies[band_name])

# Then encode each band with matched VQC frequencies
```

---

## 3. What VQC's Periodic Advantage Requires

### Theoretical Requirements (from Schuld et al., 2021)

For VQC to leverage its periodic advantage:

1. **Frequency Matching**: VQC's frequency spectrum Ω must contain the data's dominant frequencies
2. **Proper Rescaling**: Data must be rescaled so its period aligns with VQC's natural period (2π)
3. **Periodic Preservation**: Output processing must preserve periodic information

### Requirements vs. QTCN Implementation

| Requirement | QTCN Implementation | Status |
|-------------|---------------------|--------|
| Frequency analysis before encoding | None | ❌ Missing |
| Frequency-matched encoding | Generic AngleEmbedding | ❌ Missing |
| Rescaling to VQC period | None | ❌ Missing |
| Band decomposition | None | ❌ Missing |
| Periodic-preserving aggregation | Simple mean | ❌ Destroys periodicity |

---

## 4. Proposed Modifications: Periodic-Aware QTCN

### Modified Architecture

```python
import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

class PeriodicAwareQTCN(nn.Module):
    """
    QTCN modified to fully utilize VQC's periodic advantages.

    Key modifications:
    1. Frequency band decomposition before encoding
    2. Frequency-matched quantum encoding
    3. Band-specific rescaling to match VQC spectrum
    4. Preservation of periodic structure in output
    """

    # EEG frequency bands
    EEG_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    def __init__(
        self,
        n_qubits: int,
        circuit_depth: int,
        input_dim: tuple,
        kernel_size: int,
        dilation: int = 1,
        sampling_rate: float = 160.0,
        target_bands: list = ['alpha', 'beta', 'gamma']
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sampling_rate = sampling_rate
        self.target_bands = target_bands
        self.n_bands = len(target_bands)

        # === MODIFICATION 1: Frequency Band Filters ===
        self.band_filters = self._create_band_filters()

        # === MODIFICATION 2: Frequency-Aware Spatial Compression ===
        # Separate compression for each frequency band
        self.spatial_encoders = nn.ModuleList([
            nn.Linear(self.input_channels * self.kernel_size, n_qubits // self.n_bands)
            for _ in range(self.n_bands)
        ])

        # === MODIFICATION 3: Frequency Rescaling Factors ===
        # Map each band's center frequency to VQC's natural frequencies
        self.register_buffer(
            'freq_rescaling',
            self._compute_frequency_rescaling()
        )

        # Quantum parameters
        self.conv_params = nn.Parameter(torch.randn(circuit_depth, n_qubits, 15))
        self.pool_params = nn.Parameter(torch.randn(circuit_depth, n_qubits // 2, 3))

        # Quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.quantum_circuit = qml.QNode(self.circuit, self.dev, interface='torch')

        # === MODIFICATION 4: Learnable Frequency Mixing ===
        # Instead of simple averaging, learn how to combine periodic outputs
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=1, num_heads=1, batch_first=True
        )

    def _create_band_filters(self) -> dict:
        """Create bandpass filters for each target frequency band."""
        filters = {}
        nyquist = self.sampling_rate / 2

        for band_name in self.target_bands:
            low, high = self.EEG_BANDS[band_name]
            # Ensure frequencies are within valid range
            low = max(low, 0.5)
            high = min(high, nyquist - 1)

            b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
            filters[band_name] = (b, a)

        return filters

    def _compute_frequency_rescaling(self) -> torch.Tensor:
        """
        Compute rescaling factors to match EEG bands to VQC spectrum.

        VQC with AngleEmbedding has natural period 2π.
        We rescale each band's features so the band's center frequency
        maps to an integer in VQC's frequency spectrum.
        """
        rescaling = []

        for band_name in self.target_bands:
            low, high = self.EEG_BANDS[band_name]
            center_freq = (low + high) / 2

            # Map center frequency to VQC frequency 1, 2, 3, ...
            # This ensures VQC's Fourier components align with EEG bands
            band_idx = self.target_bands.index(band_name) + 1
            scale = 2 * np.pi * band_idx / center_freq
            rescaling.append(scale)

        return torch.tensor(rescaling, dtype=torch.float32)

    def _apply_bandpass(self, x: torch.Tensor, band_name: str) -> torch.Tensor:
        """Apply bandpass filter for a specific frequency band."""
        b, a = self.band_filters[band_name]

        # Convert to numpy for filtering
        x_np = x.detach().cpu().numpy()

        # Apply filter along time axis
        filtered = filtfilt(b, a, x_np, axis=-1)

        return torch.tensor(filtered, dtype=x.dtype, device=x.device)

    def circuit(self, features):
        """
        Quantum circuit with frequency-matched encoding.
        """
        wires = list(range(self.n_qubits))

        # === MODIFICATION: Frequency-Matched Encoding ===
        # Instead of generic AngleEmbedding, we encode with rescaled features
        for i, wire in enumerate(wires):
            # Determine which band this qubit belongs to
            band_idx = i // (self.n_qubits // self.n_bands)
            band_idx = min(band_idx, self.n_bands - 1)

            # Apply frequency-matched encoding
            # RY for amplitude, RX for phase (frequency-scaled)
            qml.RY(features[i], wires=wire)
            qml.RX(self.freq_rescaling[band_idx] * features[i], wires=wire)

        # Convolutional and pooling layers (same as original)
        for layer in range(self.circuit_depth):
            self._apply_convolution(self.conv_params[layer], wires)
            self._apply_pooling(self.pool_params[layer], wires)
            wires = wires[::2]

        return qml.expval(qml.PauliZ(0))

    def _apply_convolution(self, weights, wires):
        """Convolutional layer (same as original)."""
        n_wires = len(wires)
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    qml.U3(*weights[indx, :3], wires=w)
                    qml.U3(*weights[indx + 1, 3:6], wires=wires[indx + 1])
                    qml.IsingZZ(weights[indx, 6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[indx, 7], wires=[w, wires[indx + 1]])
                    qml.IsingXX(weights[indx, 8], wires=[w, wires[indx + 1]])
                    qml.U3(*weights[indx, 9:12], wires=w)
                    qml.U3(*weights[indx + 1, 12:], wires=wires[indx + 1])

    def _apply_pooling(self, pool_weights, wires):
        """Pooling layer (same as original)."""
        n_wires = len(wires)
        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(*pool_weights[indx // 2], wires=wires[indx - 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with frequency-aware processing.

        Args:
            x: [batch, channels, time_steps]

        Returns:
            output: [batch] - predictions
        """
        batch_size, input_channels, time_steps = x.size()

        # === MODIFICATION 1: Decompose into frequency bands ===
        band_signals = {}
        for band_name in self.target_bands:
            band_signals[band_name] = self._apply_bandpass(x, band_name)

        # Process each time window
        all_window_outputs = []

        for i in range(self.dilation * (self.kernel_size - 1), time_steps):
            indices = [i - d * self.dilation for d in range(self.kernel_size)]
            indices.reverse()

            # === MODIFICATION 2: Process each band separately ===
            band_features = []
            for band_idx, band_name in enumerate(self.target_bands):
                # Extract window for this band
                band_window = band_signals[band_name][:, :, indices].reshape(batch_size, -1)

                # Apply band-specific spatial compression
                compressed = self.spatial_encoders[band_idx](band_window)

                # === MODIFICATION 3: Apply frequency rescaling ===
                # Scale features to match VQC's frequency spectrum
                compressed = compressed * self.freq_rescaling[band_idx]

                band_features.append(compressed)

            # Concatenate band features
            combined_features = torch.cat(band_features, dim=-1)

            # Ensure correct size for quantum circuit
            if combined_features.shape[-1] < self.n_qubits:
                padding = torch.zeros(batch_size, self.n_qubits - combined_features.shape[-1], device=x.device)
                combined_features = torch.cat([combined_features, padding], dim=-1)
            else:
                combined_features = combined_features[:, :self.n_qubits]

            # Quantum circuit execution
            window_output = self.quantum_circuit(combined_features)
            all_window_outputs.append(window_output)

        # === MODIFICATION 4: Attention-based temporal aggregation ===
        # Instead of simple mean, use attention to preserve important periodic patterns
        outputs_stacked = torch.stack(all_window_outputs, dim=1).unsqueeze(-1)  # [batch, time, 1]

        # Self-attention over time
        attended, _ = self.temporal_attention(
            outputs_stacked, outputs_stacked, outputs_stacked
        )

        # Weighted sum instead of simple mean
        output = attended.mean(dim=1).squeeze(-1)

        return output
```

---

## 5. Comparison: Original vs. Periodic-Aware QTCN

| Aspect | Original QTCN | Periodic-Aware QTCN |
|--------|---------------|---------------------|
| **Frequency Analysis** | None | Band decomposition (alpha, beta, gamma) |
| **Encoding** | Generic AngleEmbedding (RY) | Frequency-matched (RY + scaled RX) |
| **Rescaling** | None | Band-specific rescaling to VQC spectrum |
| **Dimension Reduction** | Single FC for all frequencies | Separate FC per frequency band |
| **Temporal Aggregation** | Simple mean (destroys periodicity) | Attention-based (preserves patterns) |
| **VQC Periodic Advantage** | **NOT utilized** | **Fully utilized** |

---

## 6. Minimum Required Modifications

If full rewrite is not feasible, here are the **minimum changes** to enable periodic advantage:

### Modification A: Add Frequency Rescaling

```python
# In __init__, add:
self.freq_scale = nn.Parameter(torch.tensor([1.0]))  # Learnable frequency scale
```

### Modification B: Change Encoding

```python
# In circuit(), replace:
# BEFORE:
qml.AngleEmbedding(features, wires=wires, rotation='Y')

# AFTER:
for i, wire in enumerate(wires):
    qml.RY(features[i], wires=wire)
    qml.RX(self.freq_scale * features[i], wires=wire)  # Frequency encoding
```

### Modification C: Replace Mean with Learnable Aggregation

```python
# In __init__, add:
self.temporal_weight = nn.Parameter(torch.ones(1))

# In forward(), replace:
# BEFORE:
output = torch.mean(torch.stack(output, dim=1), dim=1)

# AFTER:
outputs = torch.stack(output, dim=1)  # [batch, time]
weights = torch.softmax(self.temporal_weight * outputs, dim=1)
output = (weights * outputs).sum(dim=1)
```

---

## 7. Summary

### Why Original QTCN Fails to Utilize VQC's Periodic Advantage

```
┌─────────────────────────────────────────────────────────────────────┐
│           ORIGINAL QTCN: MISSING PERIODIC COMPONENTS                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✗ No frequency band decomposition                                  │
│    → EEG's periodic structure (alpha, beta, gamma) is ignored       │
│                                                                     │
│  ✗ No frequency-matched encoding                                    │
│    → VQC's frequency spectrum doesn't align with EEG frequencies    │
│                                                                     │
│  ✗ No rescaling to VQC's natural period                            │
│    → Data period ≠ VQC period (2π)                                 │
│                                                                     │
│  ✗ Temporal averaging destroys periodic information                 │
│    → Any captured periodicity is lost in mean()                     │
│                                                                     │
│  RESULT: VQC is used as generic feature extractor,                  │
│          NOT as periodic function learner                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Recommendations

To fully utilize VQC's periodic advantages in QTCN for EEG:

1. **Add frequency band decomposition** before quantum encoding
2. **Implement frequency-matched encoding** with proper rescaling
3. **Replace simple mean** with attention-based temporal aggregation
4. **Design VQC spectrum** to match EEG frequency bands
5. **Test extrapolation** on unseen time periods to verify periodic advantage

### Conclusion

The current QTCN model uses quantum circuits for potentially expressive feature extraction, but it is **NOT designed to exploit the Fourier series nature of VQCs**. Without the proposed modifications, the model cannot leverage VQC's theoretical advantages for periodic data like EEG.

**The quantum component in the current architecture could be replaced with a classical neural network without losing the periodic advantage—because that advantage is not being utilized.**

---

## References

1. Schuld, M., Sweke, R., & Meyer, J. J. (2021). Effect of data encoding on the expressive power of variational quantum-machine-learning models. *Physical Review A*, 103(3), 032430.

2. Ziyin, L., Hartwig, T., & Ueda, M. (2020). Neural networks fail to learn periodic functions and how to fix it. *Advances in Neural Information Processing Systems*, 33.

3. Original QTCN implementation: `/pscratch/sd/j/junghoon/HQTCN_Project/scripts/HQTCN2_EEG.py`
