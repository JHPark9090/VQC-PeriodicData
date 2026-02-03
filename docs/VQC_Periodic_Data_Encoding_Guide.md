# VQC Encoding Strategies for Periodic Data

## A Practical Guide to Input Data Encoding for Variational Quantum Circuits

**Date**: February 2026
**Context**: Based on Schuld et al. (2021) and practical considerations for leveraging VQC's periodic structure.

---

## Executive Summary

To leverage VQC's theoretical advantages on periodic data, proper encoding is critical:

> **VQC output frequencies are determined by encoding Hamiltonian eigenvalues.**

The encoding must **match the VQC's frequency spectrum to the data's frequency content**.

---

## 1. Core Principle: Frequency Matching

### The Fundamental Relationship

From Schuld et al. (2021), a VQC computes:

$$f_{\text{VQC}}(x) = \sum_{\omega \in \Omega} c_\omega e^{i\omega x}$$

Where:
- $\Omega$ = frequency spectrum (determined by encoding)
- $c_\omega$ = Fourier coefficients (determined by trainable parameters)

### Key Insight

**If your data's dominant frequencies are NOT in $\Omega$, the VQC cannot represent them.**

| Scenario | Result |
|----------|--------|
| Data frequencies $\subseteq \Omega$ | VQC can learn exactly |
| Data frequencies $\not\subseteq \Omega$ | VQC can only approximate |

---

## 2. Encoding Strategy Framework

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENCODING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Analyze Data Frequencies                               │
│     └─→ FFT analysis to find dominant frequencies               │
│                                                                 │
│  Step 2: Design VQC Frequency Spectrum                          │
│     └─→ Choose encoding strategy to match data frequencies      │
│                                                                 │
│  Step 3: Compute Rescaling Factors                              │
│     └─→ x → αx to align periods                                 │
│                                                                 │
│  Step 4: Implement Encoding Circuit                             │
│     └─→ Build quantum circuit with proper gates                 │
│                                                                 │
│  Step 5: Validate on Extrapolation                              │
│     └─→ Test periodic extrapolation capability                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Data Characteristic | Recommended Encoding | Qubits Needed |
|---------------------|---------------------|---------------|
| Single dominant frequency | Single Pauli rotation + rescaling | 1 |
| Multiple known frequencies | Parallel encoding | 1 per frequency |
| Unknown frequency structure | Bandwidth extension + learnable rescaling | Depends on complexity |
| High-frequency content | More encoding repetitions | 1+ with deep circuit |
| Broadband signal | Hybrid classical-quantum preprocessing | Variable |

---

## 3. Encoding Methods

### Method 1: Data Rescaling

**Purpose**: Align data's natural period with VQC's natural period (2π)

**Mathematical Formula**:

$$x_{\text{encoded}} = \alpha \cdot x, \quad \text{where } \alpha = \frac{2\pi}{T_{\text{data}}}$$

**Example**: If data has period $T = 10$:

$$\alpha = \frac{2\pi}{10} = 0.628...$$

**Implementation**:

```python
import pennylane as qml
import numpy as np

# Data parameters
T_data = 10  # Data period
alpha = 2 * np.pi / T_data  # Rescaling factor

dev = qml.device('default.qubit', wires=1)

@qml.qnode(dev)
def vqc_with_rescaling(x, weights):
    """
    VQC with input rescaling for period matching.

    Args:
        x: Input data point
        weights: Trainable parameters [n_layers, 3]
    """
    # Step 1: Rescale input to match VQC period
    x_rescaled = alpha * x

    # Step 2: Encode rescaled data
    qml.RX(x_rescaled, wires=0)

    # Step 3: Trainable layers
    for layer_weights in weights:
        qml.Rot(*layer_weights, wires=0)

    return qml.expval(qml.PauliZ(0))
```

**When to Use**:
- Data has a single dominant period
- Period is known beforehand
- Simple periodic patterns

**Limitations**:
- Only aligns ONE frequency
- Other frequencies may be misaligned

---

### Method 2: Bandwidth Extension (Repeated Encoding)

**Purpose**: Access more frequencies for complex periodic patterns

**Mathematical Basis**:

With $r$ encoding repetitions (sequential or parallel):

$$\Omega = \{-r, -(r-1), ..., -1, 0, 1, ..., (r-1), r\}$$

This gives $2r + 1$ frequencies.

#### 2a. Sequential Repetition

```python
dev = qml.device('default.qubit', wires=1)

@qml.qnode(dev)
def vqc_sequential_encoding(x, weights, r=3):
    """
    Sequential encoding: same qubit, multiple layers.

    Frequency spectrum: {-r, ..., 0, ..., r}

    Args:
        x: Input data
        weights: Shape [r, 3] - rotation parameters per layer
        r: Number of encoding repetitions
    """
    for layer in range(r):
        # Data encoding
        qml.RX(x, wires=0)

        # Trainable block between encodings
        qml.Rot(*weights[layer], wires=0)

    return qml.expval(qml.PauliZ(0))
```

**Characteristics**:
- Circuit depth: $O(r)$
- Qubits needed: 1
- Barren plateau risk: Increases with $r$

#### 2b. Parallel Repetition

```python
def create_parallel_encoding_circuit(n_qubits):
    """
    Parallel encoding: multiple qubits, same layer.

    Frequency spectrum: {-n_qubits, ..., 0, ..., n_qubits}
    """
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def vqc_parallel_encoding(x, weights):
        """
        Args:
            x: Input data
            weights: Shape [n_qubits, 3] - rotation parameters per qubit
        """
        # Encode on all qubits in parallel
        for q in range(n_qubits):
            qml.RX(x, wires=q)

        # Entangling layer
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])

        # Trainable rotations
        for q in range(n_qubits):
            qml.Rot(*weights[q], wires=q)

        return qml.expval(qml.PauliZ(0))

    return vqc_parallel_encoding
```

**Characteristics**:
- Circuit depth: $O(1)$ for encoding
- Qubits needed: $r$
- Barren plateau risk: Lower than sequential

#### Comparison Table

| Aspect | Sequential | Parallel |
|--------|------------|----------|
| Frequencies | $2r + 1$ | $2r + 1$ |
| Circuit depth | $O(r)$ | $O(1)$ |
| Qubits | 1 | $r$ |
| Barren plateaus | Higher risk | Lower risk |
| Entanglement | Limited | Can be high |

---

### Method 3: Learnable Rescaling

**Purpose**: Let the model discover optimal frequency alignment (similar to Snake's learnable $a$)

**Implementation**:

```python
dev = qml.device('default.qubit', wires=3)

@qml.qnode(dev)
def vqc_learnable_rescaling(x, weights, scale_params):
    """
    VQC with learnable frequency scaling parameters.

    Args:
        x: Input data
        weights: Trainable rotation parameters
        scale_params: Learnable rescaling factors (like Snake's 'a')
    """
    n_qubits = 3

    # Each qubit gets a different learned frequency scaling
    for i in range(n_qubits):
        # Learnable rescaling: scale_params[i] is like Snake's 'a'
        x_scaled = scale_params[i] * x
        qml.RX(x_scaled, wires=i)

    # Entangling layer
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # Trainable rotations
    for i in range(n_qubits):
        qml.Rot(*weights[i], wires=i)

    return qml.expval(qml.PauliZ(0))

# Training: optimize both weights AND scale_params
# scale_params are initialized based on expected frequency range
initial_scale_params = np.array([1.0, 2.0, 4.0])  # Cover multiple frequencies
```

**Advantages**:
- Adapts to data automatically
- Can discover optimal frequencies
- More flexible than fixed encoding

**Disadvantages**:
- More parameters to optimize
- May be harder to train
- Risk of poor local minima

---

### Method 4: Multi-Frequency Targeted Encoding

**Purpose**: Directly encode at specific known frequencies

**Use Case**: When you know exactly which frequencies matter (e.g., EEG bands)

```python
def create_multi_frequency_circuit(target_frequencies):
    """
    Create VQC that targets specific frequencies.

    Args:
        target_frequencies: List of frequencies to capture [f1, f2, f3, ...]
    """
    n_qubits = len(target_frequencies)
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def vqc_multi_frequency(x, weights):
        """
        Each qubit encodes a different target frequency.
        """
        # Frequency-specific encoding
        for i, freq in enumerate(target_frequencies):
            # Encode at specific frequency
            qml.RX(2 * np.pi * freq * x, wires=i)

        # Entangling layer
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Trainable layer
        for i in range(n_qubits):
            qml.Rot(*weights[i], wires=i)

        return qml.expval(qml.PauliZ(0))

    return vqc_multi_frequency

# Example: Target EEG alpha (10 Hz) and beta (20 Hz)
target_freqs = [10, 20]  # Hz
circuit = create_multi_frequency_circuit(target_freqs)
```

---

## 4. Complete Encoding Pipeline Implementation

### Full Pipeline Class

```python
import numpy as np
import pennylane as qml
from scipy.fft import fft, fftfreq
from typing import List, Tuple, Optional

class PeriodicVQCEncoder:
    """
    Complete encoding pipeline for periodic data in VQCs.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        """
        Initialize encoder.

        Args:
            n_qubits: Number of qubits (determines max frequencies)
            n_layers: Number of variational layers
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.dominant_freqs = None
        self.rescaling_factors = None

    def analyze_data_frequencies(
        self,
        data: np.ndarray,
        sampling_rate: float,
        n_top_freqs: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1: Analyze frequency content of data.

        Args:
            data: Time series data
            sampling_rate: Sampling rate in Hz
            n_top_freqs: Number of top frequencies to extract

        Returns:
            dominant_freqs: Array of dominant frequencies
            spectrum: Full frequency spectrum
        """
        if n_top_freqs is None:
            n_top_freqs = self.n_qubits

        n = len(data)
        frequencies = fftfreq(n, 1/sampling_rate)
        spectrum = np.abs(fft(data))

        # Only consider positive frequencies
        positive_mask = frequencies > 0
        pos_freqs = frequencies[positive_mask]
        pos_spectrum = spectrum[positive_mask]

        # Find top frequencies
        top_indices = np.argsort(pos_spectrum)[-n_top_freqs:]
        self.dominant_freqs = np.sort(pos_freqs[top_indices])

        print(f"Detected dominant frequencies: {self.dominant_freqs} Hz")

        return self.dominant_freqs, spectrum

    def compute_rescaling_factors(
        self,
        method: str = 'integer_mapping'
    ) -> np.ndarray:
        """
        Step 2: Compute rescaling factors for frequency matching.

        Args:
            method: 'integer_mapping' or 'direct'

        Returns:
            rescaling_factors: Array of rescaling factors per qubit
        """
        if self.dominant_freqs is None:
            raise ValueError("Run analyze_data_frequencies first!")

        if method == 'integer_mapping':
            # Map frequencies to integer VQC spectrum
            base_freq = np.min(self.dominant_freqs)
            self.rescaling_factors = 2 * np.pi * self.dominant_freqs / base_freq
        elif method == 'direct':
            # Direct frequency encoding
            self.rescaling_factors = 2 * np.pi * self.dominant_freqs
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"Rescaling factors: {self.rescaling_factors}")

        return self.rescaling_factors

    def build_circuit(self):
        """
        Step 3: Build the VQC with proper encoding.

        Returns:
            QNode: Compiled quantum circuit
        """
        if self.rescaling_factors is None:
            raise ValueError("Run compute_rescaling_factors first!")

        rescaling = self.rescaling_factors
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev)
        def circuit(x, weights):
            """
            VQC with frequency-matched encoding.

            Args:
                x: Input data point
                weights: Shape [n_layers, n_qubits, 3]
            """
            # === ENCODING LAYER ===
            # Multi-frequency encoding
            for i in range(n_qubits):
                qml.RX(rescaling[i] * x, wires=i)

            # === VARIATIONAL LAYERS ===
            for layer in range(n_layers):
                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Circular entanglement
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])

                # Rotation layer
                for i in range(n_qubits):
                    qml.Rot(*weights[layer, i], wires=i)

            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        return circuit

    def preprocess_data(
        self,
        data: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Step 4: Preprocess data for VQC input.

        Args:
            data: Raw input data
            normalize: Whether to normalize

        Returns:
            preprocessed: Preprocessed data
        """
        if normalize:
            # Z-score normalization
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)

        return data

    def get_parameter_shape(self) -> Tuple[int, int, int]:
        """Get shape of trainable parameters."""
        return (self.n_layers, self.n_qubits, 3)

    def initialize_weights(self, seed: int = 42) -> np.ndarray:
        """Initialize trainable weights."""
        np.random.seed(seed)
        shape = self.get_parameter_shape()
        return np.random.uniform(-np.pi, np.pi, size=shape)
```

### Usage Example

```python
# Example: Process synthetic periodic data
import numpy as np

# Generate synthetic periodic data
t = np.linspace(0, 10, 1000)
sampling_rate = 100  # Hz
# Signal with 5 Hz and 12 Hz components
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)

# Initialize encoder
encoder = PeriodicVQCEncoder(n_qubits=2, n_layers=3)

# Step 1: Analyze frequencies
dominant_freqs, spectrum = encoder.analyze_data_frequencies(signal, sampling_rate)
# Output: Detected dominant frequencies: [ 5. 12.] Hz

# Step 2: Compute rescaling
rescaling = encoder.compute_rescaling_factors(method='integer_mapping')
# Output: Rescaling factors: [6.28... 15.08...]

# Step 3: Build circuit
circuit = encoder.build_circuit()

# Step 4: Preprocess data
signal_preprocessed = encoder.preprocess_data(signal)

# Initialize weights
weights = encoder.initialize_weights()

# Forward pass
output = circuit(signal_preprocessed[0], weights)
print(f"VQC output: {output}")
```

---

## 5. Domain-Specific Encoding Examples

### Example 1: EEG Data Encoding

```python
# EEG frequency bands
EEG_BANDS = {
    'delta': (0.5, 4),    # Hz
    'theta': (4, 8),      # Hz
    'alpha': (8, 13),     # Hz
    'beta': (13, 30),     # Hz
    'gamma': (30, 100)    # Hz
}

def create_eeg_encoder(target_bands: List[str] = ['alpha', 'beta']):
    """
    Create VQC encoder optimized for EEG frequency bands.

    Args:
        target_bands: List of EEG bands to target

    Returns:
        encoder: Configured PeriodicVQCEncoder
        circuit: QNode circuit
    """
    # Extract center frequencies for target bands
    target_freqs = [np.mean(EEG_BANDS[band]) for band in target_bands]
    n_qubits = len(target_bands)

    # Create encoder
    encoder = PeriodicVQCEncoder(n_qubits=n_qubits, n_layers=3)

    # Manually set frequencies (we know them for EEG)
    encoder.dominant_freqs = np.array(target_freqs)
    encoder.compute_rescaling_factors(method='integer_mapping')

    circuit = encoder.build_circuit()

    print(f"EEG Encoder configured for bands: {target_bands}")
    print(f"Target frequencies: {target_freqs} Hz")

    return encoder, circuit

# Usage
encoder, circuit = create_eeg_encoder(['alpha', 'beta', 'gamma'])
# Output:
# EEG Encoder configured for bands: ['alpha', 'beta', 'gamma']
# Target frequencies: [10.5, 21.5, 65.0] Hz
```

### Example 2: Stock Price Encoding (Multiple Periodicities)

```python
# Stock price typical periodicities
STOCK_PERIODS = {
    'daily': 1,           # 1 day
    'weekly': 5,          # 5 trading days
    'monthly': 21,        # ~21 trading days
    'quarterly': 63,      # ~63 trading days
    'yearly': 252         # ~252 trading days
}

def create_stock_encoder(target_periods: List[str] = ['weekly', 'monthly']):
    """
    Create VQC encoder for stock price periodicities.

    Args:
        target_periods: List of periodicities to capture

    Returns:
        encoder: Configured encoder
        circuit: QNode circuit
    """
    # Convert periods to frequencies (cycles per day)
    target_freqs = [1.0 / STOCK_PERIODS[period] for period in target_periods]
    n_qubits = len(target_periods)

    encoder = PeriodicVQCEncoder(n_qubits=n_qubits, n_layers=4)
    encoder.dominant_freqs = np.array(target_freqs)
    encoder.compute_rescaling_factors(method='direct')

    circuit = encoder.build_circuit()

    print(f"Stock Encoder configured for periods: {target_periods}")
    print(f"Corresponding frequencies: {target_freqs} cycles/day")

    return encoder, circuit

# Usage
encoder, circuit = create_stock_encoder(['weekly', 'monthly', 'quarterly'])
```

### Example 3: Audio Signal Encoding

```python
def create_audio_encoder(
    audio_signal: np.ndarray,
    sample_rate: int = 44100,
    n_harmonics: int = 4
):
    """
    Create VQC encoder for audio signals.

    Automatically detects fundamental frequency and harmonics.

    Args:
        audio_signal: Audio waveform
        sample_rate: Audio sample rate
        n_harmonics: Number of harmonics to capture

    Returns:
        encoder: Configured encoder
        fundamental_freq: Detected fundamental frequency
    """
    encoder = PeriodicVQCEncoder(n_qubits=n_harmonics, n_layers=3)

    # Analyze to find fundamental frequency
    dominant_freqs, _ = encoder.analyze_data_frequencies(
        audio_signal,
        sample_rate,
        n_top_freqs=1
    )
    fundamental = dominant_freqs[0]

    # Set harmonics: f, 2f, 3f, 4f, ...
    harmonics = fundamental * np.arange(1, n_harmonics + 1)
    encoder.dominant_freqs = harmonics
    encoder.compute_rescaling_factors(method='integer_mapping')

    circuit = encoder.build_circuit()

    print(f"Audio Encoder: Fundamental = {fundamental:.2f} Hz")
    print(f"Harmonics: {harmonics}")

    return encoder, circuit, fundamental
```

---

## 6. Encoding Strategy Comparison

### Summary Table

| Strategy | Complexity | Flexibility | Best For |
|----------|------------|-------------|----------|
| **Simple rescaling** | Low | Low | Single known frequency |
| **Sequential repetition** | Medium | Medium | Unknown frequencies, limited qubits |
| **Parallel repetition** | Medium | Medium | Multiple frequencies, many qubits |
| **Learnable rescaling** | High | High | Complex unknown patterns |
| **Multi-frequency targeted** | Medium | Low | Known frequency bands (EEG, audio) |

### Frequency Coverage vs. Resources

| Encoding | Frequencies | Qubits | Depth | Barren Plateau Risk |
|----------|-------------|--------|-------|---------------------|
| 1 Pauli rotation | 3 | 1 | 1 | Low |
| 3 sequential | 7 | 1 | 3 | Medium |
| 5 sequential | 11 | 1 | 5 | High |
| 3 parallel | 7 | 3 | 1 | Low |
| 5 parallel | 11 | 5 | 1 | Low |
| Learnable (3 qubits) | Continuous | 3 | 1 | Medium |

---

## 7. Best Practices

### Do's

1. **Always analyze data frequencies first**
   ```python
   # Good: Analyze before encoding
   freqs, spectrum = encoder.analyze_data_frequencies(data, sample_rate)
   ```

2. **Match VQC spectrum to data frequencies**
   ```python
   # Good: Design encoding based on analysis
   if max_data_freq > current_vqc_spectrum:
       increase_encoding_repetitions()
   ```

3. **Use rescaling to align periods**
   ```python
   # Good: Rescale data
   x_encoded = (2 * np.pi / T_data) * x
   ```

4. **Start simple, add complexity if needed**
   ```python
   # Good: Start with minimal encoding
   # Add repetitions only if validation shows need
   ```

5. **Validate extrapolation capability**
   ```python
   # Good: Test on data outside training interval
   train_interval = [0, T]
   test_interval = [T, 2*T]  # Extrapolation test
   ```

### Don'ts

1. **Don't ignore frequency mismatch**
   ```python
   # Bad: Using raw data without checking frequencies
   qml.RX(x, wires=0)  # May miss important frequencies!
   ```

2. **Don't over-extend bandwidth unnecessarily**
   ```python
   # Bad: Too many repetitions without need
   for _ in range(20):  # Excessive! Barren plateaus!
       qml.RX(x, wires=0)
   ```

3. **Don't forget data normalization**
   ```python
   # Bad: Raw data with large values
   qml.RX(1000 * x, wires=0)  # Numerical issues!

   # Good: Normalize first
   x_norm = (x - mean) / std
   qml.RX(x_norm, wires=0)
   ```

4. **Don't use fixed encoding for unknown frequency data**
   ```python
   # Bad: Assuming frequencies without analysis
   qml.RX(10 * x, wires=0)  # Why 10?

   # Good: Analyze or use learnable scaling
   ```

---

## 8. Troubleshooting Guide

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Poor training convergence | Frequency mismatch | Analyze data frequencies, adjust encoding |
| Good training, bad extrapolation | Encoding doesn't cover all frequencies | Increase bandwidth (more repetitions) |
| Barren plateaus | Too many sequential repetitions | Switch to parallel encoding |
| Numerical instability | Data not normalized | Preprocess data to reasonable range |
| VQC learns constant | Single encoding, wrong frequency | Add repetitions or rescale data |

---

## 9. Summary

### The Golden Rules of VQC Encoding for Periodic Data

1. **Frequency Matching is Critical**
   - VQC can only learn frequencies in its spectrum
   - Design encoding to include data's dominant frequencies

2. **Rescaling Aligns Periods**
   - Use $x \rightarrow \alpha x$ where $\alpha = 2\pi / T_{\text{data}}$
   - Multiple frequencies may need multiple rescaling factors

3. **Bandwidth Extension Adds Frequencies**
   - $r$ repetitions give $2r + 1$ frequencies
   - Parallel preferred over sequential (fewer barren plateaus)

4. **Learnable Parameters Add Flexibility**
   - When frequencies are unknown, use learnable rescaling
   - Similar to Snake's learnable $a$ parameter

5. **Validate on Extrapolation**
   - VQC's advantage is periodic extrapolation
   - Always test outside training interval

---

## References

1. Schuld, M., Sweke, R., & Meyer, J. J. (2021). Effect of data encoding on the expressive power of variational quantum-machine-learning models. *Physical Review A*, 103(3), 032430.

2. Pérez-Salinas, A., et al. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.

3. LaRose, R., & Coyle, B. (2020). Robust data encodings for quantum classifiers. *Physical Review A*, 102(3), 032420.

4. Havlíček, V., et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567(7747), 209-212.
