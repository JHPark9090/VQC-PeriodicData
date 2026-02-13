# Benchmark Datasets for LSTM Experiments

This document describes the four time-series benchmarks used in the VQC periodic advantage experiments. Each dataset tests a different aspect of periodic modeling, allowing us to isolate where VQC's native Fourier series structure provides genuine benefit.

---

## 1. NARMA-10 (Baseline)

**File:** `data/narma_generator.py`

### Description

NARMA (Nonlinear AutoRegressive Moving Average) is a standard benchmark for recurrent models and reservoir computing. NARMA-10 is defined by:

$$y(t) = 0.3 \cdot y(t{-}1) + 0.05 \cdot y(t{-}1) \cdot \sum_{i=0}^{9} y(t{-}1{-}i) + 1.5 \cdot u(t{-}10) \cdot u(t{-}1) + 0.1$$

where $u(t) \sim \mathcal{U}(0, 0.5)$ is the driving input.

### Why it matters

- **Nonlinear memory:** The 10-step summation term requires short-term memory; the $u(t{-}10)$ term requires longer-range recall.
- **Broadband spectrum:** NARMA-10 has no dominant periodic structure — its frequency content is spread across the spectrum due to the stochastic input.
- **Role in this study:** Serves as the control benchmark. If VQC outperforms classical baselines on NARMA-10, the advantage is not purely due to periodicity.

### Data flow

Raw NARMA series &rarr; MinMaxScaler to [-1, 1] &rarr; sliding window of length `seq_len` &rarr; predict next value.

### Usage

```bash
python models/ReLU_LSTM.py --dataset=narma --narma-order=10
```

---

## 2. Multi-Sine (K=5)

**File:** `data/multisine_generator.py`

### Description

A superposition of K=5 sinusoids with incommensurate frequencies:

$$y(t) = \sum_{k=1}^{5} a_k \cdot \sin(2\pi f_k \cdot t + \phi_k) + \varepsilon(t)$$

| Component | Frequency ($f_k$) | Amplitude ($a_k$) | Phase ($\phi_k$) |
|-----------|-------------------|-------------------|------------------|
| 1         | 0.10              | 1.0               | 0                |
| 2         | 0.23              | 0.8               | $\pi/4$          |
| 3         | 0.37              | 0.6               | $\pi/3$          |
| 4         | 0.51              | 0.4               | $\pi/6$          |
| 5         | 0.79              | 0.2               | $\pi/2$          |

Noise: $\varepsilon(t) \sim \mathcal{N}(0, 0.01)$

### Why it matters

- **Purest Fourier test:** The target function IS a finite Fourier series. This is the most direct test of the Schuld et al. (2021) theory that VQCs compute truncated Fourier series.
- **Incommensurate frequencies:** The frequencies are chosen to avoid harmonic relationships, so the model cannot exploit simple integer-ratio shortcuts.
- **Decreasing amplitudes:** Higher-frequency components have smaller amplitudes, testing whether the model can resolve weak high-frequency signals.
- **Expected outcome:** VQC should have a strong advantage here. A VQC with sufficient data-encoding repetitions can exactly represent this function, while ReLU networks must approximate periodic structure through piecewise-linear composition.

### Data flow

Same as NARMA: generate series &rarr; normalize to [-1, 1] &rarr; sliding window &rarr; predict next value.

### Usage

```bash
python models/ReLU_LSTM.py --dataset=multisine
```

---

## 3. Mackey-Glass ($\tau=17$)

**File:** `data/mackey_glass_generator.py`

### Description

The Mackey-Glass system is a delay differential equation originally proposed to model physiological control systems (blood cell regulation):

$$\frac{dx}{dt} = \frac{\beta \cdot x(t-\tau)}{1 + x(t-\tau)^n} - \gamma \cdot x(t)$$

with standard parameters $\beta=0.2$, $\gamma=0.1$, $n=10$.

The delay parameter $\tau$ controls the dynamical regime:

| $\tau$ | Behavior        | Largest Lyapunov Exponent |
|--------|-----------------|--------------------------|
| < 17   | Periodic        | $\leq 0$                 |
| 17     | Quasi-periodic  | $\approx 0$              |
| 30     | Chaotic         | $> 0$                    |

We use $\tau=17$ (quasi-periodic regime) as the default.

### Why it matters

- **Standard quantum reservoir benchmark:** Used by Fujii & Nakajima (2017) and subsequent quantum reservoir computing papers, enabling direct comparison with published results.
- **Quasi-periodic structure:** At $\tau=17$, the system oscillates with a dominant period near $1/\tau$ but is not strictly periodic. This tests whether VQC's Fourier structure helps with approximately-periodic signals.
- **Deterministic chaos:** Unlike NARMA (which is stochastic), Mackey-Glass is deterministic. Any prediction error comes from model capacity limitations, not irreducible noise.
- **Expected outcome:** VQC should have a moderate advantage. The quasi-periodic structure partially aligns with VQC's Fourier basis, but the nonlinear distortion from the $x^n$ term introduces non-periodic components.

### Data flow

Euler integration with 500-step warmup &rarr; normalize to [-1, 1] &rarr; sliding window &rarr; predict next value.

### Usage

```bash
python models/ReLU_LSTM.py --dataset=mackey_glass
# Chaotic regime (harder):
python models/ReLU_LSTM.py --dataset=mackey_glass  # modify tau in code for tau=30
```

---

## 4. Adding Problem

**File:** `data/adding_problem_generator.py`

### Description

The Adding Problem (Hochreiter & Schmidhuber, 1997) is a long-range dependency benchmark where each sample is an independent sequence:

1. **Signal:** $s_i \sim \mathcal{U}(0, 1)$ for $i = 1, \ldots, T$
2. **Mask:** $m_i \in \{0, 1\}$, exactly 2 positions set to 1 — one in the first half $[1, T/2)$, one in the second half $[T/2, T]$
3. **Target:** $y = s_{j_1} + s_{j_2}$ where $j_1, j_2$ are the marked positions
4. **Input:** $\mathbf{x} = [s_1, \ldots, s_T, m_1, \ldots, m_T]$ (concatenated, length $2T$)

This is a **regression task** (predict the sum) and a **sequence-to-scalar** task (NOT sliding-window prediction).

### Why it matters

- **Long-range selective memory:** The model must learn to attend to exactly 2 out of T positions, ignoring all others. This is fundamentally different from the smooth temporal dependencies in NARMA and Mackey-Glass.
- **No periodic structure in the data:** The signal is i.i.d. uniform noise with no temporal correlation. This is the negative control — if VQC shows advantage here, it is NOT due to periodicity.
- **Difficulty scales with T:** Larger T means the model must maintain information across longer sequences, with more distractor values to ignore.
- **Expected outcome:** VQC should have minimal or no advantage. The data has no periodic structure, so VQC's Fourier basis should not help. Any advantage observed would require a different explanation (e.g., gate expressivity).

### Data flow

Generate n independent sequences &rarr; normalize targets to [-1, 1] &rarr; input shape is [n_samples, 2*T].

### Important implementation note

Unlike the other datasets, the Adding Problem does NOT use sliding windows. Each sample is independently generated. When using `--window-size=T`, the actual input length is $2T$ (signal + mask concatenated). The model processes the full $2T$-length input.

### Usage

```bash
python models/ReLU_LSTM.py --dataset=adding --window-size=50
# Harder version (longer sequences):
python models/ReLU_LSTM.py --dataset=adding --window-size=100
```

---

## Summary Table

| Dataset       | Type                | Periodic Content      | Memory Requirement | VQC Advantage Expected |
|---------------|---------------------|-----------------------|--------------------|------------------------|
| NARMA-10      | Stochastic nonlinear| Broadband (none)      | 10 steps           | Weak (control)         |
| Multi-Sine    | Deterministic       | Exact Fourier (K=5)   | ~1/min(f_k) steps  | Strong                 |
| Mackey-Glass  | Deterministic chaos | Quasi-periodic        | ~$\tau$ steps      | Moderate               |
| Adding Problem| i.i.d. random       | None                  | T steps (selective) | None (negative control)|

## Common Interface

All datasets follow the same API pattern:

```python
from data.multisine_generator import get_multisine_data
from data.mackey_glass_generator import get_mackey_glass_data
from data.adding_problem_generator import get_adding_data

# Returns (x: [n_seq, seq_len], y: [n_seq]) as torch.float32
x, y = get_multisine_data(K=5, seq_len=8, n_samples=500, seed=2025)
x, y = get_mackey_glass_data(tau=17, seq_len=8, n_samples=500, seed=2025)
x, y = get_adding_data(T=50, n_samples=500, seed=2025)  # x shape: [500, 100]
```

All models accept `--dataset={narma, multisine, mackey_glass, adding}` to select the benchmark.

## References

- Schuld, M., Sweke, R., & Meyer, J.J. (2021). Effect of data encoding on the expressive power of variational quantum machine learning models. *Physical Review A*, 103(3), 032430.
- Fujii, K. & Nakajima, K. (2017). Harnessing disordered-ensemble quantum dynamics for machine learning. *Physical Review Applied*, 8(2), 024030.
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Ziyin, L., Hartwig, T., & Ueda, M. (2020). Neural networks fail to learn periodic functions and how to fix it. *NeurIPS 2020*.
- Mackey, M.C. & Glass, L. (1977). Oscillation and chaos in physiological control systems. *Science*, 197(4300), 287-289.
