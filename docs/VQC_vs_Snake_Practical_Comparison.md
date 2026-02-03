# Practical Comparison: VQCs vs Snake Activation for Real-World Periodic Data

## Can VQCs Outperform Snake-Based Deep Learning for EEG, fMRI, and Stock Price Extrapolation?

**Author**: Analysis based on Ziyin et al. (NeurIPS 2020) and Schuld et al. (Physical Review A, 2021)

**Date**: February 2026

---

## Executive Summary

**Question**: Given that VQCs can achieve universal extrapolation for periodic functions (as proven in the companion document `VQC_Universal_Extrapolation_Analysis.md`), will VQCs show better extrapolation performance than deep learning models with Snake activation when training on long-term, high-dimensional, spatio-temporal periodic data such as EEG, fMRI, and stock prices?

**Answer**: **No, not in general, and likely not for most real-world cases.**

The theoretical universal extrapolation capability of VQCs does **not** translate to practical superiority because:

1. Real-world data (EEG, fMRI, stocks) contains essential **non-periodic components** that VQCs fundamentally cannot model
2. Snake activation is explicitly designed to handle **both periodic and non-periodic** signals
3. VQCs face significant **practical limitations** (trainability, hardware constraints, scalability)
4. The theoretical result guarantees **capability**, not **superiority**

---

## Table of Contents

1. [The Critical Distinction: Theoretical Capability vs Practical Performance](#1-the-critical-distinction)
2. [Analysis of Real-World Data Characteristics](#2-analysis-of-real-world-data-characteristics)
3. [Fundamental Limitation: VQCs Cannot Model Non-Periodic Components](#3-fundamental-limitation)
4. [Empirical Evidence from the Original Papers](#4-empirical-evidence-from-the-original-papers)
5. [Practical Limitations of VQCs](#5-practical-limitations-of-vqcs)
6. [Detailed Comparison for Each Data Type](#6-detailed-comparison-for-each-data-type)
7. [When Would VQCs Actually Outperform Snake?](#7-when-would-vqcs-actually-outperform-snake)
8. [Recommendations for Research and Applications](#8-recommendations)
9. [Conclusion](#9-conclusion)

---

## 1. The Critical Distinction: Theoretical Capability vs Practical Performance {#1-the-critical-distinction}

### 1.1 What the Universal Extrapolation Theorem Actually Says

From our analysis in `VQC_Universal_Extrapolation_Analysis.md`, we proved:

> **Theorem (VQC Universal Periodic Extrapolation)**: A VQC with $r$ repeated Pauli-rotation encodings can uniformly approximate any piecewise $C^1$ periodic function $f: \mathbb{R} \to \mathbb{R}$ with period $2\pi$ to arbitrary precision $\varepsilon > 0$ on the entire real line.

### 1.2 What This Theorem Does NOT Say

The theorem does **NOT** claim:

| Incorrect Interpretation | Why It's Wrong |
|--------------------------|----------------|
| "VQCs will outperform other methods on periodic data" | Capability ≠ superiority; other methods may also approximate periodic functions |
| "VQCs are optimal for periodic extrapolation" | No optimality claim is made; only existence of approximation |
| "VQCs work well on real-world periodic data" | Real data is not purely periodic; theorem assumes pure periodicity |
| "VQCs are easier to train than alternatives" | Trainability is not addressed; only expressivity is proven |
| "VQCs will generalize better" | Generalization depends on many factors beyond expressivity |

### 1.3 The Logical Gap

```
Theoretical Result:
    "VQCs CAN approximate any periodic function"

         ↓ DOES NOT IMPLY ↓

Practical Claim:
    "VQCs WILL perform better on real periodic data"
```

**Reason for the gap**: Real-world data violates the theorem's assumptions (pure periodicity), and practical constraints limit VQC implementation.

---

## 2. Analysis of Real-World Data Characteristics {#2-analysis-of-real-world-data-characteristics}

### 2.1 EEG (Electroencephalography)

#### 2.1.1 Periodic Components

| Rhythm | Frequency Range | Origin | Periodicity |
|--------|-----------------|--------|-------------|
| Delta | 0.5-4 Hz | Deep sleep, pathology | Quasi-periodic |
| Theta | 4-8 Hz | Drowsiness, memory | Quasi-periodic |
| Alpha | 8-12 Hz | Relaxed wakefulness | Relatively stable |
| Beta | 12-30 Hz | Active thinking, focus | Variable |
| Gamma | 30-100 Hz | Cognitive processing | Highly variable |

#### 2.1.2 Non-Periodic Components (Critical for VQC Limitations)

| Component | Nature | Can VQC Model? |
|-----------|--------|----------------|
| **Baseline drift** | Slow, non-periodic trend | **No** |
| **Motion artifacts** | Transient, non-periodic | **No** |
| **Eye blinks/movements** | Sporadic, non-periodic | **No** |
| **Event-related potentials (ERPs)** | Transient responses to stimuli | **No** |
| **Epileptic spikes** | Paroxysmal, non-periodic | **No** |
| **Muscle artifacts (EMG)** | Broadband, non-periodic | **No** |
| **Non-stationary dynamics** | Time-varying frequency content | **Partially** |

#### 2.1.3 Mathematical Representation of Real EEG

Real EEG signal can be decomposed as:

$$x_{\text{EEG}}(t) = \underbrace{\sum_{k} A_k(t) \sin(2\pi f_k t + \phi_k(t))}_{\text{Quasi-periodic oscillations}} + \underbrace{d(t)}_{\text{Drift/trend}} + \underbrace{\sum_i s_i(t)}_{\text{Transient events}} + \underbrace{n(t)}_{\text{Noise}}$$

**Key observation**: The amplitudes $A_k(t)$ and phases $\phi_k(t)$ are **time-varying** (non-stationary), and drift $d(t)$, transients $s_i(t)$ are **non-periodic**.

VQC output is restricted to:

$$f_{\text{VQC}}(t) = \sum_{n=-r}^{r} c_n e^{i n \omega_0 t}$$

with **constant** coefficients $c_n$. This cannot capture time-varying amplitudes, drifts, or transients.

### 2.2 fMRI (Functional Magnetic Resonance Imaging)

#### 2.2.1 Periodic Components

| Source | Frequency | Nature |
|--------|-----------|--------|
| Cardiac pulsation | ~1 Hz (60-80 bpm) | Periodic physiological noise |
| Respiration | ~0.2-0.3 Hz (12-18 breaths/min) | Periodic physiological noise |
| Mayer waves | ~0.1 Hz | Quasi-periodic blood pressure oscillations |
| Low-frequency oscillations | 0.01-0.1 Hz | Resting-state networks (quasi-periodic) |

#### 2.2.2 Non-Periodic Components (Critical)

| Component | Nature | Can VQC Model? |
|-----------|--------|----------------|
| **BOLD hemodynamic response** | Transient response to neural activity | **No** |
| **Scanner drift** | Slow, non-periodic hardware artifact | **No** |
| **Motion artifacts** | Sudden head movements | **No** |
| **Task-evoked responses** | Event-related, non-periodic | **No** |
| **Spontaneous neural fluctuations** | Aperiodic broadband activity | **No** |

#### 2.2.3 The Hemodynamic Response Function (HRF)

The BOLD signal is modeled as:

$$y(t) = (s * h)(t) + \text{noise} = \int_{0}^{\infty} s(\tau) h(t - \tau) d\tau + n(t)$$

where:
- $s(t)$ = stimulus function (event timings)
- $h(t)$ = hemodynamic response function (typically a gamma-variate or double-gamma)

The HRF is **not periodic**—it's a transient response that rises, peaks around 5-6 seconds, and returns to baseline over ~20 seconds.

**VQCs cannot model the HRF** because it's fundamentally non-periodic.

### 2.3 Stock Prices and Financial Time Series

#### 2.3.1 Quasi-Periodic Components

| Cycle | Period | Nature |
|-------|--------|--------|
| Kitchin cycle | 3-5 years | Inventory cycles |
| Juglar cycle | 7-11 years | Investment cycles |
| Kuznets cycle | 15-25 years | Infrastructure/demographic |
| Kondratieff wave | 40-60 years | Technological paradigm shifts |
| Seasonal patterns | 1 year | Holiday effects, earnings seasons |
| Weekly patterns | 1 week | Weekend effects |
| Intraday patterns | 1 day | Opening/closing effects |

#### 2.3.2 Non-Periodic Components (Dominant in Practice)

| Component | Nature | Can VQC Model? | Importance |
|-----------|--------|----------------|------------|
| **Random walk / Brownian motion** | Stochastic trend | **No** | **Dominant** |
| **News/event shocks** | Sudden jumps | **No** | High |
| **Structural breaks** | Regime changes | **No** | High |
| **Long-term growth trend** | Non-periodic drift | **No** | High |
| **Volatility clustering** | Time-varying variance | **No** | High |
| **Fat tails / extreme events** | Non-Gaussian shocks | **No** | Critical for risk |

#### 2.3.3 Mathematical Model: Stock Price Dynamics

The standard model for stock prices is Geometric Brownian Motion (GBM):

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

or in discrete form:

$$\log(S_{t+1}/S_t) = \mu + \sigma \epsilon_t, \quad \epsilon_t \sim N(0,1)$$

**Key insight**: The core dynamics are a **random walk with drift**—fundamentally non-periodic.

Realistic models add periodic components:

$$\log(S_t) = \underbrace{\mu t}_{\text{Trend}} + \underbrace{\sum_k A_k \sin(2\pi f_k t + \phi_k)}_{\text{Cycles (weak)}} + \underbrace{\sigma W_t}_{\text{Random walk (dominant)}} + \underbrace{\text{jumps}}_{\text{Shocks}}$$

The periodic cycles are **weak** relative to the random walk component. VQCs can only model the periodic part, missing the dominant stochastic trend.

---

## 3. Fundamental Limitation: VQCs Cannot Model Non-Periodic Components {#3-fundamental-limitation}

### 3.1 Mathematical Proof of Limitation

**Proposition 3.1** (VQC Output is Strictly Periodic). Let $f_{\text{VQC}}: \mathbb{R} \to \mathbb{R}$ be the output of a VQC with integer-valued frequency spectrum $\Omega \subseteq \mathbb{Z}$. Then:

$$f_{\text{VQC}}(x + 2\pi) = f_{\text{VQC}}(x) \quad \forall x \in \mathbb{R}$$

*Proof*: By definition, $f_{\text{VQC}}(x) = \sum_{\omega \in \Omega} c_\omega e^{i\omega x}$. For any $\omega \in \mathbb{Z}$:

$$e^{i\omega(x + 2\pi)} = e^{i\omega x} \cdot e^{i\omega \cdot 2\pi} = e^{i\omega x} \cdot 1 = e^{i\omega x}$$

Therefore, $f_{\text{VQC}}(x + 2\pi) = \sum_{\omega \in \Omega} c_\omega e^{i\omega(x+2\pi)} = \sum_{\omega \in \Omega} c_\omega e^{i\omega x} = f_{\text{VQC}}(x)$. $\square$

**Corollary 3.2** (VQC Cannot Approximate Non-Periodic Functions). Let $g: \mathbb{R} \to \mathbb{R}$ be a non-periodic function (i.e., there exists no $T > 0$ such that $g(x+T) = g(x)$ for all $x$). Then for any VQC output $f_{\text{VQC}}$:

$$\sup_{x \in \mathbb{R}} |f_{\text{VQC}}(x) - g(x)| = \infty$$

*Proof*: Since $f_{\text{VQC}}$ is $2\pi$-periodic, $f_{\text{VQC}}$ is bounded: $|f_{\text{VQC}}(x)| \leq \sum_{\omega} |c_\omega| < \infty$.

If $g$ is non-periodic, then either:
- $g$ is unbounded (e.g., linear trend), so $|g(x)| \to \infty$ as $|x| \to \infty$, or
- $g$ is bounded but not periodic, so $\exists \varepsilon > 0, \forall T > 0, \exists x: |g(x+T) - g(x)| > \varepsilon$

In the first case, $|f_{\text{VQC}}(x) - g(x)| \to \infty$ as $|x| \to \infty$.

In the second case, consider $x_n = 2\pi n$ for $n \to \infty$. We have $f_{\text{VQC}}(x_n) = f_{\text{VQC}}(0)$ (constant), but $g(x_n)$ does not converge to $g(0)$ (since $g$ is not $2\pi$-periodic). Thus the supremum diverges. $\square$

### 3.2 Contrast with Snake Activation

**Proposition 3.3** (Snake Can Approximate Non-Periodic Functions). The Snake activation function $\text{Snake}_a(x) = x + \frac{1}{a}\sin^2(ax)$ contains a linear term $x$, enabling networks with Snake activation to approximate functions of the form:

$$f(x) = \text{trend}(x) + \text{periodic}(x)$$

*Sketch*: A two-layer Snake network can output:

$$f_{\text{Snake}}(x) = \sum_{i=1}^{N} w_i^{(2)} \cdot \text{Snake}_a(w_i^{(1)} x + b_i^{(1)}) + b^{(2)}$$

$$= \sum_{i=1}^{N} w_i^{(2)} \left[ w_i^{(1)} x + b_i^{(1)} + \frac{1}{a}\sin^2(a(w_i^{(1)} x + b_i^{(1)})) \right] + b^{(2)}$$

$$= \underbrace{\left(\sum_i w_i^{(2)} w_i^{(1)}\right) x + \text{const}}_{\text{Linear trend}} + \underbrace{\frac{1}{a}\sum_i w_i^{(2)} \sin^2(\cdot)}_{\text{Periodic component}}$$

By choosing weights appropriately, the network can:
1. Model linear trends (via the $x$ term)
2. Model periodic components (via the $\sin^2$ term)
3. Cancel the linear term if a purely periodic function is needed

**This flexibility is impossible for VQCs**, which are constrained to pure Fourier series.

### 3.3 Summary Table: Modeling Capabilities

| Signal Component | VQC | Snake | Winner |
|------------------|-----|-------|--------|
| Stationary periodic oscillations | ✓ | ✓ | Tie |
| Time-varying periodic (non-stationary) | Limited | Better | Snake |
| Linear trends | **✗ Cannot** | ✓ | **Snake** |
| Polynomial trends | **✗ Cannot** | ✓ | **Snake** |
| Exponential growth/decay | **✗ Cannot** | ✓ | **Snake** |
| Random walk / stochastic trends | **✗ Cannot** | Partially | **Snake** |
| Transient events | **✗ Cannot** | ✓ | **Snake** |
| Mixed periodic + trend | **✗ Only periodic part** | ✓ | **Snake** |

---

## 4. Empirical Evidence from the Original Papers {#4-empirical-evidence-from-the-original-papers}

### 4.1 Ziyin et al. (2020): Snake Performance on Real Data

#### 4.1.1 Atmospheric Temperature Prediction

**Setup**: Predict weekly temperature in Minamitorishima island (8+ years of data)

**Results**:
- **ReLU/tanh**: Failed to optimize, no meaningful extrapolation
- **Snake**: Successfully learned seasonal pattern AND extrapolated correctly

**Key insight**: The temperature data has both periodic (seasonal) and potentially trend components. Snake succeeded because it can model both.

#### 4.1.2 Human Body Temperature Prediction

**Setup**: Predict circadian rhythm from sparse, irregularly sampled data (25 measurements over 10 days)

**Results**:
- **ReLU/tanh**: Extrapolated to unrealistic temperatures (>39°C)
- **Snake**: Stayed within physiological range (35.5-37.5°C), captured circadian peak at ~4pm and minimum at ~4am

**Key insight**: Snake correctly extrapolated periodic pattern even with missing data (12am-8am).

#### 4.1.3 Financial Data (Wilshire 5000 Index)

**Setup**: Predict US market capitalization from 1995-2020, test on Feb-May 2020

**Results** (Table 2 from paper):

| Method | MSE on Test Set |
|--------|-----------------|
| ARIMA(2,1,1) | 0.0215 ± 0.0075 |
| ReLU DNN | 0.0113 ± 0.0002 |
| sin DNN | 0.0236 ± 0.0020 |
| **Snake** | **0.0089 ± 0.0002** |

**Critical finding**: "Snake is the only method that predicts a recession in and beyond the testing period"

**Why Snake succeeded**: It captured both:
1. Long-term business cycles (periodic component)
2. The underlying trend (which happened to be recessionary)

**Why VQC would fail here**: VQC would capture cycles but miss the trend, predicting continued oscillation around a constant level rather than a recession.

### 4.2 Schuld et al. (2021): VQC Theoretical Results (No Practical Superiority Claims)

The paper explicitly states:

> "Ideally, one would hope that our results could provide concrete guidelines for the design of quantum-machine-learning models. However, in practical settings the process of model selection should be guided not purely by model expressivity, but rather through the expected generalization performance of the model function class..."

> "...the insights on how to make models more expressive **should not be misinterpreted as recommendations for how to design good quantum models**—a question which is much more complex and whose answer depends strongly on the context."

**Key point**: The authors of the VQC paper themselves caution against interpreting expressivity results as practical performance predictions.

---

## 5. Practical Limitations of VQCs {#5-practical-limitations-of-vqcs}

### 5.1 Trainability Challenges

#### 5.1.1 Barren Plateaus

**Definition**: A barren plateau occurs when the variance of the gradient vanishes exponentially with system size:

$$\text{Var}\left[\frac{\partial \langle M \rangle}{\partial \theta_i}\right] \leq F(n) \cdot e^{-cn}$$

where $n$ is the number of qubits and $c > 0$ is a constant.

**Implications**:
- Gradient-based optimization becomes exponentially hard
- Random initialization almost surely starts in a flat region
- Affects deep circuits and global cost functions

**Contrast with Snake**: Classical backpropagation scales polynomially, with well-understood optimization landscapes.

#### 5.1.2 Local Minima and Trainability

| Aspect | VQC | Snake |
|--------|-----|-------|
| Loss landscape | Complex, many local minima | Also complex, but better understood |
| Gradient computation | Requires parameter-shift rule or finite differences | Standard backpropagation |
| Optimization maturity | Limited experience, few heuristics | Decades of research, many techniques |
| Hardware-efficient ansätze | May limit coefficient flexibility | N/A |

### 5.2 Hardware Constraints

#### 5.2.1 Circuit Depth Limitations

The theoretical proof that VQCs can realize arbitrary Fourier coefficients assumes:
- Trainable blocks $W^{(1)}, W^{(2)}$ can implement **arbitrary unitaries**

In practice:
- Arbitrary $n$-qubit unitaries require $O(4^n)$ gates in general
- NISQ devices limited to $O(100-1000)$ gates before decoherence dominates
- Hardware-efficient ansätze have limited expressivity

**Implication**: The theoretical coefficient flexibility may not be achievable on real hardware.

#### 5.2.2 Noise and Decoherence

| Noise Source | Effect on Fourier Representation |
|--------------|----------------------------------|
| Depolarizing noise | Suppresses higher-frequency coefficients |
| Amplitude damping | Biases coefficients toward ground state |
| Measurement noise | Statistical uncertainty in coefficient estimation |
| Crosstalk | Correlates errors across frequencies |

**Result**: The precise Fourier coefficients required for accurate approximation may be corrupted by hardware noise.

#### 5.2.3 Measurement Overhead

To estimate the expectation value $\langle M \rangle$ to precision $\varepsilon$:
- Requires $O(1/\varepsilon^2)$ measurement shots
- Each Fourier coefficient estimate has statistical uncertainty
- Total shots scale with number of frequencies and desired precision

### 5.3 Scalability Comparison

| Metric | VQC | Snake | Advantage |
|--------|-----|-------|-----------|
| Parameters for $m$-th order Fourier | $m$ encoding repetitions + trainable params | $4m$ neurons | Comparable |
| Computational cost (forward pass) | Quantum circuit execution | Matrix multiplications | **Snake** (classical hardware) |
| Computational cost (backward pass) | Parameter-shift rule ($2 \times$ forward passes per param) | Standard backprop | **Snake** |
| Training time | Limited by shot noise, hardware access | GPU-accelerated | **Snake** |
| Scaling to high dimensions | Exponential state space (potential advantage) | Polynomial | Unclear |

---

## 6. Detailed Comparison for Each Data Type {#6-detailed-comparison-for-each-data-type}

### 6.1 EEG Analysis

#### 6.1.1 Task: Rhythm Extraction / Spectral Analysis

| Criterion | VQC | Snake |
|-----------|-----|-------|
| Extracting alpha/beta/gamma rhythms | ✓ Good (native Fourier basis) | ✓ Good |
| Handling non-stationary rhythms | Limited (fixed coefficients) | Better (can learn time-varying) |
| **Verdict** | Comparable for stationary | **Snake** for non-stationary |

#### 6.1.2 Task: ERP / Event-Related Analysis

| Criterion | VQC | Snake |
|-----------|-----|-------|
| Modeling transient ERPs | **✗ Cannot** (non-periodic) | ✓ Can model |
| Baseline correction | **✗ Cannot** (trend removal) | ✓ Can model |
| Single-trial analysis | Challenging (noise, non-stationarity) | Feasible |
| **Verdict** | **Snake strongly preferred** | |

#### 6.1.3 Task: Seizure Prediction / Detection

| Criterion | VQC | Snake |
|-----------|-----|-------|
| Pre-ictal rhythm changes | ✓ Can detect frequency shifts | ✓ Can detect |
| Ictal onset (sudden change) | **✗ Cannot** (non-periodic event) | ✓ Can model |
| Post-ictal suppression | **✗ Cannot** (non-periodic) | ✓ Can model |
| **Verdict** | **Snake preferred** | |

#### 6.1.4 Overall Recommendation for EEG

**Use Snake-based models** for most EEG tasks because:
1. EEG contains essential non-periodic components (artifacts, ERPs, drift)
2. Non-stationarity is the norm, not the exception
3. Clinical applications require modeling transient events

**Potential VQC niche**: Pure frequency analysis where periodicity is explicitly assumed.

### 6.2 fMRI Analysis

#### 6.2.1 Task: Resting-State Connectivity

| Criterion | VQC | Snake |
|-----------|-----|-------|
| Low-frequency oscillations (0.01-0.1 Hz) | ✓ Can model | ✓ Can model |
| Physiological noise removal | ✓ Periodic components | ✓ Plus trends |
| Scanner drift correction | **✗ Cannot** | ✓ Can model |
| **Verdict** | Limited utility | **Snake preferred** |

#### 6.2.2 Task: Task-Based Activation Analysis

| Criterion | VQC | Snake |
|-----------|-----|-------|
| HRF modeling | **✗ Cannot** (non-periodic) | ✓ Can model |
| Event-related responses | **✗ Cannot** | ✓ Can model |
| Block design | Partially (if strictly periodic) | ✓ Flexible |
| **Verdict** | **Snake strongly preferred** | |

#### 6.2.3 Task: Dynamic Functional Connectivity

| Criterion | VQC | Snake |
|-----------|-----|-------|
| Time-varying connectivity | **✗ Fixed coefficients** | ✓ Can model |
| State transitions | **✗ Non-periodic** | ✓ Can model |
| **Verdict** | **Snake strongly preferred** | |

#### 6.2.4 Overall Recommendation for fMRI

**Use Snake-based models** for fMRI because:
1. BOLD response (the signal of interest) is fundamentally non-periodic
2. Scanner drift is ubiquitous and non-periodic
3. Task-based analyses require modeling transient events

**VQC is essentially unsuitable** for standard fMRI analysis paradigms.

### 6.3 Stock Price / Financial Time Series

#### 6.3.1 Task: Long-Term Trend Prediction

| Criterion | VQC | Snake |
|-----------|-----|-------|
| Business cycle detection | ✓ Can model cycles | ✓ Can model cycles |
| Trend direction | **✗ Cannot** (no trend capability) | ✓ Can model |
| Growth/recession prediction | **✗ Cannot** | ✓ Can model (as shown by Ziyin et al.) |
| **Verdict** | **Snake strongly preferred** | |

#### 6.3.2 Task: Volatility Modeling

| Criterion | VQC | Snake |
|-----------|-----|-------|
| Time-varying volatility | **✗ Fixed amplitude** | Partially (with extensions) |
| Volatility clustering | **✗ Cannot** | Partially |
| **Verdict** | Neither ideal, but **Snake better** | |

#### 6.3.3 Task: Anomaly / Shock Detection

| Criterion | VQC | Snake |
|-----------|-----|-------|
| Sudden market movements | **✗ Cannot** (non-periodic) | ✓ Can model |
| Regime changes | **✗ Cannot** | ✓ Can model |
| **Verdict** | **Snake strongly preferred** | |

#### 6.3.4 Overall Recommendation for Financial Data

**Use Snake-based models** because:
1. The dominant component (random walk/trend) is non-periodic
2. Practical value lies in trend prediction, not cycle fitting
3. Ziyin et al. empirically demonstrated Snake's superiority on real financial data

**VQC would fail** to capture the most important aspects of financial dynamics.

---

## 7. When Would VQCs Actually Outperform Snake? {#7-when-would-vqcs-actually-outperform-snake}

### 7.1 Scenario A: Purely Periodic Synthetic Data

**Conditions**:
- Target function is **exactly** a truncated Fourier series
- No noise, no trends, no non-periodic components
- Frequencies align with VQC's spectrum

**Why VQC might win**:
- Perfect inductive bias match
- No wasted capacity on non-periodic modeling

**Practicality**: **Very low**. Real-world data essentially never satisfies these conditions.

### 7.2 Scenario B: Quantum Feature Space Advantage

**Conditions**:
- The relevant features for the task lie in a quantum feature space
- Classical computation of these features is intractable
- The periodic structure is preserved in the quantum feature space

**Why VQC might win**:
- Quantum speedup in feature computation
- Classical methods cannot access the same feature space

**Practicality**: **Speculative**. No proven cases for EEG/fMRI/financial data.

### 7.3 Scenario C: Exponentially Many Interacting Periodic Modes

**Conditions**:
- Data has $2^n$ interacting periodic modes
- Correlations between modes are quantum-like (entanglement-type structure)
- Classical representation requires exponential resources

**Why VQC might win**:
- Quantum superposition naturally represents exponentially many modes
- Entanglement captures mode correlations efficiently

**Practicality**: **Highly speculative**. Would require evidence that EEG/fMRI/financial data has this structure.

### 7.4 Summary: Conditions for VQC Advantage

| Condition | Required for VQC Advantage | Present in Real EEG/fMRI/Stocks? |
|-----------|---------------------------|----------------------------------|
| Pure periodicity | Yes | **No** |
| No trends | Yes | **No** |
| No transients | Yes | **No** |
| Stationary coefficients | Yes | **No** (usually non-stationary) |
| Quantum feature space advantage | Would help | **Unproven** |

**Conclusion**: The conditions required for VQC advantage are **not met** by real EEG, fMRI, or financial data.

---

## 8. Recommendations for Research and Applications {#8-recommendations}

### 8.1 For Practitioners

#### 8.1.1 Primary Recommendation

**Use Snake-based deep learning models** for EEG, fMRI, and financial time series analysis because:

1. Real data has essential non-periodic components
2. Snake handles both periodic and trend components
3. Classical training is mature and scalable
4. Empirical evidence supports Snake's effectiveness

#### 8.1.2 When to Consider VQCs

Consider VQCs **only if**:
- The task explicitly requires pure frequency analysis
- You have prior knowledge that the signal is purely periodic
- You are exploring quantum computing for research purposes (not production)
- You want to test quantum advantage hypotheses

### 8.2 For Researchers

#### 8.2.1 Honest Framing of Theoretical Results

When publishing on VQC expressivity/universality:

**Do say**:
> "We establish that VQCs possess universal approximation/extrapolation capabilities for periodic functions, providing theoretical foundation for their use in frequency-based analyses."

**Do NOT say**:
> ~~"VQCs outperform classical methods for periodic data"~~ (Not supported by theory or experiments)

#### 8.2.2 Hybrid Architecture Research Direction

A promising research direction is **hybrid VQC-Snake architectures**:

```
Input Signal x(t)
       │
       ▼
┌──────────────────────────────────────┐
│         Signal Decomposition          │
│  (e.g., EMD, wavelet, bandpass)      │
└──────────────────────────────────────┘
       │                    │
       ▼                    ▼
┌─────────────────┐  ┌─────────────────┐
│ Periodic Part   │  │ Aperiodic Part  │
│                 │  │                 │
│    VQC Block    │  │   Snake MLP    │
│                 │  │                 │
│ f_VQC(x) =      │  │ f_Snake(x) =   │
│ Σ c_n e^{inx}   │  │ trend + noise   │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  ▼
         ┌───────────────┐
         │   Recombine   │
         │               │
         │ f(x) = f_VQC  │
         │     + f_Snake │
         └───────────────┘
                  │
                  ▼
         Final Prediction
```

**Rationale**:
- VQC handles periodic components with guaranteed extrapolation
- Snake handles trends and aperiodic components
- Combined system achieves universal extrapolation for general signals

### 8.3 For Grant Proposals / Publications

#### 8.3.1 Appropriate Claims

✓ "VQCs provide a natural framework for periodic signal analysis due to their inherent Fourier structure"

✓ "The universal periodic extrapolation property of VQCs may be advantageous for signals known to be purely periodic"

✓ "Hybrid quantum-classical architectures may combine VQC's periodic extrapolation with classical trend modeling"

#### 8.3.2 Claims to Avoid

✗ "VQCs outperform deep learning for EEG/fMRI analysis" (Not supported)

✗ "Quantum advantage for biomedical signal processing" (Not demonstrated)

✗ "VQCs solve the extrapolation problem for time series" (Only for purely periodic signals)

---

## 9. Conclusion {#9-conclusion}

### 9.1 Direct Answer to the Original Question

**Question**: Will VQCs show better extrapolation performance than Snake-based deep learning models for EEG, fMRI, and stock price data?

**Answer**: **No**, for the following reasons:

1. **Fundamental limitation**: VQC outputs are inherently periodic and cannot model the non-periodic components (trends, transients, drifts) that are essential in real EEG, fMRI, and financial data.

2. **Snake's advantage**: Snake activation explicitly enables modeling of both periodic and non-periodic components, making it more suitable for real-world signals that contain both.

3. **Empirical evidence**: Ziyin et al. demonstrated Snake's superiority on real temperature and financial data. No comparable demonstrations exist for VQCs on these data types.

4. **Practical constraints**: VQCs face trainability challenges, hardware limitations, and scalability issues that do not affect classical Snake networks.

### 9.2 The Value of the Universal Extrapolation Result

The theoretical result that VQCs can universally extrapolate periodic functions is **not useless**—it provides:

1. **Foundational understanding**: Clarifies what functions VQCs can and cannot represent
2. **Design guidance**: Explains how encoding repetitions expand the frequency spectrum
3. **Inductive bias characterization**: Identifies VQCs' natural affinity for periodic functions
4. **Potential niche applications**: Points toward tasks where pure periodicity is a valid assumption

### 9.3 Final Recommendation

For **practical applications** in EEG, fMRI, and financial time series:

| Approach | Recommendation |
|----------|----------------|
| **Primary choice** | Snake-based deep learning |
| **Research exploration** | Hybrid VQC-Snake architectures |
| **Pure frequency analysis** | VQC may be appropriate |
| **Production systems** | Classical methods (Snake, RNNs, Transformers) |

### 9.4 Closing Statement

The universal periodic extrapolation capability of VQCs is a **mathematically elegant result** that deepens our understanding of quantum machine learning models. However, **theoretical expressivity does not equal practical performance**. For real-world spatio-temporal data that inherently mixes periodic and aperiodic components, classical deep learning with Snake activation remains the superior choice.

---

## Appendix: Summary Comparison Table

| Criterion | VQC | Snake | Better for Real Data |
|-----------|-----|-------|----------------------|
| **Theoretical periodic extrapolation** | ✓ Universal | ✓ Universal | Tie |
| **Non-periodic trend modeling** | ✗ Cannot | ✓ Native | **Snake** |
| **Mixed signal modeling** | ✗ Only periodic | ✓ Both | **Snake** |
| **EEG (with artifacts, ERPs)** | Poor fit | Good fit | **Snake** |
| **fMRI (with HRF, drift)** | Very poor fit | Good fit | **Snake** |
| **Financial (with trends, shocks)** | Poor fit | Good fit | **Snake** |
| **Trainability** | Challenging | Standard | **Snake** |
| **Scalability** | Hardware-limited | GPU-scalable | **Snake** |
| **Maturity** | Emerging | Established | **Snake** |
| **Empirical validation** | Limited | Demonstrated | **Snake** |

---

*Document prepared for research on quantum machine learning for biomedical signal processing and financial time series analysis.*
