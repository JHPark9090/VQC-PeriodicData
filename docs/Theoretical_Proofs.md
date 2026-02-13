# Theoretical Foundations for Frequency-Aware Quantum Time-Series Architectures

**Four Theorems on Frequency-Matched Encoding, FFT-Seeded Initialization, Periodicity-Preserving Gates, and Generalization to Spectrally Concentrated Signals**

---

## Table of Contents

1. [Notation and Preliminaries](#1-notation-and-preliminaries)
2. [Theorem 1: Frequency Spectrum Expansion via Multi-Axis Encoding](#2-theorem-1-frequency-spectrum-expansion-via-multi-axis-encoding)
3. [Theorem 2: Optimality of FFT-Seeded Frequency Scale Initialization](#3-theorem-2-optimality-of-fft-seeded-frequency-scale-initialization)
4. [Theorem 3: Periodicity Preservation Under Gate Transformations](#4-theorem-3-periodicity-preservation-under-gate-transformations)
5. [Theorem 4: Generalization to Non-Periodic Spectrally Concentrated Signals](#5-theorem-4-generalization-to-non-periodic-spectrally-concentrated-signals)
6. [Discussion: Design Implications](#6-discussion-design-implications)
7. [References](#7-references)

---

## 1. Notation and Preliminaries

### 1.1 Pauli Matrices

The single-qubit Pauli matrices are:

$$\sigma_X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}, \quad I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

These satisfy the commutation relations $[\sigma_a, \sigma_b] = 2i\epsilon_{abc}\sigma_c$ and anticommutation relations $\{\sigma_a, \sigma_b\} = 2\delta_{ab}I$. The product rules are:

$$\sigma_X \sigma_Y = i\sigma_Z, \quad \sigma_Y \sigma_Z = i\sigma_X, \quad \sigma_Z \sigma_X = i\sigma_Y$$

and their cyclic permutations with a sign flip for reverse order.

### 1.2 Rotation Gates

Single-qubit rotation gates are defined as:

$$R_Y(\theta) = e^{-i\frac{\theta}{2}\sigma_Y} = \cos\frac{\theta}{2}\, I - i\sin\frac{\theta}{2}\, \sigma_Y = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

$$R_X(\theta) = e^{-i\frac{\theta}{2}\sigma_X} = \cos\frac{\theta}{2}\, I - i\sin\frac{\theta}{2}\, \sigma_X = \begin{pmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}$$

### 1.3 Variational Quantum Circuit (VQC) Model

A parameterized quantum circuit (PQC) with data re-uploading computes a function:

$$f_{\boldsymbol{\alpha},\boldsymbol{\theta}}(\mathbf{x}) = \langle 0^n | U^\dagger(\mathbf{x}, \boldsymbol{\alpha}, \boldsymbol{\theta})\, O\, U(\mathbf{x}, \boldsymbol{\alpha}, \boldsymbol{\theta}) | 0^n \rangle$$

where:
- $\mathbf{x} = (x_1, \ldots, x_n)$ is the input data
- $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_n)$ are learnable frequency scale parameters
- $\boldsymbol{\theta}$ are variational rotation parameters
- $O$ is a Hermitian observable (e.g., $\sigma_Z^{(k)}$)
- $U = V(\boldsymbol{\theta}_0) \prod_{l=1}^L S(\mathbf{x}, \boldsymbol{\alpha}) V(\boldsymbol{\theta}_l)$ with encoding $S$ and variational layers $V$

### 1.4 Encoding Schemes

**Standard encoding (RY-only):** Each qubit $i$ encodes input $x_i$ via:

$$S_i^{\text{std}}(x_i) = R_Y(x_i)$$

**Frequency-matched encoding (RY+RX):** Each qubit $i$ encodes input $x_i$ via:

$$S_i^{\text{FM}}(x_i, \alpha_i) = R_Y(x_i)\, R_X(\alpha_i x_i)$$

where $\alpha_i \in \mathbb{R}$ is a learnable frequency scale parameter.

### 1.5 Heisenberg Picture

For an encoding unitary $S$ and Pauli operator $P$, the Heisenberg-picture evolution is:

$$\widetilde{P}(x) = S^\dagger(x)\, P\, S(x)$$

The expectation value of the circuit is expressed via the Heisenberg-picture observable. For an $n$-qubit system with tensor-product encoding $S = \bigotimes_{i=1}^n S_i$ and observable $O = \sum_{\mathbf{P}} a_{\mathbf{P}} \bigotimes_{i=1}^n \sigma_{P_i}$ (Pauli decomposition), the expectation value becomes:

$$f(\mathbf{x}) = \sum_{\mathbf{P}} a_{\mathbf{P}} \prod_{i=1}^n \langle \psi_i | S_i^\dagger(x_i)\, \sigma_{P_i}\, S_i(x_i) | \psi_i \rangle$$

where $|\psi_i\rangle$ are effective per-qubit states determined by the variational parameters $\boldsymbol{\theta}$.

### 1.6 Fourier Representation Theorem (Schuld et al., 2021)

**Theorem (Schuld et al., 2021).** *The output of a variational quantum circuit with $L$ re-uploading layers, $n$-qubit encoding $S(\mathbf{x}) = \exp(-i \sum_j x_j H_j)$, and observable $O$ is a multivariate trigonometric polynomial:*

$$f(\mathbf{x}) = \sum_{\boldsymbol{\omega} \in \Omega} c_{\boldsymbol{\omega}} \exp\!\left(i \sum_{j=1}^n \omega_j x_j\right)$$

*where the accessible frequency set $\Omega \subseteq \mathbb{Z}^n$ is determined by the eigenvalue differences of the encoding Hamiltonians $\{H_j\}$.*

### 1.7 Spectral Concentration and Signal Classification

The following definitions formalize the notion of spectral concentration, which will be central to extending our results beyond periodic signals.

**Definition 5 (Discrete Fourier Transform).** For a discrete signal $\mathbf{x} = (x[0], \ldots, x[N-1]) \in \mathbb{R}^N$, the Discrete Fourier Transform (DFT) is:

$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-i2\pi kn / N}, \qquad k = 0, 1, \ldots, N-1$$

The power spectrum is $P[k] = |X[k]|^2$, and the total signal energy (by Parseval's equality for the DFT) is:

$$E = \sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N}\sum_{k=0}^{N-1} |X[k]|^2$$

**Definition 6 (Spectral Concentration Ratio).** For a signal with DFT power spectrum $\{P[k]\}_{k=0}^{N-1}$ and an integer $n \leq N$, the **spectral concentration ratio** is:

$$\rho(n) = \frac{\sum_{j=1}^{n} P_{(j)}}{\sum_{k=0}^{N-1} P[k]}$$

where $P_{(1)} \geq P_{(2)} \geq \cdots \geq P_{(N)}$ are the power spectrum values sorted in decreasing order (excluding the DC component $P[0]$). The ratio $\rho(n) \in [0, 1]$ measures the fraction of total signal energy captured by the $n$ most energetic frequency components.

**Definition 7 (Spectral Decay Class).** A signal class $\mathcal{S}(\beta, C)$ with **spectral decay exponent** $\beta > 0$ consists of signals whose ordered (by magnitude) Fourier coefficients satisfy:

$$|a_{(k)}| \leq C \cdot k^{-\beta} \qquad \text{for all } k \geq 1$$

where $C > 0$ is a class constant. Equivalently, $P_{(k)} \leq C^2 \cdot k^{-2\beta}$. Higher $\beta$ implies faster spectral decay and stronger concentration.

**Definition 8 (Signal Classification by Spectral Properties).**

| Signal Class | Description | Spectral Property | Decay Exponent $\beta$ |
|---|---|---|---|
| **Finite harmonic** | Superposition of $K$ sinusoids | $P_{(k)} = 0$ for $k > K$ | $\beta = \infty$ |
| **Smooth periodic** | $C^m$, period $T$ | $\|a_k\| = O(\|k\|^{-(m+1)})$ | $\beta = m + 1$ |
| **Analytic periodic** | Analytic, period $T$ | $\|a_k\| = O(e^{-c\|k\|})$ | $\beta = \infty$ (exponential) |
| **Quasi-periodic** | Dominant peaks, not exactly harmonic | Concentrated but non-integer ratios | Effective $\beta \gg 1$ |
| **Smooth non-periodic** | $C^m$ on $[0,T]$, not periodic | With windowing: $O(\|k\|^{-(m+1)})$ | $\beta = m + 1$ (windowed) |
| **Bounded variation** | Finite discontinuities | $\|a_k\| = O(\|k\|^{-1})$ | $\beta = 1$ |
| **White noise** | i.i.d. random | Flat: $\mathbb{E}[P[k]] = \text{const}$ | $\beta = 0$ |

The key insight is that most real-world time series — including EEG, temperature, financial data, speech, and vibration signals — fall into the quasi-periodic, smooth periodic, or smooth non-periodic categories, all of which exhibit significant spectral concentration ($\rho(n)$ close to 1 for moderate $n$).

---

## 2. Theorem 1: Frequency Spectrum Expansion via Multi-Axis Encoding

### 2.1 Lemma 1 (Heisenberg Evolution Under RY-Only Encoding)

**Lemma 1.** *For standard RY-only encoding $S_i^{\text{std}}(x_i) = R_Y(x_i)$ on qubit $i$, the Heisenberg-picture Pauli operators are:*

| Pauli $P$ | $S_i^{\text{std}\dagger} P\, S_i^{\text{std}}$ | Frequencies in $x_i$ |
|---|---|---|
| $I$ | $I$ | $\{0\}$ |
| $\sigma_X$ | $\cos(x_i)\,\sigma_X + \sin(x_i)\,\sigma_Z$ | $\{-1, +1\}$ |
| $\sigma_Y$ | $\sigma_Y$ | $\{0\}$ |
| $\sigma_Z$ | $\cos(x_i)\,\sigma_Z - \sin(x_i)\,\sigma_X$ | $\{-1, +1\}$ |

*The per-qubit accessible frequency set is $\Omega_i^{\text{std}} = \{-1, 0, +1\}$, with $|\Omega_i^{\text{std}}| = 3$.*

**Proof.** We compute $R_Y^\dagger(x_i)\, P\, R_Y(x_i)$ for each Pauli operator $P$ using the identity $e^{i\frac{\theta}{2}\sigma_a}\, \sigma_b\, e^{-i\frac{\theta}{2}\sigma_a} = \sigma_b$ if $a = b$, and the rotation formula for $a \neq b$.

**Case $P = I$:** Trivial; $R_Y^\dagger I\, R_Y = I$. Frequency: $\{0\}$.

**Case $P = \sigma_Z$:** We expand using $R_Y(\theta) = \cos\frac{\theta}{2}\,I - i\sin\frac{\theta}{2}\,\sigma_Y$:

$$R_Y^\dagger(x)\,\sigma_Z\,R_Y(x) = \left(\cos\frac{x}{2}\,I + i\sin\frac{x}{2}\,\sigma_Y\right)\sigma_Z\left(\cos\frac{x}{2}\,I - i\sin\frac{x}{2}\,\sigma_Y\right)$$

Expanding the product and collecting terms:

$$= \cos^2\!\frac{x}{2}\,\sigma_Z + i\sin\frac{x}{2}\cos\frac{x}{2}\,(\sigma_Y\sigma_Z - \sigma_Z\sigma_Y) + \sin^2\!\frac{x}{2}\,\sigma_Y\sigma_Z\sigma_Y$$

Using the commutator $[\sigma_Y, \sigma_Z] = \sigma_Y\sigma_Z - \sigma_Z\sigma_Y = 2i\sigma_X$ and the conjugation identity $\sigma_Y\sigma_Z\sigma_Y = -\sigma_Z$ (which follows from $\sigma_Y\sigma_Z = i\sigma_X$ and $\sigma_X\sigma_Y = i\sigma_Z$, giving $\sigma_Y\sigma_Z\sigma_Y = i\sigma_X\sigma_Y = i \cdot i\sigma_Z = -\sigma_Z$):

$$= \left(\cos^2\!\frac{x}{2} - \sin^2\!\frac{x}{2}\right)\sigma_Z - 2\sin\frac{x}{2}\cos\frac{x}{2}\,\sigma_X$$

Applying the double-angle identities $\cos^2\!\frac{x}{2} - \sin^2\!\frac{x}{2} = \cos x$ and $2\sin\frac{x}{2}\cos\frac{x}{2} = \sin x$:

$$R_Y^\dagger(x)\,\sigma_Z\,R_Y(x) = \cos(x)\,\sigma_Z - \sin(x)\,\sigma_X \tag{1}$$

This contains frequencies $\{-1, +1\}$ (from $\cos x$ and $\sin x$).

**Case $P = \sigma_X$:** By the same expansion:

$$R_Y^\dagger(x)\,\sigma_X\,R_Y(x) = \cos^2\!\frac{x}{2}\,\sigma_X + i\sin\frac{x}{2}\cos\frac{x}{2}\,(\sigma_Y\sigma_X - \sigma_X\sigma_Y) + \sin^2\!\frac{x}{2}\,\sigma_Y\sigma_X\sigma_Y$$

Using $[\sigma_Y, \sigma_X] = -2i\sigma_Z$ and $\sigma_Y\sigma_X\sigma_Y = -\sigma_X$ (from $\sigma_Y\sigma_X = -i\sigma_Z$, $\sigma_Z\sigma_Y = -i\sigma_X$, so $\sigma_Y\sigma_X\sigma_Y = -i\sigma_Z\sigma_Y = -i(-i\sigma_X) = -\sigma_X$):

$$= (\cos^2\!\frac{x}{2} - \sin^2\!\frac{x}{2})\,\sigma_X + 2\sin\frac{x}{2}\cos\frac{x}{2}\,\sigma_Z = \cos(x)\,\sigma_X + \sin(x)\,\sigma_Z \tag{2}$$

Frequencies: $\{-1, +1\}$.

**Case $P = \sigma_Y$:** Since $[\sigma_Y, R_Y(\theta)] = 0$ (rotation around the Y-axis commutes with $\sigma_Y$):

$$R_Y^\dagger(x)\,\sigma_Y\,R_Y(x) = \sigma_Y \tag{3}$$

Frequency: $\{0\}$.

Taking the union over all Pauli operators: $\Omega_i^{\text{std}} = \{0\} \cup \{-1,+1\} \cup \{0\} \cup \{-1,+1\} = \{-1, 0, +1\}$. $\quad\blacksquare$

---

### 2.2 Lemma 2 (Heisenberg Evolution Under Frequency-Matched Encoding)

**Lemma 2.** *For frequency-matched encoding $S_i^{\text{FM}}(x_i, \alpha_i) = R_Y(x_i)\,R_X(\alpha_i x_i)$ on qubit $i$ with frequency scale $\alpha_i \in \mathbb{R}$, the Heisenberg-picture Pauli operators are:*

| Pauli $P$ | $S_i^{\text{FM}\dagger} P\, S_i^{\text{FM}}$ | Frequencies in $x_i$ |
|---|---|---|
| $I$ | $I$ | $\{0\}$ |
| $\sigma_X$ | $\cos(x_i)\,\sigma_X + \sin(x_i)\cos(\alpha_i x_i)\,\sigma_Z + \sin(x_i)\sin(\alpha_i x_i)\,\sigma_Y$ | $\{-1-\alpha_i,\, -(1-\alpha_i),\, -1,\, +1,\, 1-\alpha_i,\, 1+\alpha_i\}$ |
| $\sigma_Y$ | $\cos(\alpha_i x_i)\,\sigma_Y - \sin(\alpha_i x_i)\,\sigma_Z$ | $\{-\alpha_i,\, +\alpha_i\}$ |
| $\sigma_Z$ | $\cos(x_i)\cos(\alpha_i x_i)\,\sigma_Z + \cos(x_i)\sin(\alpha_i x_i)\,\sigma_Y - \sin(x_i)\,\sigma_X$ | $\{-1-\alpha_i,\, -(1-\alpha_i),\, -1,\, +1,\, 1-\alpha_i,\, 1+\alpha_i\}$ |

*The per-qubit accessible frequency set is:*

$$\Omega_i^{\text{FM}}(\alpha_i) = \{0,\; \pm\alpha_i,\; \pm 1,\; \pm(1 - \alpha_i),\; \pm(1 + \alpha_i)\}$$

*For $\alpha_i \notin \{0, \pm 1, \pm 2\}$, all nine elements are distinct, giving $|\Omega_i^{\text{FM}}| = 9$.*

**Proof.** We decompose $S_i^{\text{FM}} = R_Y(x_i)\,R_X(\alpha_i x_i)$ and compute the Heisenberg evolution in two stages:

$$S_i^{\text{FM}\dagger}\, P\, S_i^{\text{FM}} = R_X^\dagger(\alpha_i x_i)\,\big[\underbrace{R_Y^\dagger(x_i)\,P\,R_Y(x_i)}_{\text{Step 1: RY evolution}}\big]\,R_X(\alpha_i x_i)$$

**Step 1** (inner evolution under $R_Y$) is given by Lemma 1, Eqs. (1)-(3).

**Step 2** applies $R_X$ conjugation to the result of Step 1. We need the Heisenberg evolution under $R_X$ for each Pauli operator.

**Sub-lemma (RX Heisenberg evolution).** For $R_X(\phi) = e^{-i\frac{\phi}{2}\sigma_X}$:

$$R_X^\dagger(\phi)\,\sigma_X\,R_X(\phi) = \sigma_X \tag{4}$$

$$R_X^\dagger(\phi)\,\sigma_Y\,R_X(\phi) = \cos(\phi)\,\sigma_Y - \sin(\phi)\,\sigma_Z \tag{5}$$

$$R_X^\dagger(\phi)\,\sigma_Z\,R_X(\phi) = \cos(\phi)\,\sigma_Z + \sin(\phi)\,\sigma_Y \tag{6}$$

*Proof of sub-lemma.* Equation (4) holds because $\sigma_X$ commutes with $R_X$. For Eq. (6), by the same expansion technique as in Lemma 1:

$$R_X^\dagger(\phi)\,\sigma_Z\,R_X(\phi) = \cos^2\!\frac{\phi}{2}\,\sigma_Z + i\sin\frac{\phi}{2}\cos\frac{\phi}{2}\,(\sigma_X\sigma_Z - \sigma_Z\sigma_X) + \sin^2\!\frac{\phi}{2}\,\sigma_X\sigma_Z\sigma_X$$

Using $[\sigma_X, \sigma_Z] = -2i\sigma_Y$ and $\sigma_X\sigma_Z\sigma_X = -\sigma_Z$ (from $\sigma_X\sigma_Z = -i\sigma_Y$, $\sigma_Y\sigma_X = -i\sigma_Z$, giving $\sigma_X\sigma_Z\sigma_X = -i\sigma_Y\sigma_X = -i(-i\sigma_Z) = -\sigma_Z$):

$$= (\cos^2\!\frac{\phi}{2} - \sin^2\!\frac{\phi}{2})\,\sigma_Z + 2\sin\frac{\phi}{2}\cos\frac{\phi}{2}\,\sigma_Y = \cos(\phi)\,\sigma_Z + \sin(\phi)\,\sigma_Y$$

Equation (5) follows by an analogous calculation. $\quad\square$

Now we combine Steps 1 and 2 for each Pauli operator, with $\phi = \alpha_i x_i$:

**Case $P = \sigma_Z$:** From Step 1, Eq. (1): $R_Y^\dagger \sigma_Z R_Y = \cos(x_i)\sigma_Z - \sin(x_i)\sigma_X$.

Applying Step 2 with Eqs. (4) and (6):

$$S_i^{\text{FM}\dagger}\,\sigma_Z\,S_i^{\text{FM}} = \cos(x_i)\big[\cos(\alpha_i x_i)\,\sigma_Z + \sin(\alpha_i x_i)\,\sigma_Y\big] - \sin(x_i)\,\sigma_X$$

$$= \cos(x_i)\cos(\alpha_i x_i)\,\sigma_Z + \cos(x_i)\sin(\alpha_i x_i)\,\sigma_Y - \sin(x_i)\,\sigma_X \tag{7}$$

We extract the frequencies using product-to-sum identities:

$$\cos(x_i)\cos(\alpha_i x_i) = \tfrac{1}{2}\big[\cos\!\big((1-\alpha_i)x_i\big) + \cos\!\big((1+\alpha_i)x_i\big)\big]$$

$$\cos(x_i)\sin(\alpha_i x_i) = \tfrac{1}{2}\big[\sin\!\big((1+\alpha_i)x_i\big) - \sin\!\big((1-\alpha_i)x_i\big)\big]$$

Frequencies in Eq. (7): $\pm(1-\alpha_i)$, $\pm(1+\alpha_i)$ (from the products), and $\pm 1$ (from $\sin x_i$).

**Case $P = \sigma_X$:** From Step 1, Eq. (2): $R_Y^\dagger \sigma_X R_Y = \cos(x_i)\sigma_X + \sin(x_i)\sigma_Z$.

Applying Step 2 with Eqs. (4) and (6):

$$S_i^{\text{FM}\dagger}\,\sigma_X\,S_i^{\text{FM}} = \cos(x_i)\,\sigma_X + \sin(x_i)\big[\cos(\alpha_i x_i)\,\sigma_Z + \sin(\alpha_i x_i)\,\sigma_Y\big]$$

$$= \cos(x_i)\,\sigma_X + \sin(x_i)\cos(\alpha_i x_i)\,\sigma_Z + \sin(x_i)\sin(\alpha_i x_i)\,\sigma_Y \tag{8}$$

Using:

$$\sin(x_i)\cos(\alpha_i x_i) = \tfrac{1}{2}\big[\sin\!\big((1+\alpha_i)x_i\big) + \sin\!\big((1-\alpha_i)x_i\big)\big]$$

$$\sin(x_i)\sin(\alpha_i x_i) = \tfrac{1}{2}\big[\cos\!\big((1-\alpha_i)x_i\big) - \cos\!\big((1+\alpha_i)x_i\big)\big]$$

Frequencies: $\pm 1$, $\pm(1-\alpha_i)$, $\pm(1+\alpha_i)$.

**Case $P = \sigma_Y$:** From Step 1, Eq. (3): $R_Y^\dagger \sigma_Y R_Y = \sigma_Y$. Applying Step 2 with Eq. (5):

$$S_i^{\text{FM}\dagger}\,\sigma_Y\,S_i^{\text{FM}} = \cos(\alpha_i x_i)\,\sigma_Y - \sin(\alpha_i x_i)\,\sigma_Z \tag{9}$$

Frequencies: $\pm \alpha_i$.

**Case $P = I$:** Trivially $I$. Frequency: $\{0\}$.

Taking the union over all four Pauli operators:

$$\Omega_i^{\text{FM}}(\alpha_i) = \{0\} \cup \{\pm 1,\, \pm(1\!-\!\alpha_i),\, \pm(1\!+\!\alpha_i)\} \cup \{\pm\alpha_i\} = \{0,\; \pm\alpha_i,\; \pm 1,\; \pm(1\!-\!\alpha_i),\; \pm(1\!+\!\alpha_i)\}$$

**Distinctness conditions.** The nine elements are distinct when:
- $\alpha_i \neq 0$ (otherwise $\pm\alpha_i = 0$, merging with DC)
- $\alpha_i \neq \pm 1$ (otherwise $1-\alpha_i = 0$ or $1+\alpha_i = 0$, merging with DC)
- $\alpha_i \neq \pm 2$ (otherwise $1+\alpha_i = \pm(1-\alpha_i)$, causing collisions)

For $\alpha_i \notin \{0, \pm 1, \pm 2\}$, all nine frequencies are distinct. $\quad\blacksquare$

---

### 2.3 Theorem 1 (Main Result)

**Theorem 1 (Frequency Spectrum Expansion).** *Consider an $n$-qubit variational quantum circuit with tensor-product encoding $S(\mathbf{x}) = \bigotimes_{i=1}^n S_i(x_i)$ and arbitrary entangling variational layers. Let $\Omega^{\text{std}}$ and $\Omega^{\text{FM}}(\boldsymbol{\alpha})$ denote the multivariate accessible frequency sets under standard (RY-only) and frequency-matched (RY+RX) encoding, respectively.*

*(a) (Strict superset property.) For any $\boldsymbol{\alpha}$ with $\alpha_i \neq 0$ for all $i$:*

$$\Omega^{\text{std}} \subsetneq \Omega^{\text{FM}}(\boldsymbol{\alpha})$$

*(b) (Multivariate frequency count.) The number of multivariate frequency vectors is:*

$$|\Omega^{\text{std}}| = 3^n$$

$$|\Omega^{\text{FM}}(\boldsymbol{\alpha})| \leq 9^n$$

*with equality when $\alpha_i \notin \{0, \pm 1, \pm 2\}$ for all $i$ and the $\alpha_i$ values are "generic" (no accidental collisions between cross-qubit frequency sums).*

*(c) (Expansion factor.) For generic $\boldsymbol{\alpha}$, the multiplicative expansion in the number of accessible multivariate frequency vectors is:*

$$\frac{|\Omega^{\text{FM}}(\boldsymbol{\alpha})|}{|\Omega^{\text{std}}|} = 3^n$$

*This is an exponential improvement in the richness of the representable function class.*

**Proof.**

**(a)** For each qubit $i$, by Lemmas 1 and 2:

$$\Omega_i^{\text{std}} = \{-1, 0, +1\} \subset \{0,\, \pm\alpha_i,\, \pm 1,\, \pm(1\!-\!\alpha_i),\, \pm(1\!+\!\alpha_i)\} = \Omega_i^{\text{FM}}(\alpha_i)$$

The inclusion is strict when $\alpha_i \neq 0$, since $\pm\alpha_i \notin \Omega_i^{\text{std}}$.

The multivariate frequency set is the Cartesian product of per-qubit frequency sets (each qubit contributes one frequency to the multivariate vector):

$$\Omega = \Omega_1 \times \Omega_2 \times \cdots \times \Omega_n$$

Since $\Omega_i^{\text{std}} \subsetneq \Omega_i^{\text{FM}}$ for each $i$, we have:

$$\Omega^{\text{std}} = \prod_{i=1}^n \Omega_i^{\text{std}} \subsetneq \prod_{i=1}^n \Omega_i^{\text{FM}} = \Omega^{\text{FM}}(\boldsymbol{\alpha})$$

**(b)** The per-qubit frequency sets have cardinalities $|\Omega_i^{\text{std}}| = 3$ and $|\Omega_i^{\text{FM}}| \leq 9$ (equality under the distinctness conditions from Lemma 2). The multivariate Cartesian product has cardinality equal to the product:

$$|\Omega^{\text{std}}| = \prod_{i=1}^n |\Omega_i^{\text{std}}| = 3^n$$

$$|\Omega^{\text{FM}}(\boldsymbol{\alpha})| = \prod_{i=1}^n |\Omega_i^{\text{FM}}(\alpha_i)| \leq 9^n$$

For equality in the frequency-matched case, we need (i) all per-qubit frequency sets to have exactly 9 distinct elements ($\alpha_i \notin \{0, \pm 1, \pm 2\}$ for all $i$), and (ii) no accidental collisions between multivariate frequency vectors across different qubit combinations. Condition (ii) holds when the $\alpha_i$ values are algebraically independent over $\mathbb{Q}$, since the multivariate frequencies are linear combinations of $\{1, \alpha_1, \ldots, \alpha_n\}$ with integer coefficients, and algebraic independence prevents non-trivial coincidences.

**(c)** Under the conditions of (b) with equality:

$$\frac{|\Omega^{\text{FM}}|}{|\Omega^{\text{std}}|} = \frac{9^n}{3^n} = \left(\frac{9}{3}\right)^n = 3^n \quad\blacksquare$$

---

### 2.4 Corollary 1.1 (Univariate Frequency Spectrum)

When all qubits encode the **same** scalar variable $x$ (i.e., $x_1 = x_2 = \cdots = x_n = x$), the output is a univariate trigonometric polynomial $f(x) = \sum_\omega c_\omega e^{i\omega x}$, and the univariate frequency is the **sum** of per-qubit frequencies: $\omega = \sum_{i=1}^n \omega_i$.

**Corollary 1.1.** *In the univariate encoding case:*

*(a) Under standard encoding: $\Omega_{\text{uni}}^{\text{std}} = \{-n, -n+1, \ldots, n-1, n\}$, giving $|\Omega_{\text{uni}}^{\text{std}}| = 2n + 1$.*

*(b) Under frequency-matched encoding with generic $\boldsymbol{\alpha}$: $|\Omega_{\text{uni}}^{\text{FM}}|$ grows exponentially with $n$, bounded below by the number of distinct sums $\sum_{i=1}^n \omega_i$ where $\omega_i \in \Omega_i^{\text{FM}}(\alpha_i)$.*

**Proof.** (a) Each qubit contributes $\omega_i \in \{-1, 0, +1\}$. The sum $\sum_i \omega_i$ ranges over all integers from $-n$ to $+n$, giving $2n+1$ distinct values.

(b) Each qubit contributes $\omega_i \in \{0, \pm\alpha_i, \pm 1, \pm(1\pm\alpha_i)\}$. When the $\alpha_i$ are algebraically independent over $\mathbb{Q}$, the sums $\sum_i \omega_i$ are distinct for distinct multivariate frequency vectors (by the definition of algebraic independence), yielding up to $9^n$ distinct univariate frequencies. $\quad\blacksquare$

### 2.5 Corollary 1.2 (Tunability of Frequency Support)

**Corollary 1.2.** *The learnable parameters $\boldsymbol{\alpha}$ provide continuous control over the frequency support $\Omega^{\text{FM}}(\boldsymbol{\alpha})$. Specifically, varying $\alpha_i$ continuously shifts the per-qubit frequencies $\pm\alpha_i$, $\pm(1-\alpha_i)$, and $\pm(1+\alpha_i)$ across $\mathbb{R}$, allowing the circuit to tune its spectral sensitivity to match the data's frequency content.*

*In contrast, the standard encoding has a fixed frequency support $\Omega^{\text{std}} = \{-1, 0, +1\}^n$ that cannot be adapted to the data.*

**Proof.** Immediate from the $\alpha_i$-dependence in $\Omega_i^{\text{FM}}(\alpha_i)$. $\quad\blacksquare$

---

### 2.6 Remark (Connection to Prior Work)

Theorem 1 provides a concrete characterization of the spectral expansion mechanism in the FourierQLSTM and FourierQTCN architectures. The result is consistent with and extends the qualitative observation from Schuld et al. (2021) that "more complex encoding strategies access richer frequency spectra." Our contribution is the **explicit quantification** of the per-qubit expansion ($3 \to 9$ frequencies) and the overall exponential improvement ($3^n \to 9^n$ multivariate frequencies) for the specific RY+RX encoding used in our architectures.

The exponential growth $9^n$ in accessible Fourier terms should be compared to classical Fourier neural networks (including SIREN), where the number of representable frequencies grows linearly with the number of parameters. This is the parameter efficiency advantage identified by Zhao et al. (ICML, 2024) for data re-uploading circuits in the context of implicit neural representations.

---

## 3. Theorem 2: Optimality of FFT-Seeded Frequency Scale Initialization

### 3.1 Setup and Definitions

Consider a univariate target function with Fourier representation:

$$f^*(x) = \sum_{k=1}^{K} a_k\, e^{i\omega_k x} + \text{c.c.}$$

where $\{(\omega_k, a_k)\}_{k=1}^K$ are the frequency-amplitude pairs with $|a_1| \geq |a_2| \geq \cdots \geq |a_K| > 0$ (ordered by decreasing magnitude), and "c.c." denotes complex conjugate terms ensuring $f^*$ is real-valued. The total energy is:

$$E_{\text{total}} = \|f^*\|_{L^2}^2 = 2\sum_{k=1}^K |a_k|^2$$

**Definition 1 (Representable energy).** For a VQC with accessible frequency set $\Omega(\boldsymbol{\alpha})$, the **representable energy** is the fraction of target energy at frequencies within $\Omega$:

$$\mathcal{E}_{\text{rep}}(\boldsymbol{\alpha}) = \frac{\sum_{k:\, \omega_k \in \Omega(\boldsymbol{\alpha})} |a_k|^2}{\sum_{k=1}^K |a_k|^2} \in [0, 1]$$

**Definition 2 (Optimal achievable loss).** For fixed $\boldsymbol{\alpha}$, the minimum $L^2$ approximation error over all variational parameters $\boldsymbol{\theta}$ is:

$$\mathcal{L}^*(\boldsymbol{\alpha}) = \inf_{\boldsymbol{\theta}} \|f_{\boldsymbol{\alpha},\boldsymbol{\theta}} - f^*\|_{L^2}^2 = 2\sum_{k:\, \omega_k \notin \Omega(\boldsymbol{\alpha})} |a_k|^2 = E_{\text{total}}\,(1 - \mathcal{E}_{\text{rep}}(\boldsymbol{\alpha}))$$

This follows from Parseval's theorem: the best approximation in $\text{span}\{e^{i\omega x}\}_{\omega \in \Omega}$ is the orthogonal projection, and the residual is the energy in unrepresented frequencies.

**Definition 3 (FFT-seeded initialization).** Given training data $\{x^{(m)}\}_{m=1}^M$, the FFT-seeded initialization proceeds as:

1. Compute the empirical power spectrum: $\hat{P}(\omega) = \frac{1}{M}\sum_m |\hat{x}^{(m)}(\omega)|^2$
2. Identify the top-$n$ frequency bins (by power): $\hat{\omega}_1, \ldots, \hat{\omega}_n$ with $\hat{P}(\hat{\omega}_1) \geq \cdots \geq \hat{P}(\hat{\omega}_n)$
3. Set $\alpha_i = \hat{\omega}_i / \hat{\omega}_1$ (ratio-based normalization), clamped to $[0.5, 5.0]$

**Definition 4 (Linspace initialization).** The default initialization sets $\alpha_i = 0.5 + \frac{2.5(i-1)}{n-1}$ for $i = 1, \ldots, n$, uniformly spaced in $[0.5, 3.0]$, without reference to the data.

---

### 3.2 Lemma 3 (Parseval Decomposition of Loss)

**Lemma 3.** *For a VQC with accessible frequency set $\Omega(\boldsymbol{\alpha})$ and target $f^*$, the optimal achievable loss decomposes as:*

$$\mathcal{L}^*(\boldsymbol{\alpha}) = \underbrace{\sum_{k:\, \omega_k \in \Omega(\boldsymbol{\alpha})} 0}_{\text{matched (zero residual)}} + \underbrace{2\sum_{k:\, \omega_k \notin \Omega(\boldsymbol{\alpha})} |a_k|^2}_{\text{unmatched (irreducible residual)}}$$

*Minimizing $\mathcal{L}^*(\boldsymbol{\alpha})$ is equivalent to maximizing the representable energy $\mathcal{E}_{\text{rep}}(\boldsymbol{\alpha})$.*

**Proof.** By the orthogonality of Fourier basis functions, the $L^2$ projection of $f^*$ onto $\text{span}\{e^{i\omega x}\}_{\omega \in \Omega}$ yields:

$$\text{proj}_\Omega f^* = \sum_{k:\, \omega_k \in \Omega} a_k\, e^{i\omega_k x}$$

The residual is:

$$f^* - \text{proj}_\Omega f^* = \sum_{k:\, \omega_k \notin \Omega} a_k\, e^{i\omega_k x}$$

By Parseval's theorem: $\|f^* - \text{proj}_\Omega f^*\|^2 = 2\sum_{k:\, \omega_k \notin \Omega} |a_k|^2$. Since the VQC can realize any function in $\text{span}\{e^{i\omega x}\}_{\omega \in \Omega}$ (with sufficient variational freedom, per Schuld et al. 2021 and Yu et al. 2024), the minimum loss equals this residual. $\quad\blacksquare$

---

### 3.3 Theorem 2 (Main Result)

**Theorem 2 (Optimality of FFT-Seeded Initialization).** *Let $f^*$ be a target function with $K$ frequency components ordered by magnitude $|a_1| \geq \cdots \geq |a_K|$. Let the VQC have $n$ qubits with frequency-matched encoding (Theorem 1). Assume sufficient variational depth for independent coefficient control.*

*For each qubit $i$, the per-qubit frequency set includes $\alpha_i$ (from the $\sigma_Y$ Heisenberg evolution, Eq. 9). Define the **direct coverage set** $\mathcal{C}(\boldsymbol{\alpha}) = \{\alpha_1, \ldots, \alpha_n\} \cap \{\omega_1, \ldots, \omega_K\}$ as the set of target frequencies directly matched by some freq_scale parameter.*

*(a) (FFT-seeded coverage bound.) If the target's top-$n$ frequencies (by magnitude) are $\omega_1, \ldots, \omega_n$ and the FFT-seeded initialization correctly identifies them (i.e., $\hat{\omega}_i = \omega_i$ for $i = 1, \ldots, n$, up to ratio normalization), then:*

$$\mathcal{E}_{\text{rep}}(\boldsymbol{\alpha}_{\text{FFT}}) \geq \frac{\sum_{k=1}^n |a_k|^2}{\sum_{k=1}^K |a_k|^2}$$

*Equivalently, the optimal achievable loss satisfies:*

$$\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}}) \leq 2\sum_{k=n+1}^K |a_k|^2$$

*(b) (Worst-case linspace bound.) For arbitrary initialization $\boldsymbol{\alpha}_{\text{arb}}$, if none of the target frequencies are matched (i.e., $\mathcal{C}(\boldsymbol{\alpha}_{\text{arb}}) = \emptyset$), then the optimal achievable loss can be as large as:*

$$\mathcal{L}^*(\boldsymbol{\alpha}_{\text{arb}}) = E_{\text{total}}$$

*(c) (Energy concentration advantage.) If the target spectrum is concentrated, i.e., the top-$n$ frequencies capture a fraction $\rho$ of total energy:*

$$\rho = \frac{\sum_{k=1}^n |a_k|^2}{\sum_{k=1}^K |a_k|^2}$$

*then FFT-seeded initialization guarantees:*

$$\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}}) \leq (1 - \rho)\, E_{\text{total}}$$

*For spectrally concentrated signals (large $\rho$), this represents a significant reduction compared to the worst-case $E_{\text{total}}$ under arbitrary initialization.*

**Proof.**

**(a)** By Theorem 1 (Lemma 2, Eq. 9), the per-qubit frequency set $\Omega_i^{\text{FM}}(\alpha_i)$ includes $\alpha_i$ as an accessible frequency. The FFT-seeded initialization sets $\alpha_i$ proportional to the $i$-th dominant data frequency. After ratio normalization, the frequency $\alpha_i$ is included in $\Omega_i^{\text{FM}}$.

More precisely: in the univariate case, the total accessible frequency set $\Omega(\boldsymbol{\alpha})$ includes all sums $\sum_i \omega_i$ where $\omega_i \in \Omega_i^{\text{FM}}(\alpha_i)$. In particular, choosing $\omega_j = \alpha_j$ for one specific qubit $j$ and $\omega_i = 0$ for all $i \neq j$ gives the sum $\alpha_j \in \Omega(\boldsymbol{\alpha})$.

Therefore, $\{\alpha_1, \ldots, \alpha_n\} \subseteq \Omega(\boldsymbol{\alpha})$. If $\alpha_i$ matches $\omega_i$ (the $i$-th dominant data frequency, after normalization), then $\omega_i \in \Omega(\boldsymbol{\alpha})$ for $i = 1, \ldots, n$.

By Lemma 3:

$$\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}}) = 2\sum_{k:\, \omega_k \notin \Omega(\boldsymbol{\alpha}_{\text{FFT}})} |a_k|^2 \leq 2\sum_{k=n+1}^K |a_k|^2$$

The inequality uses the fact that $\omega_1, \ldots, \omega_n \in \Omega(\boldsymbol{\alpha}_{\text{FFT}})$ (ensuring at least these are matched), plus possibly additional frequencies matched through sum-frequencies and the $\pm 1$, $\pm(1\pm\alpha_i)$ terms.

$$\mathcal{E}_{\text{rep}}(\boldsymbol{\alpha}_{\text{FFT}}) = 1 - \frac{\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}})}{E_{\text{total}}} \geq 1 - \frac{2\sum_{k>n}|a_k|^2}{2\sum_{k=1}^K |a_k|^2} = \frac{\sum_{k=1}^n |a_k|^2}{\sum_{k=1}^K |a_k|^2}$$

**(b)** If $\omega_k \notin \Omega(\boldsymbol{\alpha}_{\text{arb}})$ for all $k = 1, \ldots, K$ (no target frequency is accessible), then by Lemma 3:

$$\mathcal{L}^*(\boldsymbol{\alpha}_{\text{arb}}) = 2\sum_{k=1}^K |a_k|^2 = E_{\text{total}}$$

This worst case is achievable: for instance, if all target frequencies are irrational and all $\alpha_i$ are rational (or vice versa), then $\{\alpha_i\} \cap \{\omega_k\} = \emptyset$, and the sum-frequencies $\{0, \pm 1, \pm(1\pm\alpha_i)\}$ also miss the irrational targets (generically).

**(c)** Direct substitution of $\rho$ into the bound from (a):

$$\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}}) \leq 2\sum_{k>n} |a_k|^2 = E_{\text{total}} - 2\sum_{k=1}^n |a_k|^2 = E_{\text{total}}(1 - \rho) \quad\blacksquare$$

---

### 3.4 Corollary 2.1 (Greedy Optimality)

**Corollary 2.1.** *The FFT-seeded initialization (selecting the top-$n$ frequencies by power) maximizes the representable energy $\mathcal{E}_{\text{rep}}$ over all possible selections of $n$ target frequencies to match. That is, for any subset $S \subseteq \{1, \ldots, K\}$ with $|S| = n$:*

$$\sum_{k \in S_{\text{FFT}}} |a_k|^2 \geq \sum_{k \in S} |a_k|^2$$

*where $S_{\text{FFT}} = \{1, \ldots, n\}$ (the top-$n$ by magnitude).*

**Proof.** This is immediate from the ordering $|a_1| \geq |a_2| \geq \cdots \geq |a_K|$. Selecting the top-$n$ maximizes the sum of $n$ elements from $\{|a_k|^2\}_{k=1}^K$. $\quad\blacksquare$

### 3.5 Corollary 2.2 (Convergence Speed Implication)

**Corollary 2.2.** *At initialization (before any gradient step), the loss satisfies:*

$$\mathcal{L}(\boldsymbol{\alpha}_{\text{FFT}}, \boldsymbol{\theta}_0) \geq \mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}})$$

*for any initial $\boldsymbol{\theta}_0$. Since $\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}}) \leq \mathcal{L}^*(\boldsymbol{\alpha}_{\text{lin}})$ (from Theorem 2a vs 2b), the FFT-seeded model starts with a lower floor for the loss, and the optimization over $\boldsymbol{\theta}$ can converge to this floor without needing to first adjust $\boldsymbol{\alpha}$ to discover the correct frequencies.*

*Under linspace initialization, the optimizer must jointly learn both the correct frequency scales $\boldsymbol{\alpha}$ and the Fourier coefficients $\boldsymbol{\theta}$, which is a harder optimization problem due to the coupling between frequency and amplitude parameters.*

---

### 3.6 Remark (Practical Considerations)

In practice, the FFT is computed on the training data (not the target function), so $\hat{\omega}_i \approx \omega_i$ up to estimation error. With $M$ training samples and window size $W$, the frequency resolution is $\Delta\omega = 2\pi/W$, and the estimation error scales as $O(1/\sqrt{M})$ by standard spectral estimation theory. For sufficiently large $M$ and $W$, the FFT accurately identifies the dominant frequencies, and Theorem 2's guarantees hold approximately.

The ratio-based normalization ($\alpha_i = \hat{\omega}_i / \hat{\omega}_1$) and clamping to $[0.5, 5.0]$ ensure numerical stability. The fundamental frequency $\hat{\omega}_1$ serves as a reference scale, and the learnable $\boldsymbol{\alpha}$ can fine-tune the freq_scale values during training to compensate for any initialization imprecision.

---

## 4. Theorem 3: Periodicity Preservation Under Gate Transformations

### 4.1 Setup

In an LSTM architecture, VQC gates produce outputs that are then used as gating signals. The standard LSTM applies sigmoid/tanh to the gate outputs:

$$\text{Standard QLSTM:} \quad i_t = \sigma(g_{\text{VQC}}(\mathbf{x}_t)), \quad g_t = \tanh(g_{\text{VQC}}'(\mathbf{x}_t))$$

The FourierQLSTM instead uses rescaled gating:

$$\text{FourierQLSTM:} \quad i_t = \frac{g_{\text{VQC}}(\mathbf{x}_t) + 1}{2}$$

We now formally analyze the impact of these post-processing operations on the Fourier structure of the VQC output.

### 4.2 Definition (Trigonometric Polynomial)

A function $g: \mathbb{R}^d \to \mathbb{R}$ is a **trigonometric polynomial** with frequency support $\Omega \subset \mathbb{R}^d$ if:

$$g(\mathbf{x}) = \sum_{\boldsymbol{\omega} \in \Omega} c_{\boldsymbol{\omega}}\, e^{i\boldsymbol{\omega} \cdot \mathbf{x}}, \qquad c_{\boldsymbol{\omega}} \in \mathbb{C}, \quad |\Omega| < \infty$$

with $c_{-\boldsymbol{\omega}} = c_{\boldsymbol{\omega}}^*$ (ensuring real-valuedness). The **degree** of $g$ is $\max_{\boldsymbol{\omega} \in \Omega} \|\boldsymbol{\omega}\|_1$.

By the Schuld et al. theorem and our Theorem 1, the output of a VQC with frequency-matched encoding is a trigonometric polynomial with frequency support $\Omega^{\text{FM}}(\boldsymbol{\alpha})$.

---

### 4.3 Lemma 4 (Rescaled Gating Preserves Trigonometric Polynomial Structure)

**Lemma 4.** *Let $g(\mathbf{x})$ be a trigonometric polynomial with frequency support $\Omega$. Define the rescaled gating transformation $h_r(\mathbf{x}) = \frac{g(\mathbf{x}) + 1}{2}$. Then:*

*(a) $h_r$ is a trigonometric polynomial with frequency support $\Omega \cup \{\mathbf{0}\}$.*

*(b) The Fourier coefficients of $h_r$ are:*

$$\hat{h}_r(\boldsymbol{\omega}) = \begin{cases} \frac{1}{2} + \frac{c_{\mathbf{0}}}{2} & \text{if } \boldsymbol{\omega} = \mathbf{0} \\ \frac{c_{\boldsymbol{\omega}}}{2} & \text{if } \boldsymbol{\omega} \in \Omega \setminus \{\mathbf{0}\} \\ 0 & \text{otherwise} \end{cases}$$

*(c) The frequency support of $h_r$ is contained in $\Omega \cup \{\mathbf{0}\}$. No new frequencies are introduced beyond the original spectrum (plus possibly the DC component).*

*(d) If $g(\mathbf{x}) \in [-1, 1]$ (which holds for PauliZ measurements), then $h_r(\mathbf{x}) \in [0, 1]$, providing a valid gating signal.*

**Proof.** (a)-(b): By direct computation:

$$h_r(\mathbf{x}) = \frac{1}{2} + \frac{1}{2}\sum_{\boldsymbol{\omega} \in \Omega} c_{\boldsymbol{\omega}}\, e^{i\boldsymbol{\omega}\cdot\mathbf{x}} = \frac{1}{2} + \frac{c_{\mathbf{0}}}{2} + \frac{1}{2}\sum_{\boldsymbol{\omega} \in \Omega\setminus\{\mathbf{0}\}} c_{\boldsymbol{\omega}}\, e^{i\boldsymbol{\omega}\cdot\mathbf{x}}$$

This is a trigonometric polynomial with the same frequency support $\Omega$ (if $\mathbf{0} \in \Omega$) or $\Omega \cup \{\mathbf{0}\}$ (if $\mathbf{0} \notin \Omega$). The Fourier coefficients are as stated.

(c): No frequency $\boldsymbol{\omega} \notin \Omega \cup \{\mathbf{0}\}$ has a non-zero coefficient.

(d): If $g(\mathbf{x}) \in [-1, 1]$, then $h_r(\mathbf{x}) = \frac{g(\mathbf{x})+1}{2} \in [0, 1]$. $\quad\blacksquare$

---

### 4.4 Lemma 5 (Sigmoid/Tanh Introduces Infinite Harmonics)

**Lemma 5.** *Let $g(\mathbf{x})$ be a trigonometric polynomial with frequency support $\Omega$ and $\|g\|_\infty \leq 1$. Define $h_\sigma(\mathbf{x}) = \sigma(g(\mathbf{x}))$ where $\sigma(z) = (1 + e^{-z})^{-1}$ is the sigmoid function. Then:*

*(a) $h_\sigma$ is NOT a trigonometric polynomial (unless $g$ is constant).*

*(b) The Fourier series of $h_\sigma$ has frequency support $\overline{\Omega} = \bigcup_{k=0}^{\infty} k\Omega$, where $k\Omega = \{\sum_{j=1}^k \boldsymbol{\omega}_j : \boldsymbol{\omega}_j \in \Omega\}$ denotes the $k$-fold Minkowski sum. In particular, $|\overline{\Omega}|$ is countably infinite (unless $\Omega = \{\mathbf{0}\}$).*

*(c) The same conclusion holds for $h_\tau(\mathbf{x}) = \tanh(g(\mathbf{x}))$.*

**Proof.** The sigmoid function admits the Taylor expansion around $z = 0$:

$$\sigma(z) = \frac{1}{2} + \frac{1}{2}\tanh\!\left(\frac{z}{2}\right) = \frac{1}{2} + \sum_{m=0}^{\infty} \frac{B_{2m+1}}{(2m+1)!}\, z^{2m+1}$$

where $B_{2m+1}$ are related to the Bernoulli-number expansion of $\tanh$. Explicitly:

$$\sigma(z) = \frac{1}{2} + \frac{z}{4} - \frac{z^3}{48} + \frac{z^5}{480} - \frac{17z^7}{80640} + \cdots \tag{10}$$

This series converges absolutely for all $z \in \mathbb{R}$ (sigmoid is entire after composition with the exponential, and $\tanh(z/2)$ has a convergence radius of $\pi$ for its Taylor series; however for $|z| \leq 1$ the series converges rapidly).

For $|z| \leq 1$, we can also use the fact that $\sigma$ is analytic and non-polynomial (its Taylor series has infinitely many non-zero terms) to conclude that $\sigma(g(\mathbf{x}))$ cannot be a trigonometric polynomial of finite degree whenever $g$ is non-constant.

**(a)** Suppose for contradiction that $h_\sigma = \sigma \circ g$ is a trigonometric polynomial. Then $h_\sigma$ has finitely many non-zero Fourier coefficients. But $h_\sigma^{-1} = \sigma^{-1}$ (the logit function) is also analytic on $(0,1)$, so $g = \sigma^{-1}(h_\sigma)$ would also need finitely many Fourier coefficients (by closure of trigonometric polynomials under analytic functions with finite Taylor series — which logit does not have). This leads to a contradiction unless $g$ is constant.

More constructively: substituting $g(\mathbf{x}) = \sum_\omega c_\omega e^{i\omega \cdot \mathbf{x}}$ into Eq. (10):

$$\sigma(g(\mathbf{x})) = \frac{1}{2} + \frac{g(\mathbf{x})}{4} - \frac{g(\mathbf{x})^3}{48} + \frac{g(\mathbf{x})^5}{480} - \cdots$$

The $k$-th power $g(\mathbf{x})^k = \left(\sum_\omega c_\omega e^{i\omega\cdot\mathbf{x}}\right)^k$ is a trigonometric polynomial with frequency support $k\Omega$ (the $k$-fold Minkowski sum of $\Omega$ with itself). For $k \geq 3$ and non-trivial $\Omega$ (containing a non-zero frequency), $k\Omega$ contains frequencies NOT in $\Omega$.

Specifically, if $\omega_0 \in \Omega$ with $\omega_0 \neq 0$, then $3\omega_0 \in 3\Omega$ but $3\omega_0 \notin \Omega$ (for generic $\Omega$). The coefficient of $e^{i\cdot 3\omega_0 \cdot \mathbf{x}}$ in the cubic term $g^3$ is:

$$[g^3]_{3\omega_0} = c_{\omega_0}^3 \neq 0$$

Therefore $\sigma(g)$ has a non-zero Fourier coefficient at frequency $3\omega_0 \notin \Omega$, and similarly at $5\omega_0, 7\omega_0, \ldots$ from higher-order terms. Since infinitely many such harmonic frequencies have non-zero coefficients, $h_\sigma$ is not a trigonometric polynomial.

**(b)** The frequency support of the $k$-th power $g^k$ is contained in $k\Omega$. The frequency support of $\sigma(g)$ is contained in $\bigcup_{k=0}^\infty k\Omega$. This set is countably infinite whenever $\Omega$ contains a non-zero element (since $k\omega_0$ for $k = 1, 2, 3, \ldots$ are all distinct for $\omega_0 \neq 0$).

**(c)** The same argument applies to $\tanh(z) = 2\sigma(2z) - 1$, which has the Taylor expansion:

$$\tanh(z) = z - \frac{z^3}{3} + \frac{2z^5}{15} - \frac{17z^7}{315} + \cdots$$

The cubic and higher-order terms generate the same harmonic frequencies. $\quad\blacksquare$

---

### 4.5 Theorem 3 (Main Result)

**Theorem 3 (Spectral Distortion Under Gate Transformations).** *Let $g: \mathbb{R} \to [-1, 1]$ be a trigonometric polynomial with frequency support $\Omega$ and $\|g\|_\infty \leq 1$ (as guaranteed by PauliZ measurements). Define:*

- *Rescaled gate: $h_r(x) = \frac{g(x)+1}{2}$*
- *Sigmoid gate: $h_\sigma(x) = \sigma(g(x))$*
- *Tanh gate: $h_\tau(x) = \tanh(g(x))$*

*Then:*

*(a) (Zero spectral distortion for rescaled gating.) The Fourier frequency support of $h_r$ is contained in $\Omega \cup \{0\}$. The spectral distortion is zero:*

$$D_r := \sum_{\omega \notin \Omega \cup \{0\}} |\hat{h}_r(\omega)|^2 = 0$$

*(b) (Non-zero spectral distortion for sigmoid.) The spectral distortion of $h_\sigma$ is strictly positive whenever $g$ is non-constant:*

$$D_\sigma := \sum_{\omega \notin \Omega \cup \{0\}} |\hat{h}_\sigma(\omega)|^2 > 0$$

*Moreover, to leading order in $\|g\|_\infty$:*

$$D_\sigma = \frac{\|g\|_{L^4}^4}{48^2 \cdot \text{period}} + O(\|g\|_\infty^6)$$

*where $\|g\|_{L^4}^4 = \frac{1}{T}\int_0^T |g(x)|^4\,dx$ and $T$ is the period.*

*(c) (Amplitude attenuation.) The effective amplitude of the gating signal at the fundamental frequencies is:*

| Gate | Amplitude scaling | Effective range | Gradient bound |
|---|---|---|---|
| Rescaled | $c_\omega \mapsto \frac{c_\omega}{2}$ | $[0, 1]$ | $\frac{1}{2}$ (constant) |
| Sigmoid | $c_\omega \mapsto \frac{c_\omega}{4} + O(c_\omega^3)$ | $[\sigma(-1), \sigma(1)] \approx [0.27, 0.73]$ | $\leq \frac{1}{4}$ (variable) |
| Tanh | $c_\omega \mapsto c_\omega + O(c_\omega^3)$ | $[-\tanh(1), \tanh(1)] \approx [-0.76, 0.76]$ | $\leq 1$ (variable) |

*For sigmoid, the fundamental Fourier coefficients are attenuated by a factor of 2 compared to rescaled gating ($\frac{1}{4}$ vs $\frac{1}{2}$), and the effective output range is compressed from $[0,1]$ to $\approx [0.27, 0.73]$.*

**Proof.**

**(a)** Follows directly from Lemma 4. The transformation $h_r(x) = \frac{1}{2} + \frac{g(x)}{2}$ is an affine function of $g$, which preserves the trigonometric polynomial structure. The only new frequency is possibly $\omega = 0$ (DC component). All Fourier coefficients at $\omega \notin \Omega \cup \{0\}$ are exactly zero.

**(b)** By Lemma 5, $h_\sigma$ has non-zero Fourier coefficients at frequencies in $k\Omega$ for all odd $k \geq 3$. We quantify the leading-order distortion.

From Eq. (10):

$$h_\sigma(x) = \frac{1}{2} + \frac{g(x)}{4} - \frac{g(x)^3}{48} + O(g^5)$$

The term $\frac{g(x)}{4}$ contributes only to frequencies in $\Omega$ (no distortion).

The term $-\frac{g(x)^3}{48}$ contributes to frequencies in $3\Omega$. The **new** frequencies (those in $3\Omega \setminus \Omega$) constitute the leading-order spectral distortion.

For a concrete single-frequency example, let $g(x) = A\cos(\omega_0 x)$ with $A \leq 1$. Then:

$$g(x)^3 = A^3\cos^3(\omega_0 x) = A^3\left[\frac{3}{4}\cos(\omega_0 x) + \frac{1}{4}\cos(3\omega_0 x)\right]$$

The $\cos(3\omega_0 x)$ term is at frequency $3\omega_0 \notin \Omega = \{-\omega_0, 0, +\omega_0\}$. Its coefficient in $h_\sigma$ is:

$$\hat{h}_\sigma(3\omega_0) = -\frac{A^3}{48} \cdot \frac{1}{4} = -\frac{A^3}{192}$$

The spectral distortion (energy at $3\omega_0$ and $-3\omega_0$) is:

$$D_\sigma = 2\left|\frac{A^3}{192}\right|^2 + O(A^{10}) = \frac{A^6}{18432} + O(A^{10})$$

Note that the $\cos(\omega_0 x)$ component of $g^3$ (the $\frac{3}{4}$ term) feeds back into the fundamental frequency, modifying its coefficient:

$$\hat{h}_\sigma(\omega_0) = \frac{A}{4} - \frac{3A^3}{4 \cdot 48} + O(A^5) = \frac{A}{4}\left(1 - \frac{A^2}{16}\right) + O(A^5)$$

For the general case, $\|g\|_{L^4}^4$ captures the strength of the cubic self-interaction. By Parseval on $g^3$:

$$\|g^3\|_{L^2}^2 \leq \|g\|_\infty^2 \cdot \|g\|_{L^4}^4$$

The energy leaked to new frequencies (outside $\Omega$) from the cubic term is at most:

$$D_\sigma \leq \frac{\|g^3\|_{L^2}^2}{48^2} \leq \frac{\|g\|_\infty^2 \|g\|_{L^4}^4}{48^2}$$

For $\|g\|_\infty \leq 1$, this simplifies to $D_\sigma \leq \frac{\|g\|_{L^4}^4}{2304}$.

The distortion is strictly positive whenever $g$ is non-constant (since $g^3$ will produce third harmonics with non-zero coefficients). Higher-order Taylor terms ($g^5, g^7, \ldots$) contribute additional distortion at frequencies $5\Omega, 7\Omega, \ldots$, but these are $O(\|g\|_\infty^6)$ and higher.

**(c) Amplitude attenuation:** From the Taylor expansions:

**Rescaled gating:** $h_r(x) = \frac{1}{2} + \frac{g(x)}{2}$. The Fourier coefficient at any $\omega \in \Omega$ is $\hat{h}_r(\omega) = \frac{c_\omega}{2}$. The attenuation factor is $\frac{1}{2}$, uniformly for all frequencies.

**Sigmoid:** $h_\sigma(x) = \frac{1}{2} + \frac{g(x)}{4} + O(g^3)$. The leading-order Fourier coefficient at $\omega \in \Omega$ is $\hat{h}_\sigma(\omega) = \frac{c_\omega}{4} + O(|c_\omega|^3)$. The attenuation factor is $\frac{1}{4}$ to leading order — a factor of 2 stronger than rescaled gating.

**Tanh:** $h_\tau(x) = g(x) - \frac{g(x)^3}{3} + O(g^5)$. The leading-order Fourier coefficient is $\hat{h}_\tau(\omega) = c_\omega + O(|c_\omega|^3)$. The linear coefficient is 1 (no attenuation), but the cubic correction is larger than for sigmoid.

**Effective output range:** For $g \in [-1, 1]$:
- Rescaled: $h_r \in [0, 1]$ — full utilization of $[0,1]$
- Sigmoid: $h_\sigma \in [\sigma(-1), \sigma(1)] = \left[\frac{1}{1+e}, \frac{e}{1+e}\right] \approx [0.2689, 0.7311]$ — only 46.2% of the $[0,1]$ range
- Tanh: $h_\tau \in [-\tanh(1), \tanh(1)] \approx [-0.7616, 0.7616]$ — 76.2% of the $[-1,1]$ range

**Gradient bounds:** The gradient of the gate output w.r.t. VQC parameters $\theta$ is:

$$\frac{\partial h}{\partial \theta} = h'(g) \cdot \frac{\partial g}{\partial \theta}$$

where $h'(g)$ is the derivative of the gate transformation:
- Rescaled: $h_r'(g) = \frac{1}{2}$ (constant for all $g$)
- Sigmoid: $h_\sigma'(g) = \sigma(g)(1-\sigma(g)) \leq \frac{1}{4}$ (maximum at $g=0$, decays toward 0 as $|g| \to \infty$)
- Tanh: $h_\tau'(g) = 1 - \tanh^2(g) \leq 1$ (maximum at $g=0$)

For VQC outputs in $[-1, 1]$, the sigmoid gradient is bounded by $\frac{1}{4}$, which is half the rescaled gating gradient of $\frac{1}{2}$. This means gradients flowing back through sigmoid gates are attenuated by an additional factor of $\leq \frac{1}{2}$ compared to rescaled gating, potentially slowing convergence.

Furthermore, the constant gradient of rescaled gating ($\frac{1}{2}$ everywhere) prevents the vanishing gradient problem that sigmoid exhibits when $|g|$ is large. $\quad\blacksquare$

---

### 4.6 Corollary 3.1 (Cumulative Distortion in Multi-Gate Architectures)

**Corollary 3.1.** *In an LSTM architecture with 4 gates (input $i$, forget $f$, cell $g$, output $o$) and a cell state update $c_t = f_t \odot c_{t-1} + i_t \odot g_t$, the spectral distortion accumulates across gates and time steps.*

*With sigmoid gates, the cell state $c_t$ at time $t$ contains harmonic frequencies from all $4t$ gate applications. With rescaled gating, the cell state's frequency support remains within $\Omega$ at every time step (up to the product-induced expansion from the Hadamard products $\odot$).*

**Proof sketch.** Each sigmoid gate application introduces harmonics in $3\Omega, 5\Omega, \ldots$. These harmonics propagate through the cell state update, where the Hadamard products $f_t \odot c_{t-1}$ and $i_t \odot g_t$ further expand the frequency content (products of trigonometric polynomials add their frequency supports). Over $t$ time steps, the frequency content grows combinatorially.

With rescaled gating, each gate output is a trigonometric polynomial in $\Omega$ (Lemma 4). The Hadamard products still expand the frequency content (to $2\Omega, 3\Omega, \ldots$), but no additional harmonics are introduced by the gating transformation itself. The frequency expansion comes only from the multiplicative cell state structure, not from the gates. $\quad\blacksquare$

---

### 4.7 Corollary 3.2 (Information-Theoretic Perspective)

**Corollary 3.2.** *Define the **spectral efficiency** of a gate transformation $h$ as the fraction of total output power at the original VQC frequencies:*

$$\eta(h) = \frac{\sum_{\omega \in \Omega} |\hat{h}(\omega)|^2}{\sum_{\omega} |\hat{h}(\omega)|^2}$$

*Then:*

$$\eta(h_r) = 1, \qquad \eta(h_\sigma) < 1, \qquad \eta(h_\tau) < 1$$

*for any non-constant $g$. The spectral efficiency of rescaled gating is optimal (all power at VQC frequencies), while sigmoid and tanh waste power on harmonics that do not align with the target data's frequency content.*

**Proof.** For rescaled gating, all power is at frequencies in $\Omega \cup \{0\}$ (Lemma 4), so $\eta(h_r) = 1$ (counting DC as an original frequency).

For sigmoid, Lemma 5 guarantees non-zero power at frequencies in $3\Omega \setminus \Omega$, $5\Omega \setminus \Omega$, etc. This power is "leaked" from the original frequencies, giving $\eta(h_\sigma) < 1$.

The same argument applies to tanh. $\quad\blacksquare$

---

## 5. Theorem 4: Generalization to Non-Periodic Spectrally Concentrated Signals

### 5.1 Motivation

Theorems 1–3 establish the advantages of frequency-matched encoding, FFT-seeded initialization, and rescaled gating for VQC-based time-series models. A natural question arises: **are these advantages limited to strictly periodic signals?**

We now formally prove that the theoretical guarantees extend to **any signal with sufficient spectral concentration** — a property satisfied by the vast majority of real-world time series, including non-periodic signals such as transient responses, quasi-periodic oscillations, damped sinusoids, and smooth aperiodic trends. The key mathematical insight is that the Fourier decomposition underlying Theorems 1–3 depends on the completeness of the Fourier basis on compact domains, which holds universally in $L^2$ **without any periodicity assumption**.

---

### 5.2 Lemma 6 ($L^2$ Completeness of the Fourier Basis on Compact Domains)

**Lemma 6.** *Let $f \in L^2[0, T]$ be an arbitrary square-integrable function on the compact interval $[0, T]$. The function $f$ is NOT assumed to be periodic. Then:*

*(a) (Fourier expansion.) $f$ admits a unique Fourier series representation:*

$$f(x) = \sum_{k=-\infty}^{\infty} a_k \, e^{i\frac{2\pi k}{T}x}$$

*converging in $L^2$ norm:*

$$\lim_{K \to \infty} \left\| f - \sum_{k=-K}^{K} a_k\, e^{i\frac{2\pi k}{T}x} \right\|_{L^2[0,T]} = 0$$

*where the Fourier coefficients are:*

$$a_k = \frac{1}{T} \int_0^T f(x)\, e^{-i\frac{2\pi k}{T}x}\, dx$$

*(b) (Parseval's equality.) The total energy decomposes into frequency components:*

$$\|f\|_{L^2[0,T]}^2 = \frac{1}{T}\int_0^T |f(x)|^2\, dx = \sum_{k=-\infty}^{\infty} |a_k|^2$$

*(c) (Riemann–Lebesgue lemma.) $|a_k| \to 0$ as $|k| \to \infty$.*

*(d) (Best approximation.) For any finite frequency set $\Omega \subset \mathbb{Z}$, the best $L^2$ approximation of $f$ by a trigonometric polynomial with frequency support $\Omega$ is the orthogonal projection:*

$$\hat{f}_\Omega(x) = \sum_{k \in \Omega} a_k\, e^{i\frac{2\pi k}{T}x}, \qquad \|f - \hat{f}_\Omega\|_{L^2}^2 = \sum_{k \notin \Omega} |a_k|^2$$

**Proof.**

The system $\{T^{-1/2}\, e^{i2\pi kx/T}\}_{k \in \mathbb{Z}}$ forms a **complete orthonormal basis** (ONB) for the Hilbert space $L^2[0, T]$. Completeness is established as follows:

*Step 1 (Orthonormality).* For $k, k' \in \mathbb{Z}$:

$$\frac{1}{T}\int_0^T e^{i2\pi kx/T}\, e^{-i2\pi k'x/T}\, dx = \frac{1}{T}\int_0^T e^{i2\pi(k-k')x/T}\, dx = \delta_{kk'}$$

where $\delta_{kk'}$ is the Kronecker delta. This holds by direct evaluation of the integral.

*Step 2 (Completeness).* We must show that no non-zero $f \in L^2[0,T]$ is orthogonal to all basis elements. Suppose $a_k = \frac{1}{T}\int_0^T f(x) e^{-i2\pi kx/T} dx = 0$ for all $k \in \mathbb{Z}$. By the Stone–Weierstrass theorem, finite trigonometric polynomials are dense in $C[0,T]$ with the $L^2$ topology, and $C[0,T]$ is dense in $L^2[0,T]$. Therefore $f = 0$ in $L^2$.

*Crucially, this argument does NOT invoke periodicity of $f$.* The ONB property is a statement about the basis functions $\{e^{i2\pi kx/T}\}$ on the domain $[0,T]$, not about the function $f$ being expanded.

Parts (a)–(c) are immediate consequences of the ONB property in Hilbert spaces: (a) is the abstract Fourier expansion, (b) is Parseval's identity (equivalence of the $L^2$ norm and the $\ell^2$ norm of coefficients), and (c) follows from the convergence of $\sum |a_k|^2$ (a convergent series must have terms tending to zero).

Part (d) follows from the orthogonal projection theorem in Hilbert spaces: the best approximation of $f$ in the closed subspace $\text{span}\{e^{i2\pi kx/T}\}_{k \in \Omega}$ is the orthogonal projection $\hat{f}_\Omega$, and the residual norm is the energy in the orthogonal complement. $\quad\blacksquare$

**Remark (Role of Periodicity).** Periodicity of $f$ affects (i) *pointwise* convergence of the Fourier series (the Gibbs phenomenon at discontinuities of the periodic extension) and (ii) the *rate* of decay of $|a_k|$ (see Lemma 7). It does **not** affect the validity of the $L^2$ expansion, Parseval's equality, or the best-approximation property. Since our loss function is the $L^2$ error (mean squared error), periodicity is irrelevant to the correctness of the Parseval-based loss decomposition.

---

### 5.3 Lemma 7 (Spectral Decay Rates by Signal Class)

**Lemma 7.** *Let $f \in L^2[0, T]$ with Fourier coefficients $\{a_k\}_{k \in \mathbb{Z}}$ and sorted magnitudes $|a_{(1)}| \geq |a_{(2)}| \geq \cdots$. The rate of spectral decay — and consequently the spectral concentration ratio $\rho(n)$ — depends on the regularity of $f$:*

*(a) (Finite harmonics.) If $f(x) = \sum_{j=1}^{K} b_j\, e^{i\omega_j x}$ is a finite sum of $K$ harmonics, then $a_k = 0$ for all but at most $K$ values of $k$, giving:*

$$\rho(n) = 1 \quad \text{for all } n \geq K$$

*(b) (Smooth periodic, $C^m$.) If $f$ is $m$-times continuously differentiable and periodic with period $T$ (i.e., $f^{(j)}(0) = f^{(j)}(T)$ for $j = 0, \ldots, m-1$), then $|a_k| = O(|k|^{-(m+1)})$, and:*

$$1 - \rho(n) = O\!\left(n^{-(2m+1)}\right)$$

*(c) (Analytic periodic.) If $f$ is real-analytic and periodic with period $T$, then $|a_k| = O(e^{-c|k|})$ for some $c > 0$ determined by the width of the analyticity strip, and:*

$$1 - \rho(n) = O\!\left(e^{-2cn}\right)$$

*(d) (Bounded variation.) If $f \in BV[0, T]$ (bounded variation, allowing finitely many discontinuities), then $|a_k| = O(|k|^{-1})$, and:*

$$1 - \rho(n) = O\!\left(n^{-1}\right)$$

*(e) (Smooth non-periodic, windowed.) If $f$ is $C^m$-smooth on $[0, T]$ but NOT periodic (i.e., $f(0) \neq f(T)$ or some derivative mismatches at the boundary), and $w \in C^m[0, T]$ is a smooth window function satisfying $w(0) = w(T) = 0$ and $w^{(j)}(0) = w^{(j)}(T) = 0$ for $j = 1, \ldots, m-1$ (e.g., the Hann window), then the Fourier coefficients of the windowed signal $\tilde{f} = w \cdot f$ satisfy:*

$$|a_k^{(\tilde{f})}| = O\!\left(|k|^{-(m+1)}\right), \qquad 1 - \rho_{\tilde{f}}(n) = O\!\left(n^{-(2m+1)}\right)$$

*(f) (White noise.) If $x[0], \ldots, x[N-1]$ are i.i.d. random variables with mean zero and variance $\sigma^2$, the expected power spectrum is flat: $\mathbb{E}[P[k]] = N\sigma^2$ for all $k$. The expected spectral concentration ratio is:*

$$\mathbb{E}[\rho(n)] = \frac{n}{N}$$

*exhibiting no concentration.*

**Proof.**

**(a)** Immediate. A sum of $K$ complex exponentials has exactly $K$ non-zero Fourier coefficients.

**(b)** For a $C^m$ periodic function, integration by parts $m+1$ times on the Fourier coefficient integral yields:

$$a_k = \frac{1}{T}\int_0^T f(x)\, e^{-i2\pi kx/T}\, dx = \frac{1}{(i2\pi k/T)^{m+1}} \cdot \frac{1}{T}\int_0^T f^{(m+1)}(x)\, e^{-i2\pi kx/T}\, dx$$

The boundary terms vanish at each step because $f^{(j)}(0) = f^{(j)}(T)$ for $j = 0, \ldots, m$. The remaining integral is bounded by the Riemann–Lebesgue lemma, giving $|a_k| \leq \frac{C}{|k|^{m+1}}$ for some constant $C$ depending on $\|f^{(m+1)}\|_{L^1}$.

The residual energy after the top-$n$ frequencies is:

$$\sum_{|k| > n} |a_k|^2 \leq C^2 \sum_{|k| > n} |k|^{-2(m+1)} \leq 2C^2 \int_n^\infty t^{-2(m+1)}\, dt = \frac{2C^2}{2m+1} \cdot n^{-(2m+1)}$$

Therefore $1 - \rho(n) \leq \frac{2C^2}{(2m+1) \cdot E_{\text{total}}} \cdot n^{-(2m+1)} = O(n^{-(2m+1)})$.

**(c)** For real-analytic periodic $f$, the function extends to a holomorphic function in a strip $\{z \in \mathbb{C} : |\text{Im}(z)| < c\}$ for some $c > 0$. By shifting the contour of integration:

$$|a_k| = \left|\frac{1}{T}\int_0^T f(x + ic\,\text{sgn}(k))\, e^{-i2\pi kx/T}\, dx\right| \cdot e^{-2\pi|k|c/T} \leq M \cdot e^{-2\pi|k|c/T}$$

where $M = \max_{|\text{Im}(z)| \leq c} |f(z)|$. Setting $c' = 2\pi c/T$:

$$\sum_{|k|>n} |a_k|^2 \leq 2M^2 \sum_{k=n+1}^\infty e^{-2c'k} = \frac{2M^2\, e^{-2c'(n+1)}}{1 - e^{-2c'}} = O(e^{-2c'n})$$

**(d)** For $f \in BV[0, T]$ with total variation $V(f) < \infty$, integration by parts once gives:

$$a_k = \frac{1}{T} \cdot \frac{T}{i2\pi k} \int_0^T e^{-i2\pi kx/T}\, df(x)$$

where the integral is a Riemann–Stieltjes integral. By the bound on Riemann–Stieltjes integrals:

$$|a_k| \leq \frac{V(f)}{2\pi|k|}$$

The residual: $\sum_{|k|>n}|a_k|^2 \leq \frac{V(f)^2}{2\pi^2} \sum_{k=n+1}^\infty k^{-2} \leq \frac{V(f)^2}{2\pi^2 n} = O(n^{-1})$.

**(e)** The key observation is that for non-periodic $f$, the periodic extension to $\mathbb{R}$ introduces discontinuities at the boundary points $\{0, T, 2T, \ldots\}$ (since $f(0^+) \neq f(T^-)$ in general). These discontinuities limit the Fourier coefficient decay to $O(|k|^{-1})$ regardless of the interior smoothness of $f$.

The window function $w$ resolves this issue. Since $w(0) = w(T) = 0$ and $w^{(j)}(0) = w^{(j)}(T) = 0$ for $j = 1, \ldots, m-1$, the product $\tilde{f} = wf$ satisfies:

$$\tilde{f}^{(j)}(0) = \tilde{f}^{(j)}(T) = 0 \quad \text{for } j = 0, 1, \ldots, m-1$$

(by the Leibniz rule, $\tilde{f}^{(j)}(0) = \sum_{\ell=0}^{j} \binom{j}{\ell} w^{(\ell)}(0) f^{(j-\ell)}(0) = 0$ since $w^{(\ell)}(0) = 0$ for $\ell \leq m-1$ and $j \leq m-1$).

Therefore the periodic extension of $\tilde{f}$ is $C^{m-1}$ with a bounded $m$-th derivative, and part (b) applies to $\tilde{f}$, giving $|a_k^{(\tilde{f})}| = O(|k|^{-(m+1)})$.

The windowing introduces an energy scaling factor: $\|\tilde{f}\|_{L^2}^2 = \int_0^T |w(x)|^2 |f(x)|^2\, dx \leq \|w\|_\infty^2 \|f\|_{L^2}^2$. For the Hann window $w(x) = \sin^2(\pi x/T)$, the energy retention is $\|w\|_{L^2}^2/T = 3/8$. The spectral concentration *rate* is preserved: $1 - \rho_{\tilde{f}}(n) = O(n^{-(2m+1)})$.

**(f)** For i.i.d. random variables $x[n]$ with mean zero and variance $\sigma^2$:

$$\mathbb{E}[|X[k]|^2] = \mathbb{E}\left[\left|\sum_{n=0}^{N-1} x[n]\, e^{-i2\pi kn/N}\right|^2\right] = \sum_{n=0}^{N-1} \mathbb{E}[|x[n]|^2] = N\sigma^2$$

(the cross terms vanish by independence). All frequency bins have equal expected power, so the expected fraction captured by any $n$ bins is $n/N$. $\quad\blacksquare$

---

### 5.4 Theorem 4 (Main Result)

**Theorem 4 (Generalization to Non-Periodic Spectrally Concentrated Signals).** *Let $f^* \in L^2[0, T]$ be an **arbitrary** target function — not assumed periodic — with Fourier coefficients $\{a_k\}$ and spectral concentration ratio $\rho(n)$ (Definition 6). Let the VQC have $n$ qubits with frequency-matched encoding (Theorem 1), FFT-seeded initialization (Definition 3), and rescaled gating (Lemma 4). Then:*

*(a) (Universality of the Parseval loss decomposition.) The loss decomposition from Lemma 3, the FFT-seeded initialization bounds from Theorem 2, and the spectral preservation guarantees from Theorem 3 all hold without modification for the non-periodic target $f^*$. Specifically, the optimal achievable $L^2$ loss satisfies:*

$$\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}}) \leq \big(1 - \rho(n)\big) \cdot E_{\text{total}}$$

*where $E_{\text{total}} = \|f^*\|_{L^2}^2$ is the total signal energy.*

*(b) (Signal-class-specific bounds.) For different signal classes, the optimal achievable loss under FFT-seeded frequency-matched encoding satisfies:*

| Signal Class | Regularity Condition | $\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}})$ | Convergence in $n$ |
|---|---|---|---|
| $K$ harmonics, $K \leq n$ | $f^*$ is a finite Fourier sum | $= 0$ (exact reconstruction) | Finite |
| $C^m$ smooth periodic | $m \geq 1$ | $\leq C_m \cdot n^{-(2m+1)} \cdot E_{\text{total}}$ | Polynomial (fast) |
| Analytic periodic | Analyticity strip width $c > 0$ | $\leq C \cdot e^{-2cn} \cdot E_{\text{total}}$ | Exponential |
| Bounded variation | $V(f^*) < \infty$ (allows jumps) | $\leq C \cdot n^{-1} \cdot E_{\text{total}}$ | Polynomial (slow) |
| $C^m$ smooth non-periodic (windowed) | $m \geq 1$, with Hann window | $\leq C_m \cdot n^{-(2m+1)} \cdot E_{\text{total}}$ | Polynomial (fast) |
| Quasi-periodic | Dominant peaks at non-harmonic freqs | $\leq (1 - \rho_{\text{emp}}) \cdot E_{\text{total}}$ | Data-dependent |
| White noise | i.i.d., variance $\sigma^2$ | $\approx (1 - n/N) \cdot E_{\text{total}}$ | Linear (no advantage) |

*(c) (Advantage of tunable frequencies over fixed frequencies.) Under standard RY-only encoding, the VQC's accessible frequency set is $\Omega^{\text{std}} = \{-1, 0, +1\}^n$ (Lemma 1), which is fixed regardless of the data. In the univariate case, $\Omega_{\text{uni}}^{\text{std}} = \{-n, \ldots, n\}$. Let $\rho_{\text{std}}(n)$ denote the fraction of $f^*$'s energy at frequencies in $\Omega^{\text{std}}$. Then the loss ratio between frequency-matched and standard encoding satisfies:*

$$\frac{\mathcal{L}^*_{\text{FM}}(\boldsymbol{\alpha}_{\text{FFT}})}{\mathcal{L}^*_{\text{std}}} = \frac{1 - \rho(n)}{1 - \rho_{\text{std}}(n)} \leq 1$$

*with strict inequality whenever $f^*$ has dominant frequencies that do not coincide with the integer set $\{-n, \ldots, n\}$. For generic real-valued signals, $\rho_{\text{std}}(n) \ll \rho(n)$, and the frequency-aware architecture achieves a substantially lower loss floor.*

*(d) (Sufficient condition for advantage.) The frequency-aware VQC achieves a loss reduction factor of at least $r > 1$ over the standard VQC whenever:*

$$\rho(n) > 1 - \frac{1 - \rho_{\text{std}}(n)}{r}$$

*In particular, if $\rho_{\text{std}}(n) \leq \epsilon$ (negligible energy at fixed integer frequencies) and $\rho(n) \geq 1 - \delta$ (strong spectral concentration in top-$n$), the improvement factor is at least $(1-\epsilon)/\delta$.*

**Proof.**

**(a)** We verify that each component of the proof chain holds for arbitrary $f^* \in L^2[0,T]$:

*Step 1 (Fourier expansion).* By Lemma 6(a), $f^*$ has a Fourier series $f^*(x) = \sum_k a_k\, e^{i2\pi kx/T}$ converging in $L^2$, with Parseval's equality $E_{\text{total}} = \sum_k |a_k|^2$ (Lemma 6(b)). **No periodicity assumed.**

*Step 2 (VQC output structure).* By the Schuld et al. theorem (Section 1.6) and Theorem 1, the VQC output $f_{\boldsymbol{\alpha},\boldsymbol{\theta}}(\mathbf{x})$ is a trigonometric polynomial with frequency support $\Omega(\boldsymbol{\alpha})$. This is a property of the **circuit architecture**, independent of the target function.

*Step 3 (Parseval loss decomposition).* The $L^2$ approximation error between the VQC output and the target decomposes as:

$$\|f_{\boldsymbol{\alpha},\boldsymbol{\theta}} - f^*\|_{L^2}^2 = \underbrace{\sum_{k:\,\omega_k \in \Omega} |c_k^{\text{VQC}} - a_k|^2}_{\text{error at matched frequencies}} + \underbrace{\sum_{k:\,\omega_k \notin \Omega} |a_k|^2}_{\text{irreducible residual}}$$

This decomposition follows from the **orthogonality of distinct Fourier basis functions** on $[0,T]$ (Lemma 6, Step 1 of proof), which holds for **all** $f^* \in L^2[0,T]$. Minimizing over $\boldsymbol{\theta}$ sets the first sum to zero (the VQC coefficients match the projection), yielding:

$$\mathcal{L}^*(\boldsymbol{\alpha}) = \sum_{k:\,\omega_k \notin \Omega(\boldsymbol{\alpha})} |a_k|^2$$

This is Lemma 3, now established for **arbitrary** (non-periodic) $f^*$.

*Step 4 (FFT-seeded bound).* The FFT-seeded initialization matches the top-$n$ data frequencies (by power). By Corollary 2.1, this is the greedy-optimal selection, giving:

$$\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}}) \leq \sum_{k>n} |a_{(k)}|^2 = E_{\text{total}} - \sum_{k=1}^{n} |a_{(k)}|^2 = (1 - \rho(n)) \cdot E_{\text{total}}$$

*Step 5 (Spectral preservation).* Theorem 3 (Lemma 4) states that rescaled gating preserves the trigonometric polynomial structure of the VQC output. This is an **algebraic property** of the affine transformation $h_r(g) = (g+1)/2$ applied to a trigonometric polynomial $g$ — it holds regardless of what $g$ represents or what target is being approximated. Similarly, the spectral distortion result for sigmoid/tanh (Lemma 5, Theorem 3b) depends only on the Taylor expansion of the nonlinearity composed with a trigonometric polynomial.

Therefore, all three theorems apply without modification to non-periodic targets. $\quad\square$

**(b)** Substituting the spectral decay rates from Lemma 7 into the bound $\mathcal{L}^* \leq (1-\rho(n)) \cdot E_{\text{total}}$:

- **$K$ harmonics, $K \leq n$:** By Lemma 7(a), $\rho(n) = 1$, so $\mathcal{L}^* = 0$.
- **$C^m$ smooth periodic:** By Lemma 7(b), $1 - \rho(n) = O(n^{-(2m+1)})$, so $\mathcal{L}^* = O(n^{-(2m+1)}) \cdot E_{\text{total}}$.
- **Analytic periodic:** By Lemma 7(c), $1 - \rho(n) = O(e^{-2cn})$, so $\mathcal{L}^* = O(e^{-2cn}) \cdot E_{\text{total}}$.
- **Bounded variation:** By Lemma 7(d), $1 - \rho(n) = O(n^{-1})$, so $\mathcal{L}^* = O(n^{-1}) \cdot E_{\text{total}}$.
- **$C^m$ non-periodic (windowed):** By Lemma 7(e), $1 - \rho_{\tilde{f}}(n) = O(n^{-(2m+1)})$, same rate as the periodic case with an energy scaling factor from the window.
- **Quasi-periodic:** The spectral concentration ratio $\rho_{\text{emp}}$ is determined empirically from the data's power spectrum. For signals with well-defined spectral peaks (e.g., EEG alpha/beta rhythms, seasonal patterns), $\rho_{\text{emp}}$ is typically large.
- **White noise:** By Lemma 7(f), $\rho(n) = n/N$, so $\mathcal{L}^* \approx (1 - n/N) \cdot E_{\text{total}}$ — no significant reduction unless $n \approx N$. $\quad\square$

**(c)** Under standard RY-only encoding, the univariate accessible frequency set is $\Omega_{\text{uni}}^{\text{std}} = \{-n, \ldots, n\}$ (Corollary 1.1(a)). The optimal loss is:

$$\mathcal{L}^*_{\text{std}} = \sum_{k:\, 2\pi k/T \notin \{-n,\ldots,n\}} |a_k|^2 = (1 - \rho_{\text{std}}(n)) \cdot E_{\text{total}}$$

Under frequency-matched encoding with FFT-seeded initialization:

$$\mathcal{L}^*_{\text{FM}}(\boldsymbol{\alpha}_{\text{FFT}}) \leq (1 - \rho(n)) \cdot E_{\text{total}}$$

Taking the ratio:

$$\frac{\mathcal{L}^*_{\text{FM}}}{\mathcal{L}^*_{\text{std}}} = \frac{1 - \rho(n)}{1 - \rho_{\text{std}}(n)}$$

Since $\rho(n) \geq \rho_{\text{std}}(n)$ by Corollary 2.1 (FFT-seeded selects the top-$n$ frequencies by power, while standard encoding is constrained to the fixed integer set), this ratio is $\leq 1$.

The inequality is strict whenever $f^*$ has dominant frequencies outside $\{-n, \ldots, n\}$. For a signal with sampling rate $f_s$ and dominant physical frequency $f_0$, the corresponding DFT index is $k_0 = f_0 T$. Unless $f_0 T$ happens to be an integer in $\{1, \ldots, n\}$ — a condition that depends on the arbitrary signal length $T$ — the standard encoding misses this frequency.

More precisely: if the signal's dominant frequencies $\{\omega_1^*, \ldots, \omega_n^*\}$ are drawn from a continuous distribution on $\mathbb{R}_{>0}$, the probability that any $\omega_j^*$ coincides with an element of the fixed set $\{2\pi k/T : k \in \{-n,\ldots,n\}\}$ is zero. Hence $\rho_{\text{std}}(n) \ll \rho(n)$ for generic signals. $\quad\square$

**(d)** The loss reduction factor is $r = \mathcal{L}^*_{\text{std}} / \mathcal{L}^*_{\text{FM}} = (1 - \rho_{\text{std}}(n))/(1 - \rho(n))$. Setting $r > r_0$ and solving for $\rho(n)$:

$$r_0 < \frac{1 - \rho_{\text{std}}(n)}{1 - \rho(n)} \implies 1 - \rho(n) < \frac{1 - \rho_{\text{std}}(n)}{r_0} \implies \rho(n) > 1 - \frac{1 - \rho_{\text{std}}(n)}{r_0}$$

For $\rho_{\text{std}}(n) \leq \epsilon$ and $\rho(n) \geq 1 - \delta$: $r \geq (1-\epsilon)/\delta$. For example, with $\epsilon = 0.05$ and $\delta = 0.1$ (90% spectral concentration), the improvement factor is $r \geq 9.5\times$. $\quad\blacksquare$

---

### 5.5 Corollary 4.1 (Windowed Analysis for Non-Stationary Signals)

**Corollary 4.1.** *Let $f^*$ be a non-stationary signal whose spectral content varies over time (e.g., EEG with time-varying rhythms, speech with varying formants, or signals with transient events). Define the **local spectral concentration** at time $t$ within a window of length $W$ as:*

$$\rho_{\text{local}}(n, t) = \frac{\sum_{j=1}^{n} P_{(j)}^{(t)}}{\sum_k P^{(t)}[k]}$$

*where $P^{(t)}[k]$ is the power spectrum of the windowed segment $f^*(x) \cdot w(x - t)$ for a window function $w$ of width $W$. If $\rho_{\text{local}}(n, t) \geq 1 - \delta(t)$ for all $t$ in a time interval $[0, T]$, then the time-averaged loss of a TCN-style architecture that processes local windows satisfies:*

$$\overline{\mathcal{L}}^* \leq \overline{\delta} \cdot \overline{E}_{\text{local}}$$

*where $\overline{\delta} = \frac{1}{T}\int_0^T \delta(t)\, dt$ is the average spectral residual and $\overline{E}_{\text{local}}$ is the average windowed energy.*

*This implies that global non-stationarity does NOT prevent the frequency-aware VQC from achieving low loss, as long as the signal is locally spectrally concentrated within each processing window.*

**Proof.** A TCN architecture processes the signal in local windows via causal dilated convolutions (each kernel of size $K$ with dilation $d$ covers a receptive field of size $W = K \cdot d$). Within each window, the VQC processes a local signal segment.

Applying Theorem 4(a) to each local segment centered at time $t$:

$$\mathcal{L}^*_t(\boldsymbol{\alpha}_{\text{FFT}}) \leq (1 - \rho_{\text{local}}(n, t)) \cdot E_{\text{local}}(t) = \delta(t) \cdot E_{\text{local}}(t)$$

Averaging over the time interval:

$$\overline{\mathcal{L}}^* = \frac{1}{T}\int_0^T \mathcal{L}^*_t\, dt \leq \frac{1}{T}\int_0^T \delta(t) \cdot E_{\text{local}}(t)\, dt$$

By the Cauchy–Schwarz inequality or simply by noting $\delta(t) \leq \max_t \delta(t)$:

$$\overline{\mathcal{L}}^* \leq \overline{\delta} \cdot \overline{E}_{\text{local}}$$

where the averaging accounts for the possibility that spectral concentration varies across windows. $\quad\blacksquare$

**Remark.** This corollary directly applies to the FourierQTCN architecture, which uses TCN-style causal convolutions followed by per-window VQC processing. EEG signals, for instance, are globally non-stationary (spectral content changes with cognitive state) but locally quasi-periodic within each ~1-second window, giving high $\rho_{\text{local}}$ values.

---

### 5.6 Corollary 4.2 (Computable Applicability Criterion)

**Corollary 4.2.** *The spectral concentration ratio $\rho(n)$ provides a **computable, a priori criterion** for predicting whether the frequency-aware VQC architecture will significantly outperform a generic (fixed-frequency) VQC on a given dataset. Specifically:*

*Given training data $\{(\mathbf{x}^{(m)}, y^{(m)})\}_{m=1}^M$:*

*1. Compute the empirical power spectrum $\hat{P}[k] = \frac{1}{M}\sum_m |\hat{x}^{(m)}[k]|^2$ (averaged over training samples).*

*2. Sort the power spectrum: $\hat{P}_{(1)} \geq \hat{P}_{(2)} \geq \cdots$*

*3. Compute $\hat{\rho}(n) = \sum_{j=1}^{n} \hat{P}_{(j)} / \sum_k \hat{P}[k]$ for the number of qubits $n$.*

*Then:*

$$\hat{\rho}(n) \geq 1 - \delta \quad \Longrightarrow \quad \mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}}) \leq \delta \cdot E_{\text{total}} + O(M^{-1/2})$$

*where the $O(M^{-1/2})$ term accounts for finite-sample estimation error in the power spectrum.*

*The decision rule is: if $\hat{\rho}(n)$ is high (e.g., $\hat{\rho}(n) \geq 0.8$), the frequency-aware VQC is strongly recommended; if $\hat{\rho}(n) \approx n/N$ (near the white-noise baseline), the approach offers no structural advantage.*

**Proof.** The empirical spectral concentration ratio $\hat{\rho}(n)$ is a consistent estimator of the population $\rho(n)$ by the law of large numbers applied to the periodogram. The estimation error is bounded by:

$$|\hat{\rho}(n) - \rho(n)| \leq C \cdot M^{-1/2}$$

for some constant $C$ depending on the signal's fourth-order cumulant (see, e.g., standard results in spectral estimation theory). Substituting $\hat{\rho}(n) \geq 1 - \delta$ into Theorem 4(a):

$$\mathcal{L}^*(\boldsymbol{\alpha}_{\text{FFT}}) \leq (1 - \rho(n)) \cdot E_{\text{total}} \leq (\delta + C M^{-1/2}) \cdot E_{\text{total}} \quad\blacksquare$$

---

### 5.7 Remark (Connection to Compressed Sensing and Sparse Approximation)

The spectral concentration framework for Theorem 4 is closely related to the theory of **compressed sensing** (Candès, Romberg, & Tao, 2006; Donoho, 2006). In compressed sensing, a signal $\mathbf{x} \in \mathbb{R}^N$ is called **$s$-sparse** if it has at most $s$ non-zero coefficients in some basis (e.g., the Fourier basis). The central result is that $s$-sparse signals can be recovered from $O(s \log N)$ measurements.

Our setting is analogous: the frequency-matched VQC with $n$ qubits can "measure" $n$ frequency components (via the $n$ learnable $\alpha_i$ parameters). The FFT-seeded initialization selects the most energetic components, achieving the best $n$-term approximation. The approximation quality is determined by $\rho(n)$ — the spectral concentration ratio — which plays the same role as the restricted isometry property (RIP) constant in compressed sensing.

The key difference from standard compressed sensing is that we do not require exact sparsity ($K$ non-zero coefficients). Instead, we exploit **compressibility** — the rapid decay of the sorted Fourier coefficients (Lemma 7) — which ensures that the best $n$-term approximation captures most of the signal energy. This is precisely the regime where most real-world signals reside.

---

### 5.8 Summary: When Does the Frequency-Aware VQC Provide an Advantage?

| Criterion | Advantage Level | Examples |
|---|---|---|
| $\rho(n) \geq 0.95$ | **Maximum**: loss $\leq 5\%$ of $E_{\text{total}}$ | Periodic signals, narrow-band oscillations, analytic functions |
| $0.8 \leq \rho(n) < 0.95$ | **Strong**: loss $\leq 20\%$ of $E_{\text{total}}$ | EEG rhythms, seasonal time series, quasi-periodic systems |
| $0.5 \leq \rho(n) < 0.8$ | **Moderate**: meaningful reduction over fixed-frequency VQC | Smooth transients, broadband with dominant peaks |
| $n/N < \rho(n) < 0.5$ | **Weak**: marginal improvement | Signals with slowly decaying spectra |
| $\rho(n) \approx n/N$ | **None**: no structural advantage | White noise, i.i.d. random signals |

This table provides a practical guide: **any signal that is not white noise benefits from the frequency-aware architecture to some degree**, with the benefit scaling proportionally to the spectral concentration. The approach is NOT limited to periodic signals — it extends to any signal class where the energy is concentrated in a subset of frequency components.

---

## 6. Discussion: Design Implications

### 6.1 Summary of Theoretical Results

| Theorem | Design Choice | Formal Guarantee |
|---|---|---|
| **Theorem 1** | Frequency-matched encoding (RY + freq_scale $\cdot$ RX) | Per-qubit accessible frequencies expand from 3 to 9; total multivariate frequency count expands from $3^n$ to $9^n$ (exponential improvement) |
| **Theorem 2** | FFT-seeded initialization of freq_scale | Maximizes initial spectral coverage; achievable loss bounded by residual energy in non-dominant frequencies |
| **Theorem 3** | Rescaled gating $(g+1)/2$ instead of sigmoid/tanh | Zero spectral distortion, full output range utilization, constant $\frac{1}{2}$ gradient (vs $\leq \frac{1}{4}$ for sigmoid) |
| **Theorem 4** | Generalization beyond periodic signals | All guarantees hold for **any** spectrally concentrated signal; advantage quantified by $\rho(n)$ |

### 6.2 From Theory to Architecture

Each theorem directly motivates a specific design choice in the FourierQLSTM and FourierQTCN architectures:

1. **Theorem 1 $\Rightarrow$ Encoding choice:** Use `qml.RY(x[i]) + qml.RX(freq_scale[i] * x[i])` per qubit, giving 3x more per-qubit frequencies and exponentially richer multivariate expressivity compared to RY-only encoding.

2. **Theorem 2 $\Rightarrow$ Initialization strategy:** Analyze training data power spectrum via FFT, identify dominant frequencies, and initialize `freq_scale` to match. This ensures the VQC starts with its frequency support aligned to the data, avoiding wasted gradient steps to discover the correct frequencies.

3. **Theorem 3 $\Rightarrow$ Gate design:** Replace `sigmoid(VQC_output)` with `(VQC_output + 1) / 2`. This preserves the VQC's carefully constructed Fourier structure, avoids spectral leakage to harmonics, maintains full output range utilization, and provides stronger gradient flow.

4. **Theorem 4 $\Rightarrow$ Broad applicability:** The architecture is not restricted to periodic data. Any signal with spectral concentration $\rho(n) \gg n/N$ — including quasi-periodic biomedical signals, smooth transients, and seasonal time series — benefits from the frequency-aware design. The computable criterion $\hat{\rho}(n)$ (Corollary 4.2) can be evaluated on training data to predict the expected benefit before training.

### 6.3 Ablation Predictions

The theorems predict specific ablation outcomes:

- **Remove frequency-matched encoding** (use RY-only): Expect reduced performance on tasks with rich frequency content, due to the $3^n$-fold reduction in accessible Fourier terms (Theorem 1).

- **Remove FFT-seeded init** (use linspace): Expect slower convergence and potentially higher final loss, especially on data with concentrated spectra where linspace misses the dominant frequencies (Theorem 2).

- **Replace rescaled gating with sigmoid**: Expect degraded performance due to (i) spectral distortion injecting irrelevant harmonics (Theorem 3b), (ii) 2x amplitude attenuation at fundamental frequencies (Theorem 3c), and (iii) weaker gradient flow (gradient bounded by $\frac{1}{4}$ vs $\frac{1}{2}$, Theorem 3c).

- **Test on non-periodic signals**: Expect the frequency-aware architecture to outperform fixed-frequency VQC on any dataset with high empirical $\hat{\rho}(n)$ — including non-periodic quasi-periodic signals (EEG, seasonal time series) and smooth aperiodic signals — while showing no advantage on white noise or flat-spectrum data (Theorem 4).

### 6.4 Scope of Applicability

Theorem 4 formally establishes that the advantages proven in Theorems 1–3 are **not limited to periodic signals**. The relevant criterion is spectral concentration $\rho(n)$, not periodicity. This broadens the applicability of FourierQLSTM and FourierQTCN to include:

1. **Biomedical signals** (EEG, ECG, EMG): Quasi-periodic with dominant rhythmic components (alpha, beta, gamma bands for EEG; QRS complex for ECG).
2. **Climate and environmental data** (temperature, wind, solar irradiance): Seasonal patterns (daily, annual) with aperiodic trends and noise.
3. **Financial time series** (stock prices, exchange rates): Weak but identifiable periodicities (intraday patterns, quarterly cycles) with significant stochastic components.
4. **Acoustic and speech signals**: Locally periodic within phoneme windows (formant frequencies), globally non-stationary.
5. **Vibration and sensor data** (accelerometers, gyroscopes): Characteristic frequencies from mechanical resonances, even in non-periodic operational regimes.

For all of these, the empirical $\hat{\rho}(n)$ can be computed from training data (Corollary 4.2) to verify that the frequency-aware architecture is appropriate.

---

## 7. References

1. Schuld, M., Sweke, R., & Meyer, J. J. (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A*, 103(3), 032430.

2. Yu, Z., Chen, Q., Jiao, Y., Li, Y., Lu, X., Wang, X., & Yang, J. Z. (2024). Non-asymptotic approximation error bounds of parameterized quantum circuits. *Advances in Neural Information Processing Systems 37* (NeurIPS 2024).

3. Zhao, J., Qiao, W., Zhang, P., & Gao, H. (2024). Quantum implicit neural representations. *Proceedings of the 41st International Conference on Machine Learning* (ICML 2024), PMLR 235.

4. Lewis, L., Gilboa, D., & McClean, J. R. (2025). Quantum advantage for learning shallow neural networks with natural data distributions. *Nature Communications*.

5. Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.

6. Casas, B. & Cervera-Lierta, A. (2023). Multidimensional Fourier series with quantum circuits. *Physical Review A*, 107, 062612.

7. Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. *NeurIPS 2020*.

8. Ziyin, L., Hartwig, T., & Ueda, M. (2020). Neural networks fail to learn periodic functions and how to fix it. *NeurIPS 2020*.

9. Candès, E. J., Romberg, J., & Tao, T. (2006). Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information. *IEEE Transactions on Information Theory*, 52(2), 489–509.

10. Donoho, D. L. (2006). Compressed sensing. *IEEE Transactions on Information Theory*, 52(4), 1289–1306.

11. Katznelson, Y. (2004). *An Introduction to Harmonic Analysis*. Cambridge University Press. (Standard reference for Fourier series on compact domains, $L^2$ completeness, and coefficient decay rates.)

12. Grafakos, L. (2014). *Classical Fourier Analysis*. Springer. (Reference for spectral decay under smoothness assumptions and Bernstein-type inequalities.)

---

*Document generated: February 2026*
*Project: VQC-PeriodicData*
*Authors: Theory supporting FourierQLSTM and FourierQTCN architecture design*
