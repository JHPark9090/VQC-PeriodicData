# Theoretical Gaps Analysis and Recommendations for Publication

**Date:** February 13, 2026
**Purpose:** Assess whether the current theoretical framework in the VQC-PeriodicData project is sufficient for top-tier AI/ML venues (NeurIPS, ICML, ICLR) and identify specific gaps and remedies.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [What the Current Docs Contain](#what-the-current-docs-contain)
3. [Critical Theoretical Gaps](#critical-theoretical-gaps)
4. [Key Published Papers That Fill These Gaps](#key-published-papers-that-fill-these-gaps)
5. [Experimental Weaknesses](#experimental-weaknesses)
6. [Recommendations for a Publishable Paper](#recommendations-for-a-publishable-paper)
7. [Proposed Paper Structure](#proposed-paper-structure)
8. [References](#references)

---

## Executive Summary

The documentation in this project contains well-organized **informal arguments** and **qualitative observations** about VQC advantages on periodic data. However, it **does not contain formal theorems, proofs, or quantified bounds** — the minimum standard for a theory contribution at NeurIPS/ICML.

The good news: three recently published papers (NeurIPS 2024, ICML 2024, Nature Communications 2025) provide exactly the formal foundations this work needs. The project should **cite and build on** these results rather than attempt to prove everything from scratch, and position the contribution as architecture design + empirical validation.

### Assessment at a Glance

| Question | Answer |
|----------|--------|
| Are the informal arguments correct? | **Yes** — the intuitions about VQC = Fourier series and periodic advantage are sound |
| Are they sufficient as a theory contribution? | **No** — no formal theorems, no error bounds, no separation results |
| Can they serve as paper motivation/background? | **Yes** — they provide excellent intuition for a methods paper |
| Do published papers exist with the needed formal results? | **Yes** — three key papers from 2024-2025 |

---

## What the Current Docs Contain

### Inventory of Theoretical Claims Across All Docs

| Document | Core Claim | Type of Argument |
|----------|-----------|------------------|
| `VQC_Universal_Extrapolation_Analysis.md` | VQCs achieve universal extrapolation for periodic functions | Informal argument based on Fourier representation |
| `VQC_Periodic_Data_Performance_Analysis.md` | VQCs have provable advantages: zero approximation error, better sample efficiency, generalization, extrapolation | Mixed — some justified by Schuld et al., others unproven |
| `VQC_vs_ReLU_Periodic_Data_Comparison.md` | VQCs outperform ReLU/tanh for periodic extrapolation (ReLU→linear, tanh→constant, VQC→periodic) | Qualitative observation (well-known, Ziyin et al. 2020) |
| `VQC_vs_SIREN_Comparison.md` | SIREN also computes Fourier series; VQC and SIREN have identical representational structure | Honest analysis — **undermines** quantum advantage narrative |
| `VQC_vs_Snake_Practical_Comparison.md` | VQCs cannot outperform Snake for real-world data because non-periodic components are essential | Honest analysis — identifies limitations |
| `Hybrid_VQC_Snake_Architecture_Analysis.md` | Hybrid VQC-Snake partially solves VQC limitations but introduces signal decomposition problem | Architectural analysis |
| `VQC_Periodic_Data_Encoding_Guide.md` | Practical encoding strategies: data rescaling, bandwidth extension, learnable rescaling, multi-frequency targeting | Practical guide, not theory |
| `QTCN_Periodic_Advantage_Analysis.md` | Original HQTCN2_EEG.py does NOT utilize VQC periodic advantage | Code audit — correct |
| `QLSTM_Periodic_Advantage_Analysis.md` | Original QLSTM_v0.py destroys VQC periodic advantage via sigmoid/tanh | Code audit — correct |

### What Is and Isn't Proven

| Element | Status | Notes |
|---------|--------|-------|
| VQC output = truncated Fourier series | **Cited** (Schuld et al. 2021) | Prior work, not a contribution |
| VQC outputs are inherently periodic | **Trivially follows** | Not publishable as a new result |
| ReLU extrapolates linearly, VQC extrapolates periodically | **Qualitative observation** | Well-known (Ziyin et al. 2020), not novel |
| SIREN computes the same Fourier series as VQC | **Acknowledged honestly** | Undermines quantum advantage claim |
| Formal approximation error bounds for VQC | **Missing entirely** | Critical gap |
| Sample complexity: VQC vs classical for periodic functions | **Missing entirely** | Critical gap |
| Generalization bounds (PAC-Bayes, Rademacher, VC-dim) | **Missing entirely** | Critical gap |
| Formal separation result: VQC vs classical on periodic data | **Missing entirely** | Most critical gap |
| Parameter efficiency: VQC vs classical (formal) | **Informal only** | Observed 100x fewer params, no theorem |

---

## Critical Theoretical Gaps

### Gap 1: The SIREN Equivalence Problem (Most Fundamental)

**The problem:** The docs correctly identify that VQC output is a Fourier series:

$$f_{\text{VQC}}(x) = \sum_{\omega \in \Omega} c_\omega e^{i\omega x}$$

But `VQC_vs_SIREN_Comparison.md` also correctly identifies that SIREN computes:

$$f_{\text{SIREN}}(x) = \sin(\omega_0 \cdot (Wx + b))$$

which is also a Fourier series. A NeurIPS/ICML reviewer will immediately ask:

> *"If both VQC and SIREN compute Fourier series, what exactly is the quantum advantage? Why not just use SIREN?"*

The docs acknowledge this equivalence but never resolve it. Without resolving this, the paper has no theoretical contribution.

**Resolution (from QIREN, ICML 2024):** The answer is **exponential parameter efficiency**. A VQC with n qubits and L re-uploading layers can represent up to $O(L \cdot 2^n)$ Fourier frequencies, while a classical FNN (including SIREN) requires explicit parameters per frequency — i.e., the number of representable frequencies grows **linearly** with model size. This is the formal separation between VQC and SIREN.

### Gap 2: No Formal Approximation Bounds

**The problem:** The docs claim "approximation error = 0 for periodic targets" (`VQC_Periodic_Data_Performance_Analysis.md`), but this is only true when the target's frequency spectrum is a subset of the VQC's accessible frequencies — a very restrictive condition.

What's missing is a formal statement like:

> *"Theorem: For a target function $f \in \mathcal{F}_K$ (Fourier series with K non-zero frequencies), a PQC with n qubits and L layers achieves approximation error $\|f - f_{\text{PQC}}\|_\infty \leq \varepsilon(n, L, K)$ where $\varepsilon$ is characterized by..."*

No such bound exists anywhere in the docs.

**Resolution (from Yu et al., NeurIPS 2024):** This paper provides exactly these bounds — the first non-asymptotic approximation error bounds for PQCs in terms of qubits, circuit depth, and number of trainable parameters. Crucially, they show that PQC circuit size can be **smaller than deep ReLU networks** for the same approximation quality.

### Gap 3: No Formal Separation Result (VQC vs Classical)

**The problem:** The docs argue qualitatively that VQCs should be better at periodic data, but provide no formal proof that classical algorithms *cannot* efficiently learn what quantum algorithms can.

A NeurIPS reviewer would say:

> *"You show VQC can learn periodic functions. But you haven't shown that classical methods can't. Maybe ReLU with enough parameters and proper initialization does equally well."*

Indeed, the experimental results (`Benchmark_Experiment_Results.md`) show ReLU-LSTM beating FourierQLSTM on Multi-Sine (the purest periodic benchmark), suggesting classical methods can learn periodic functions just fine given enough parameters.

**Resolution (from Lewis et al., Nature Communications 2025):** This paper proves an **exponential quantum advantage** for learning periodic neurons $f(x) = g(\mathbf{w} \cdot \mathbf{x})$ where $g$ is periodic. They show:
- Classical gradient-based algorithms provably **cannot** learn periodic neurons efficiently
- Classical statistical query (SQ) algorithms also **cannot** learn them efficiently
- Quantum algorithms **can** learn them efficiently
- This holds for **non-uniform** distributions (Gaussians, mixtures of Gaussians, Schwartz functions) — i.e., natural data distributions, not just artificial uniform inputs

This is precisely the separation result the docs are missing.

### Gap 4: Informal "Extrapolation Advantage" Argument

**The problem:** The extrapolation argument (ReLU→linear, tanh→constant, VQC→periodic outside training domain) is presented as a key advantage. But:

1. This observation is well-known from Ziyin et al. (2020) — not a new contribution
2. It's never formalized with error bounds on extrapolation quality
3. It's never tested experimentally (all current benchmarks test interpolation, not extrapolation)
4. Snake activation $\text{Snake}_a(x) = x + \frac{1}{a}\sin^2(ax)$ also extrapolates periodically

**Resolution:** This requires both formal analysis AND experimental validation:
- Formally: bound the extrapolation error of VQC vs ReLU/tanh on $[T, 2T]$ after training on $[0, T]$
- Experimentally: add extrapolation benchmarks (see [Experimental Weaknesses](#experimental-weaknesses))

---

## Key Published Papers That Fill These Gaps

### Paper 1: Non-asymptotic Approximation Error Bounds of Parameterized Quantum Circuits

- **Authors:** Zhan Yu, Qiuhao Chen, Yuling Jiao, Yinan Li, Xiliang Lu, Xin Wang, Jerry Zhijian Yang
- **Venue:** NeurIPS 2024 (Poster)
- **arXiv:** [2310.07528](https://arxiv.org/abs/2310.07528)
- **Published proceedings:** [NeurIPS 2024, Paper 94787](https://neurips.cc/virtual/2024/poster/94787)

#### Summary

This paper provides the **first non-asymptotic approximation error bounds** for parameterized quantum circuits (PQCs) with data re-uploading, expressed in terms of the number of qubits, quantum circuit depth, and number of trainable parameters.

#### Key Results

1. **Constructive universality:** They explicitly construct data re-uploading PQCs for approximating multivariate polynomials and smooth functions (previous universality results were either nonconstructive or relied on classical preprocessing, making it unclear whether the expressivity came from the classical or quantum parts).

2. **PQC vs ReLU comparison:** For multivariate polynomials and smooth functions, the quantum circuit size and parameter count of their proposed PQCs can be **smaller than deep ReLU neural networks** achieving the same approximation quality.

3. **Explicit bounds:** For a target function in a Sobolev ball $\mathcal{W}^{s,\infty}([0,1]^d)$, they bound the approximation error in terms of:
   - $n$: number of qubits
   - $L$: number of re-uploading layers (circuit depth)
   - $|\theta|$: number of trainable parameters

4. **Scope:** The results cover general smooth function classes, not just periodic functions. However, for periodic functions (which are smooth), the bounds are particularly favorable because PQCs naturally compute Fourier series.

#### Relevance to This Project

- **Fills Gap 2** (no formal approximation bounds): Provides the formal error bounds absent from the docs
- **Partially fills Gap 1** (SIREN equivalence): Shows PQCs need fewer parameters than ReLU for the same approximation quality, though doesn't directly compare to SIREN
- **Important caveat:** The authors note that for hybrid models (classical preprocessing + PQC), it's unclear whether expressivity comes from the classical or quantum part — directly relevant to FourierQLSTM/FourierQTCN which use FFT preprocessing

#### How to Cite in Your Work

Use as the formal backing for claims about VQC approximation power. Replace informal arguments like "VQCs can approximate periodic functions with zero error" with references to the specific bounds from this paper.

---

### Paper 2: Quantum Implicit Neural Representations (QIREN)

- **Authors:** Jiaming Zhao, Wenbo Qiao, Peng Zhang, Hui Gao
- **Venue:** ICML 2024
- **arXiv:** [2406.03873](https://arxiv.org/abs/2406.03873)
- **Published proceedings:** [ICML 2024, Proceedings of Machine Learning Research Vol. 235](https://icml.cc/Downloads/2024)

#### Summary

This paper proposes QIREN (Quantum Implicit Representation Network), a quantum generalization of Fourier Neural Networks (FNNs) for implicit neural representations. Through theoretical analysis, they demonstrate that QIREN possesses a **quantum advantage over classical FNNs** specifically in terms of Fourier series representation capacity.

#### Key Results

1. **Exponential Fourier advantage:** A data re-uploading quantum circuit with $n$ qubits can represent $O(2^n)$ Fourier frequencies. A classical Fourier Neural Network with $N$ parameters represents $O(N)$ frequencies. This gives an **exponential advantage** in the number of representable frequencies per parameter.

2. **Multi-layer multi-qubit analysis:** Previous work (Schuld et al. 2021, Yu et al. 2022) only analyzed single-qubit multi-layer or multi-qubit single-layer circuits. QIREN extends this to the practically relevant multi-layer multi-qubit case.

3. **Empirical validation:** Experiments on signal representation, image super-resolution, and image generation show QIREN outperforms classical SOTA (including SIREN and RFF-based MLPs) with fewer parameters.

4. **Direct comparison with SIREN:** QIREN explicitly compares against SIREN (Sitzmann et al. 2020) and RFF-MLP (Tancik et al. 2020), the two strongest classical Fourier baselines.

#### Relevance to This Project

- **Directly resolves Gap 1** (SIREN equivalence problem): While both VQC and SIREN compute Fourier series, VQC represents exponentially more frequencies per parameter. This is the key differentiator that the docs are missing.
- **Supports the "100x fewer parameters" observation:** The project's docs note that FourierQLSTM uses ~199 parameters vs ~19,600 for classical LSTMs. QIREN's theory explains *why* — exponential Fourier expressivity.
- **Provides a formal framework:** The theoretical analysis can be adapted to justify FourierQLSTM and FourierQTCN's parameter efficiency.

#### How to Cite in Your Work

Use as the formal resolution to the SIREN equivalence problem. When arguing why VQC is preferred over SIREN despite both computing Fourier series, cite QIREN's exponential advantage result. This transforms the narrative from "VQC and SIREN are equivalent" to "VQC is exponentially more parameter-efficient than SIREN for the same Fourier expressivity."

---

### Paper 3: Quantum Advantage for Learning Shallow Neural Networks with Natural Data Distributions

- **Authors:** Laura Lewis, Dar Gilboa, Jarrod R. McClean
- **Venue:** Nature Communications (published December 31, 2025)
- **DOI:** [10.1038/s41467-025-68097-2](https://www.nature.com/articles/s41467-025-68097-2)
- **Affiliations:** Google Quantum AI

#### Summary

This paper proves an **exponential quantum advantage** for learning periodic neurons — functions of the form $f(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x})$ where $g$ is a periodic function — over non-uniform distributions of classical data. This is the most directly relevant paper to the VQC-PeriodicData project's theoretical claims.

#### Key Results

1. **Formal problem definition:** Learning periodic neurons: given access to labeled examples $(\mathbf{x}, f(\mathbf{x}))$ drawn from a distribution $\mathcal{D}$ where $f(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x})$ and $g$ is periodic, find $\mathbf{w}$ and $g$.

2. **Classical hardness (gradient-based algorithms):** Any gradient-based classical algorithm cannot efficiently learn periodic neurons when the input data distribution has a sufficiently sparse Fourier transform. This condition is satisfied by many natural distributions including:
   - Gaussian distributions
   - Mixtures of Gaussians
   - Schwartz functions (rapidly decaying functions)

   This strengthens prior work (Goel et al.) to apply to their specific parameter regime.

3. **Classical hardness (statistical query algorithms):** The hardness result extends beyond gradient-based methods to the broader class of **statistical query (SQ) algorithms**, which encompasses virtually all practical classical ML algorithms.

4. **Quantum efficiency:** They design an efficient quantum algorithm in the **Quantum Statistical Query (QSQ)** model that can learn periodic neurons over these non-uniform distributions. The quantum algorithm runs in polynomial time where classical algorithms require exponential time.

5. **Non-uniform distributions:** Unlike prior work on quantum learning advantages (which typically assumed uniform input distributions), this result holds for **natural, non-uniform distributions** — making it relevant to real-world data.

6. **Real-valued functions:** This is the first explicit treatment of real-valued function learning in the QSQ model (prior work focused on Boolean functions), making it directly applicable to regression tasks like those in this project.

#### Relevance to This Project

- **Directly fills Gap 3** (no formal separation result): This is exactly the separation theorem the docs need. It proves quantum algorithms have an exponential advantage over classical algorithms for learning periodic functions.
- **Validates the project's premise:** The project's central hypothesis — that VQCs should have an advantage on periodic data — is formally proven correct by this paper.
- **Addresses the "ReLU can also learn periodic functions" objection:** Yes, ReLU can fit periodic functions with enough parameters on interpolation tasks, but this paper shows classical algorithms (including gradient-based ones like those training ReLU networks) provably cannot *learn* periodic functions efficiently in general.
- **Natural distributions:** The result holds for Gaussian inputs, directly relevant to standardized EEG/time-series data.

#### Important Caveats

1. **QSQ model vs practical circuits:** The quantum algorithm operates in the Quantum Statistical Query model. Translating this to specific VQC architectures (like FourierQLSTM) is non-trivial and not addressed.
2. **Learning vs optimization:** The paper proves efficient *learnability*, not that any specific VQC training procedure converges efficiently. The gap between theoretical learnability and practical trainability remains.
3. **Periodic neurons vs general periodic functions:** The result is for single-layer periodic neurons $g(\mathbf{w} \cdot \mathbf{x})$, not arbitrary periodic functions. Time-series with multiple periodic components require extension of the theory.
4. **Sample complexity vs computational complexity:** The advantage is computational (exponential time separation), not sample complexity. Both classical and quantum algorithms may need similar numbers of samples.

#### How to Cite in Your Work

Use as the **primary theoretical justification** for the project's existence. The paper structure should cite this in the introduction and theoretical background to establish that quantum advantage for periodic data is not just a heuristic intuition but a **formally proven exponential separation**. Acknowledge the caveats (QSQ model, periodic neurons specifically, practical gap) honestly.

---

## Experimental Weaknesses

Beyond theory, the current experimental results have issues that NeurIPS/ICML reviewers would flag:

### Issue 1: ReLU-LSTM Wins on the Purest Periodic Benchmark

| Dataset | FourierQLSTM | ReLU-LSTM | Expected Winner |
|---------|-------------|-----------|-----------------|
| NARMA-10 (broadband) | **0.0554** | 0.0673 | Toss-up (control) |
| Multi-Sine K=5 (pure periodic) | 0.009506 | **0.000783** | FourierQLSTM should win |
| Mackey-Glass (quasi-periodic) | 0.012190 | **0.003309** | FourierQLSTM should win |

The theory predicts VQC advantage on periodic data, but FourierQLSTM loses on Multi-Sine (the purest Fourier test) and Mackey-Glass (quasi-periodic). A reviewer will see this as the theory being contradicted by experiments.

**Mitigation:** The comparison is parameter-unfair (199 vs 19,599 parameters). Add parameter-matched comparisons and per-parameter efficiency metrics.

### Issue 2: No Extrapolation Benchmarks

All current benchmarks test **interpolation** (train and test from the same temporal range). The strongest theoretical advantage of VQC — periodic extrapolation — is never tested.

**Mitigation:** Add extrapolation experiments:
- Train on $[0, T]$, test on $[T, 2T]$ (one period ahead)
- Train on $[0, T]$, test on $[2T, 3T]$ (two periods ahead)
- Plot extrapolation error as a function of distance from training domain
- Compare VQC (periodic extrapolation) vs ReLU (linear extrapolation) vs tanh (constant extrapolation)

### Issue 3: Unfair Parameter Comparison

| Model | Parameters |
|-------|-----------|
| FourierQLSTM | ~199 |
| ReLU-LSTM | ~19,599 |

Comparing a 199-parameter model to a 19,599-parameter model and declaring the larger model "wins" is not meaningful. The fair comparison is:

1. **Equal parameters:** Give ReLU-LSTM only ~199 parameters (reduce hidden dim and MLP dim)
2. **Equal compute:** Give both models the same wall-clock time budget
3. **Pareto frontier:** Plot accuracy vs parameter count for both model families

### Issue 4: Only 50 Epochs

50 epochs may be insufficient for FourierQLSTM to converge, especially with the smaller parameter budget. Classical models with 100x more parameters may converge faster simply due to overparameterization.

### Issue 5: Limited to Univariate/Small-Scale

The benchmarks (NARMA, Multi-Sine, Mackey-Glass) are all small-scale. For NeurIPS/ICML, reviewers expect results on at least some standard benchmarks. The ETTh1/Weather/ECL datasets are a good start but only have 1-epoch verification results so far.

---

## Recommendations for a Publishable Paper

### Recommended Framing: Methods/Architecture Paper

**Do not frame as a theory paper.** The formal results are already proven by Yu et al. (NeurIPS 2024), QIREN (ICML 2024), and Lewis et al. (Nature Comms 2025). Trying to re-prove these would be redundant.

**Instead, frame as:** *"Principled architecture design for VQC-based time-series models, informed by the Fourier structure of variational quantum circuits."*

The contribution becomes:
1. **Architecture design:** FourierQLSTM and FourierQTCN as the first time-series models that explicitly exploit VQC's Fourier series structure through frequency-matched encoding, FFT-seeded initialization, and periodicity-preserving gates
2. **Design principles:** Replace sigmoid/tanh (which destroys VQC periodicity) with rescaled gating; use FFT preprocessing instead of FC projection; use learnable frequency scaling
3. **Empirical validation:** Demonstrate advantages on periodic/quasi-periodic benchmarks, especially extrapolation tasks and parameter-efficiency comparisons

### Concrete Action Items

#### Theory Section (Background/Motivation)

- [ ] Cite Schuld et al. (2021) for VQC = Fourier series (existing)
- [ ] Cite Yu et al. (NeurIPS 2024) for formal approximation bounds — replace informal "zero error" claims with their specific bounds
- [ ] Cite QIREN (ICML 2024) for exponential parameter efficiency over classical FNNs — resolves the SIREN equivalence problem
- [ ] Cite Lewis et al. (Nature Comms 2025) for the formal separation result on periodic neurons — establishes that quantum advantage for periodic data is provably real, not just heuristic
- [ ] Clearly distinguish what is prior work (theory) vs what is your contribution (architecture + experiments)

#### Architecture Contribution

- [ ] Formalize the design principles as a coherent framework:
  - Principle 1: Preserve VQC periodicity (no sigmoid/tanh post-processing)
  - Principle 2: Match VQC frequencies to data frequencies (FFT-seeded initialization)
  - Principle 3: Operate in frequency domain (FFT preprocessing, magnitude/phase cell state)
- [ ] Show ablation studies: what happens when each principle is removed?

#### Experimental Improvements (Critical)

- [ ] **Add extrapolation benchmarks:** Train on $[0, T]$, test on $[T, 2T]$ for Multi-Sine, Mackey-Glass, ETTh1. This is where the theoretical advantage should manifest most clearly.
- [ ] **Add parameter-matched comparisons:** Reduce classical model sizes to ~200 parameters. Show VQC wins at equal parameter budgets.
- [ ] **Add Pareto efficiency plots:** Accuracy vs parameter count curves for all models.
- [ ] **Run full 50-100 epoch experiments on ETTh1/Weather/ECL:** The 1-epoch verification results are insufficient.
- [ ] **Add SIREN baseline with proper initialization:** Current SIREN results use w0=30.0 (standard for images), which is too high for time-series. Sweep w0 in {1, 5, 10, 30} for a fair comparison.
- [ ] **Add Snake baseline with proper initialization:** Sweep a_init in {0.1, 0.5, 1.0, 5.0}.
- [ ] **Statistical significance:** Run each experiment with 3-5 seeds and report mean +/- std.

---

## Proposed Paper Structure

```
Title: "Frequency-Matched Variational Quantum Circuits for Periodic Time-Series Learning"

1. Introduction
   - VQCs compute Fourier series (Schuld et al. 2021)
   - Exponential quantum advantage for learning periodic neurons (Lewis et al. 2025)
   - Existing quantum time-series models (QLSTM, QTCN) don't exploit this structure
   - Our contribution: principled architecture design that aligns VQC structure with data

2. Background and Related Work
   2.1 VQC as Fourier Series (Schuld et al. 2021)
   2.2 Approximation Bounds (Yu et al., NeurIPS 2024)
   2.3 Exponential Parameter Efficiency (QIREN, ICML 2024)
   2.4 Quantum Advantage for Periodic Data (Lewis et al., Nature Comms 2025)
   2.5 Classical Periodic Activations: SIREN, Snake

3. Analysis: Why Existing Quantum Time-Series Models Fail
   3.1 QLSTM: Sigmoid/tanh destroys VQC periodicity
   3.2 QTCN: Mean aggregation and FC projection lose frequency information
   3.3 Design principles for periodic-aware quantum models

4. FourierQLSTM and FourierQTCN Architecture
   4.1 FFT-based frequency extraction (preserve periodic structure)
   4.2 Frequency-matched encoding with learnable freq_scale
   4.3 FFT-seeded initialization (data-informed freq_scale)
   4.4 Rescaled gating (periodicity-preserving alternative to sigmoid)
   4.5 Frequency-domain cell state (magnitude/phase representation)

5. Experiments
   5.1 Benchmarks: NARMA-10, Multi-Sine, Mackey-Glass, ETTh1
   5.2 Baselines: ReLU-LSTM/TCN, Tanh-LSTM/TCN, SIREN-LSTM/TCN, Snake-LSTM/TCN
   5.3 Interpolation results (standard next-step prediction)
   5.4 Extrapolation results (train on [0,T], test on [T,2T]) [KEY EXPERIMENT]
   5.5 Parameter efficiency (Pareto frontier analysis) [KEY EXPERIMENT]
   5.6 Ablation studies (each design principle)
   5.7 Learned frequency analysis (freq_scale evolution during training)

6. Discussion
   6.1 When does VQC advantage manifest? (Periodic vs non-periodic data)
   6.2 Computational cost and quantum hardware prospects
   6.3 Limitations (simulation overhead, non-periodic components)

7. Conclusion
```

---

## References

### Papers to Cite (Critical)

1. **Schuld, M., Sweke, R., & Meyer, J.J.** (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A*, 103(3), 032430. [arXiv:2008.08605](https://arxiv.org/abs/2008.08605)
   - Foundation: VQC outputs are truncated Fourier series

2. **Yu, Z., Chen, Q., Jiao, Y., Li, Y., Lu, X., Wang, X., & Yang, J.Z.** (2024). Non-asymptotic Approximation Error Bounds of Parameterized Quantum Circuits. *Advances in Neural Information Processing Systems 37* (NeurIPS 2024), 99089. [arXiv:2310.07528](https://arxiv.org/abs/2310.07528)
   - Formal approximation bounds; PQC parameter count can be smaller than ReLU networks

3. **Zhao, J., Qiao, W., Zhang, P., & Gao, H.** (2024). Quantum Implicit Neural Representations. *Proceedings of the 41st International Conference on Machine Learning* (ICML 2024), PMLR 235. [arXiv:2406.03873](https://arxiv.org/abs/2406.03873)
   - Exponential Fourier advantage of data re-uploading circuits over classical FNNs; resolves VQC vs SIREN comparison

4. **Lewis, L., Gilboa, D., & McClean, J.R.** (2025). Quantum advantage for learning shallow neural networks with natural data distributions. *Nature Communications*. [DOI:10.1038/s41467-025-68097-2](https://www.nature.com/articles/s41467-025-68097-2)
   - Exponential quantum advantage for learning periodic neurons over non-uniform distributions; separation from classical gradient-based and SQ algorithms

### Papers to Cite (Supporting)

5. **Ziyin, L., Hartwig, T., & Ueda, M.** (2020). Neural networks fail to learn periodic functions and how to fix it. *NeurIPS 2020*.
   - Classical periodic activation functions (Snake); ReLU failure on periodic data

6. **Sitzmann, V., Martel, J.N.P., Bergman, A.W., Lindell, D.B., & Wetzstein, G.** (2020). Implicit Neural Representations with Periodic Activation Functions. *NeurIPS 2020*.
   - SIREN: classical Fourier baseline

7. **Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J.I.** (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226.
   - Data re-uploading framework underlying frequency-matched encoding

8. **Casas, B. & Cervera-Lierta, A.** (2023). Multidimensional Fourier series with quantum circuits. *Physical Review A*, 107, 062612.
   - Extension of Schuld et al. to multidimensional inputs

### Papers to Cite (Time-Series Benchmarks)

9. **Zhou, H. et al.** (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI 2021*.
   - ETTh1 dataset

10. **Wu, H. et al.** (2021). Autoformer: Decomposition Transformers with Auto-Correlation. *NeurIPS 2021*.
    - Weather and ECL datasets

11. **Nie, Y. et al.** (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*.
    - PatchTST; benchmark methodology

---

*This analysis is based on all 13 documentation files in `/VQC-PeriodicData/docs/` and a literature search conducted February 13, 2026.*
