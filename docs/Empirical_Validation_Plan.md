# Empirical Validation Plan: Testing Theoretical Claims for NeurIPS/ICML Submission

**Project**: VQC-PeriodicData
**Date**: February 2026
**Target Venues**: NeurIPS 2026, ICML 2026
**Paper Framing**: Methods/architecture paper with principled design from theory + rigorous empirical validation

---

## Table of Contents

1. [Objectives and Hypotheses](#1-objectives-and-hypotheses)
2. [Experimental Infrastructure to Build](#2-experimental-infrastructure-to-build)
3. [Phase 0: Spectral Concentration Profiling](#phase-0-spectral-concentration-profiling-theorem-4)
4. [Phase 1: Ablation Study](#phase-1-ablation-study-theorems-1-2-3)
5. [Phase 2: Extrapolation Benchmark](#phase-2-extrapolation-benchmark-core-vqc-advantage)
6. [Phase 3: Parameter Efficiency and Pareto Analysis](#phase-3-parameter-efficiency-and-pareto-analysis)
7. [Phase 4: Initialization Strategy Comparison](#phase-4-initialization-strategy-comparison-theorem-2)
8. [Phase 5: Qubit Scaling Experiment](#phase-5-qubit-scaling-experiment-theorem-1)
9. [Phase 6: Full Multivariate Benchmarks](#phase-6-full-multivariate-benchmarks)
10. [Phase 7: Statistical Significance Runs](#phase-7-statistical-significance-runs)
11. [Phase 8: Hyperparameter Sensitivity for Baselines](#phase-8-hyperparameter-sensitivity-for-baselines)
12. [Contingency Plans](#contingency-plans)
13. [Computational Budget](#computational-budget)
14. [Paper Figure and Table Map](#paper-figure-and-table-map)
15. [Timeline](#timeline)

---

## 1. Objectives and Hypotheses

Each experiment tests a specific theoretical claim. The paper's narrative follows the structure: *theory predicts X → we test X → results confirm/refine the prediction*.

### Theorem-to-Experiment Mapping

| Theorem | Claim | Testable Prediction | Experiment |
|---------|-------|---------------------|------------|
| **Thm 1** (Freq Expansion) | RY+RX encoding gives 9^n vs 3^n frequencies | More qubits → exponentially better fit on multi-frequency targets | Phase 5 (qubit scaling) |
| **Thm 2** (FFT Init) | FFT-seeded freq_scale is provably optimal starting point | FFT init converges faster and reaches lower loss than random/linspace | Phase 4 (init comparison) |
| **Thm 3** (Gate Preservation) | Rescaled gating has zero spectral distortion; sigmoid distorts | Replacing rescaled with sigmoid degrades periodic tasks most | Phase 1 (ablation) |
| **Thm 4** (Spectral Concentration) | Advantage scales with ρ(n); any spectrally concentrated signal benefits | ρ(n) of a dataset predicts VQC relative advantage | Phase 0 (profiling) + cross-phase correlation |

### Core Hypotheses (must all be supported for a strong paper)

**H1**: FourierQLSTM outperforms parameter-matched classical models on spectrally concentrated univariate data (Multi-Sine, Mackey-Glass) and FourierQTCN outperforms on spectrally concentrated multivariate data (ETTh1, PhysioNet EEG), but neither shows advantage on broadband data (Adding Problem, White Noise).

**H2**: VQC models maintain prediction quality under temporal extrapolation (test beyond training domain), while classical models degrade significantly.

**H3**: Each design principle (FFT preprocessing, frequency-matched encoding, rescaled gating, FFT-seeded init) independently contributes to performance; removing any one degrades results.

**H4**: The spectral concentration ratio ρ(n) of a dataset, computable a priori from training data, predicts the magnitude of VQC advantage.

**H5**: FourierQLSTM achieves Pareto-superior parameter efficiency: at any given parameter budget, it matches or exceeds classical models on spectrally concentrated tasks. *(Note: This hypothesis is strongest for LSTM models, where the gap is ~100×. For TCN models, the classical projection layer reduces the gap to ~3×; see Phase 3 discussion.)*

---

## 2. Experimental Infrastructure to Build

Before running experiments, several pieces of infrastructure must be implemented.

### 2.1 Extrapolation Data Generator

**File to create**: `data/extrapolation_generator.py`

```python
def get_extrapolation_splits(dataset_name, train_range, test_ranges, **kwargs):
    """Generate data for extrapolation experiments.

    Args:
        dataset_name: 'multisine', 'mackey_glass', 'etth1'
        train_range: (t_start, t_end) for training
        test_ranges: list of (t_start, t_end) for extrapolation testing

    Returns:
        train_loader, list_of_test_loaders (one per test_range)
    """
```

**Design**:
- Multi-Sine: Generate 2000 time steps. Train on [0, 1000], test on [1000, 1200] (near), [1200, 1500] (mid), [1500, 2000] (far).
- Mackey-Glass: Generate 2000 steps (after warmup). Same split scheme.
- ETTh1: Train on months 1-12. Test on months 13-16 (near), 17-20 (standard), but also generate synthetic continuation for months 21-24 using the periodic components.

### 2.2 Parameter-Matched Classical Models

**File to create**: `models/param_matched_configs.py`

```python
# Configuration file for parameter-matched experiments
PARAM_BUDGETS = {
    'tiny':   200,    # Match FourierQLSTM default
    'small':  2000,
    'medium': 20000,  # Match current classical defaults
}

def get_param_matched_config(model_type, target_params):
    """Return (mlp_hidden_dim, n_layers) to match target param count."""
```

For ~200 params with the current LSTM architecture (FFT features + freq-domain cell state), the classical MLP hidden dim needs to be ~4-8 (very small).

### 2.3 Spectral Concentration Analyzer

**File to create**: `data/spectral_analysis.py`

```python
def compute_spectral_concentration(signal, n_max=20):
    """Compute ρ(n) for n=1,...,n_max.

    Args:
        signal: 1D numpy array (time series)
        n_max: maximum number of top frequencies to consider

    Returns:
        rho: array of shape (n_max,) with ρ(1), ρ(2), ..., ρ(n_max)
        top_freqs: array of the top-n_max frequency indices
        top_powers: corresponding power values
    """
```

### 2.4 Ablation Variants of FourierQLSTM

**Modify**: `models/FourierQLSTM.py` to accept ablation flags:

```bash
--ablate-fft          # Skip FFT, feed raw time-domain input
--ablate-freq-match   # Use standard RY-only encoding (no RX, no freq_scale)
--ablate-rescaled     # Use sigmoid gating instead of (VQC+1)/2
--ablate-fft-init     # Use random initialization instead of FFT-seeded
```

Each flag removes exactly one design principle while keeping all others intact.

### 2.5 Experiment Runner Script

**File to create**: `run_experiments.py`

A unified experiment runner that:
- Accepts a phase name (e.g., `--phase=ablation`)
- Runs all models for that phase with all seeds
- Saves results in a structured directory: `results/<phase>/<dataset>/<model>_seed<N>.csv`
- Supports `--dry-run` to print commands without executing
- Supports `--slurm` to generate SLURM batch scripts

---

## Phase 0: Spectral Concentration Profiling (Theorem 4)

### Purpose

Compute the spectral concentration ratio ρ(n) for every dataset used in the paper. This serves two purposes:
1. **Motivates the paper**: shows which datasets have exploitable spectral structure
2. **Predicts outcomes**: Theorem 4 says VQC advantage scales with ρ(n)

### Protocol

For each dataset, extract a representative time series and compute:
- ρ(n) for n = 1, 2, ..., 20 (fraction of energy in top-n frequencies)
- The top-20 frequency indices and their power contributions
- Spectral decay class (estimate β from power-law fit to sorted spectrum)

### Datasets

| Dataset | Signal for Analysis | Expected ρ(6) | Spectral Class |
|---------|-------------------|---------------|----------------|
| Multi-Sine (K=5) | Raw generated signal | ~0.99 (5 exact harmonics) | Finite harmonic |
| Mackey-Glass (τ=17) | Raw generated signal | ~0.85-0.95 (dominant oscillation) | Quasi-periodic |
| NARMA-10 | Raw generated signal | ~0.3-0.5 (broadband) | Broadband/stochastic |
| Adding Problem | Raw signal component | ~0.05 (flat spectrum) | White noise |
| ETTh1 (OT column) | Oil temperature | ~0.7-0.9 (diurnal + weekly) | Smooth periodic |
| Weather (temperature) | Air temperature | ~0.6-0.8 (diurnal + seasonal) | Multi-scale periodic |
| ECL (MT_001) | Client electricity | ~0.8-0.95 (strong diurnal) | Smooth periodic |
| PhysioNet EEG | Motor imagery (C3/C4 channels) | ~0.7-0.9 (alpha/beta bands) | Multi-band periodic |

### Commands

```bash
cd /pscratch/sd/j/junghoon/VQC-PeriodicData

# After implementing spectral_analysis.py:
python data/spectral_analysis.py --dataset=all --n-max=20 --output=results/spectral_profiles/
```

### Output

- **Figure 1 (paper)**: Spectral concentration curves ρ(n) vs n for all 8 datasets, with horizontal dashed line at ρ = 0.9 (the "high concentration" threshold from Theorem 4)
- **Table (appendix)**: Top-6 frequencies and powers for each dataset
- **Prediction table**: Expected VQC advantage ranking based on ρ(6)

### Success Criteria

- Multi-Sine and ECL should have ρ(6) > 0.9
- PhysioNet EEG should have ρ(6) in the 0.7-0.9 range (alpha/beta rhythms dominate motor imagery)
- NARMA-10 and Adding should have ρ(6) < 0.5
- The ranking should correlate with VQC relative performance in later experiments

### Estimated Compute

Negligible (~5 minutes total). This is pure FFT computation, no model training.

---

## Phase 1: Ablation Study (Theorems 1, 2, 3)

### Purpose

Demonstrate that each of the three design principles independently contributes to FourierQLSTM's performance. This is **the most important experiment for reviewers** — it proves the architecture choices are principled, not arbitrary.

### Protocol

Run FourierQLSTM with each design principle ablated individually, keeping all others intact.

| Variant | FFT Preprocessing | Freq-Matched Encoding (RY+RX) | Rescaled Gating | FFT-Seeded Init | Tests Theorem |
|---------|:-:|:-:|:-:|:-:|:-:|
| **Full FourierQLSTM** | Yes | Yes | Yes | Yes | (baseline) |
| **A: No FFT** | **No** (raw time-domain) | Yes | Yes | Linspace (can't FFT) | — |
| **B: No Freq-Match** | Yes | **No** (RY-only, freq_scale=0) | Yes | Yes | Thm 1 |
| **C: No Rescaled Gate** | Yes | Yes | **No** (sigmoid) | Yes | Thm 3 |
| **D: No FFT Init** | Yes | Yes | Yes | **No** (random uniform) | Thm 2 |

### Datasets

Run on two datasets that span the spectrum:
- **Multi-Sine (K=5)**: High spectral concentration, purest Fourier test
- **NARMA-10**: Moderate/low spectral concentration, broadband control

### Configuration

```bash
# Shared params
COMMON="--seed=2025 --n-qubits=6 --vqc-depth=2 --hidden-size=4 --window-size=8 --n-epochs=50 --batch-size=10 --lr=0.01"

# Full model (baseline)
python models/FourierQLSTM.py --dataset=multisine ${COMMON} --freq-init=fft
python models/FourierQLSTM.py --dataset=narma    ${COMMON} --freq-init=fft

# A: No FFT preprocessing
python models/FourierQLSTM.py --dataset=multisine ${COMMON} --ablate-fft
python models/FourierQLSTM.py --dataset=narma    ${COMMON} --ablate-fft

# B: No frequency-matched encoding (RY-only)
python models/FourierQLSTM.py --dataset=multisine ${COMMON} --ablate-freq-match
python models/FourierQLSTM.py --dataset=narma    ${COMMON} --ablate-freq-match

# C: No rescaled gating (use sigmoid)
python models/FourierQLSTM.py --dataset=multisine ${COMMON} --ablate-rescaled
python models/FourierQLSTM.py --dataset=narma    ${COMMON} --ablate-rescaled

# D: No FFT-seeded init (random uniform init)
python models/FourierQLSTM.py --dataset=multisine ${COMMON} --freq-init=random
python models/FourierQLSTM.py --dataset=narma    ${COMMON} --freq-init=random
```

### Expected Results

| Variant | Multi-Sine MSE (predicted) | NARMA-10 MSE (predicted) | Explanation |
|---------|:-:|:-:|---|
| Full | Best | Best | All innovations active |
| A: No FFT | Significantly worse | Moderately worse | Raw time-domain loses frequency info |
| B: No Freq-Match | Moderately worse | Slightly worse | 3^n vs 9^n frequencies (Thm 1) |
| C: No Rescaled | Moderately worse on Multi-Sine, less on NARMA | Slightly worse | Spectral distortion matters more for periodic (Thm 3) |
| D: No FFT Init | Same final MSE, slower convergence | Same final, slower | Init affects convergence, not capacity (Thm 2) |

### FourierQTCN Ablation on PhysioNet EEG (Classification)

In addition to the LSTM ablation on synthetic data, run the same ablation design on FourierQTCN with PhysioNet EEG — the project's original motivation and the only classification task.

| Variant | FFT Preprocessing | Freq-Matched Encoding (RY+RX) | Rescaled Gating | FFT-Seeded Init | Tests Theorem |
|---------|:-:|:-:|:-:|:-:|:-:|
| **Full FourierQTCN** | Yes | Yes | Yes | Yes | (baseline) |
| **A: No FFT** | **No** (raw time-domain) | Yes | Yes | Linspace | — |
| **B: No Freq-Match** | Yes | **No** (RY-only, freq_scale=0) | Yes | Yes | Thm 1 |
| **C: No Rescaled Gate** | Yes | Yes | **No** (sigmoid) | Yes | Thm 3 |
| **D: No FFT Init** | Yes | Yes | Yes | **No** (random uniform) | Thm 2 |

```bash
# PhysioNet EEG ablation (FourierQTCN)
COMMON_EEG="--seed=2025 --n-qubits=8 --vqc-depth=2 --kernel-size=12 --dilation=3 --num-epochs=50"

python models/FourierQTCN_EEG.py --dataset=physionet_eeg ${COMMON_EEG} --freq-init=fft
python models/FourierQTCN_EEG.py --dataset=physionet_eeg ${COMMON_EEG} --ablate-fft
python models/FourierQTCN_EEG.py --dataset=physionet_eeg ${COMMON_EEG} --ablate-freq-match
python models/FourierQTCN_EEG.py --dataset=physionet_eeg ${COMMON_EEG} --ablate-rescaled
python models/FourierQTCN_EEG.py --dataset=physionet_eeg ${COMMON_EEG} --freq-init=random
```

**Metric**: ROC-AUC (binary classification: left vs right motor imagery).

**Expected**: The EEG ablation should mirror the Multi-Sine trends — ablating FFT preprocessing (variant A) and rescaled gating (variant C) should hurt most, since EEG motor imagery has strong alpha/beta-band periodicity.

### Key Insight to Test

**Theorem 3 prediction**: The ablation of rescaled gating (variant C) should hurt Multi-Sine (high ρ) much more than NARMA-10 (low ρ), because spectral distortion from sigmoid destroys the periodic structure that only matters when the data IS periodic. The PhysioNet EEG ablation (ρ ≈ 0.7-0.9) should fall between these extremes, providing a third data point for the ρ-vs-ablation-impact correlation.

### Output

- **Table 2 (paper)**: Ablation results matrix (5 variants × 3 datasets × test MSE/AUC)
- **Figure (appendix)**: Training curves for all variants overlaid

### Estimated Compute

LSTM ablation: 10 runs × ~40s/epoch × 50 epochs = ~5.5 hours on CPU.
TCN EEG ablation: 5 runs × ~90s/epoch × 50 epochs = ~6.3 hours on CPU (or ~0.6 hours on GPU).

---

## Phase 2: Extrapolation Benchmark (Core VQC Advantage)

### Purpose

This is **the single most important experiment in the paper**. The theoretical advantage of VQC over classical models is periodic extrapolation — VQC outputs a Fourier series (periodic outside training domain) while ReLU extrapolates linearly and tanh extrapolates to a constant. No existing quantum time-series paper tests this.

### Protocol

**Training**: Fit all models on a temporal window [0, T].
**Testing**: Evaluate on multiple windows beyond T.

```
Training region        Near extrap.     Mid extrap.      Far extrap.
[────────────]         [────]           [────]           [────]
0              T       T    1.2T        1.5T   1.8T     2T     2.5T
```

**Metric**: For each test window, compute MSE. Report the *extrapolation degradation ratio*:

$$\text{EDR}(d) = \frac{\text{MSE}_{\text{extrap at distance } d}}{\text{MSE}_{\text{interpolation}}}$$

Lower EDR means the model degrades less when extrapolating. VQC should have EDR ≈ 1 (periodic functions extrapolate perfectly). ReLU should have EDR >> 1.

### Dataset-Specific Designs

#### Multi-Sine (K=5) — Cleanest Test

```
Generate 2000 time steps (t = 0, 1, ..., 1999)
Train:          t ∈ [0, 999]      (1000 steps)
Val:            t ∈ [1000, 1199]  (200 steps, near extrapolation)
Test-Near:      t ∈ [1200, 1399]  (200 steps)
Test-Mid:       t ∈ [1400, 1599]  (200 steps)
Test-Far:       t ∈ [1600, 1999]  (400 steps)
```

Since Multi-Sine is periodic, a perfect periodic model should have EDR ≈ 1 at all distances.

#### Mackey-Glass (τ=17) — Quasi-Periodic Test

```
Generate 2000 steps (after 500-step warmup)
Train:          t ∈ [0, 999]
Test-Near:      t ∈ [1000, 1199]
Test-Mid:       t ∈ [1200, 1499]
Test-Far:       t ∈ [1500, 1999]
```

Mackey-Glass at τ=17 is quasi-periodic. VQC should extrapolate better than ReLU/Tanh but not perfectly (because the signal isn't exactly periodic).

#### ETTh1 — Real-World Periodic Test

```
Total: 17,420 hourly steps (~2 years)
Train:          steps [0, 10000]      (~14 months)
Val:            steps [10000, 12000]  (~3 months)
Test-Near:      steps [12000, 14000]  (~3 months)
Test-Far:       steps [14000, 17420]  (~5 months)
```

ETTh1 has strong diurnal (24h) periodicity. The question: can VQC exploit this for temporal extrapolation?

### Models

- **Univariate datasets (Multi-Sine, Mackey-Glass)**: All 5 LSTM variants — FourierQLSTM, ReLU-LSTM, Tanh-LSTM, Snake-LSTM, SIREN-LSTM
- **Multivariate dataset (ETTh1)**: All 6 TCN variants — FourierQTCN, HQTCN2, ReLU-TCN, Tanh-TCN, Snake-TCN, SIREN-TCN

(FourierQLSTM is a univariate model and cannot process multivariate inputs like ETTh1.)

### Commands

```bash
# Multi-Sine extrapolation (LSTM — univariate)
python models/FourierQLSTM.py --dataset=multisine --extrap-mode --n-samples=2000 \
    --train-end=1000 --seed=2025 --n-epochs=100

python models/ReLU_LSTM.py --dataset=multisine --extrap-mode --n-samples=2000 \
    --train-end=1000 --seed=2025 --n-epochs=100

# ... same for Tanh, Snake, SIREN LSTMs on Multi-Sine and Mackey-Glass

# ETTh1 extrapolation (TCN — multivariate)
for model in FourierQTCN_EEG HQTCN2_EEG ReLU_TCN_EEG Tanh_TCN_EEG Snake_TCN_EEG SIREN_TCN_EEG; do
    python models/${model}.py --dataset=etth1 --extrap-mode --seed=2025 --num-epochs=100
done
```

(The `--extrap-mode` flag needs to be implemented — it changes the data split from random to temporal.)

### Expected Results

**Univariate (LSTM models)**:

| Model | Multi-Sine EDR | Mackey-Glass EDR |
|-------|:-:|:-:|
| FourierQLSTM | **~1.0-1.5** | **~2-5** |
| SIREN-LSTM | ~1.5-3 | ~5-10 |
| Snake-LSTM | ~3-8 | ~5-15 |
| ReLU-LSTM | ~10-100 | ~10-50 |
| Tanh-LSTM | ~10-100 | ~10-50 |

**Multivariate (TCN models)**:

| Model | ETTh1 EDR |
|-------|:-:|
| FourierQTCN | **~2-4** |
| SIREN-TCN | ~3-6 |
| Snake-TCN | ~5-10 |
| ReLU-TCN | ~5-20 |
| Tanh-TCN | ~5-20 |
| HQTCN2 (original) | ~4-15 |

If FourierQLSTM has EDR ≈ 1 on Multi-Sine while ReLU has EDR >> 10, this is the paper's strongest result.

### Output

- **Figure 3 (paper)**: Extrapolation degradation ratio vs distance from training domain, all models overlaid. One subplot per dataset.
- **Table (appendix)**: Raw MSE values at each distance.

### Estimated Compute

5 models × 3 datasets × ~50-100 epochs × ~40s/epoch (quantum) = ~28-55 hours for quantum models. Classical models: ~1 hour total.

Priority: Run Multi-Sine first (cleanest signal), then ETTh1 (real-world relevance), then Mackey-Glass.

---

## Phase 3: Parameter Efficiency and Pareto Analysis

### Purpose

Address the reviewer objection: "ReLU-LSTM beats FourierQLSTM on absolute MSE." Response: "ReLU-LSTM uses 100× more parameters. At equal parameter budget, FourierQLSTM wins."

### Important: LSTM vs TCN Parameter Gap

The parameter efficiency story differs fundamentally between the two architectures:

| Architecture | Quantum Model Params | Classical Model Params | Gap | Why |
|-------------|:-:|:-:|:-:|---|
| **LSTM** (4 gates) | ~199 | ~19,600 | **~100×** | Each classical gate is a full MLP (hidden=64); quantum gates are VQCs with ~12 params each |
| **TCN** (1 block) | ~1,000-1,200 | ~3,500 | **~3×** | FourierQTCN's `nn.Linear(fft_features, n_qubits)` projection layer dominates, adding ~800 classical params |

**Why the TCN gap is small**: FourierQTCN requires a classical linear projection from FFT feature space (e.g., 7 channels × 7 freq bins × 2 = 98 features for ETTh1) down to n_qubits. This projection layer alone has ~800 parameters — more than the quantum circuit itself (~264 params for conv + pool). The quantum part IS parameter-efficient, but the classical "adapter" layer dilutes the advantage.

**Implication for H5**: The Pareto superiority claim is **strongest for LSTM models** and should be presented primarily in that context. For TCN models, the story is different: the advantage (if any) comes from the quantum circuit's expressivity at a given width, not from raw parameter count.

This distinction is scientifically important and should be discussed transparently in the paper (it connects to Yu et al. (NeurIPS 2024)'s caveat about hybrid models).

### Protocol

#### Part A: LSTM Pareto Analysis (Primary — strong 100× gap)

Run each LSTM model family at three parameter budgets:

| Budget | FourierQLSTM Config | Classical LSTM Config |
|--------|--------------------|-----------------------|
| **Tiny (~200 params)** | n_qubits=6, depth=2 (current default) | mlp_hidden_dim=4-8 |
| **Small (~2000 params)** | n_qubits=8, depth=3 | mlp_hidden_dim=32 |
| **Medium (~20K params)** | n_qubits=12, depth=4 | mlp_hidden_dim=64 (current default) |

**Note**: FourierQLSTM at n_qubits=12 will be very slow (~4× slower than n_qubits=6 for circuit simulation). Consider using PennyLane's `lightning.qubit` backend or `default.qubit` with adjoint differentiation for speed.

#### Part B: TCN Pareto Analysis (Secondary — weaker ~3× gap)

For TCN models, the parameter-matching experiment is less dramatic but still informative. Test at two budgets:

| Budget | FourierQTCN Config | Classical TCN Config |
|--------|-------------------|-----------------------|
| **Matched (~1,200 params)** | n_qubits=8, depth=2 (default) | mlp_hidden_dim=16 (reduced from 64) |
| **Default (~3,500 params)** | n_qubits=8, depth=3 | mlp_hidden_dim=64 (current default) |

**Datasets**: ETTh1 (regression) and PhysioNet EEG (classification). Running on both a regression and classification task demonstrates the TCN Pareto story is not task-specific.

The question for TCN: *Does the quantum circuit provide better expressivity per parameter than the classical MLP at matched total model size?* This is a subtler claim than the LSTM case.

#### Part C: Decomposed Analysis (Distinguish quantum vs classical contribution)

To address Yu et al.'s hybrid model caveat, report parameter counts decomposed:

```
Model Total Params = Classical Params (projection, output) + Quantum Params (circuit)
```

For FourierQLSTM:  ~199 total ≈ ~80 classical + ~119 quantum
For FourierQTCN:   ~1,200 total ≈ ~900 classical + ~300 quantum

This transparency strengthens the paper: we acknowledge the classical overhead and argue that the quantum circuit contributes disproportionate expressivity relative to its parameter share.

### Datasets

- Multi-Sine (K=5): Pure periodic
- NARMA-10: Broadband control
- Mackey-Glass: Quasi-periodic

### Expected Results (Multi-Sine, LSTM)

```
Test MSE
  ↑
  │  ×  ×  ×  ×  ×               ← Classical LSTM (tiny ~200 params): high MSE
  │
  │     ★                         ← FourierQLSTM (tiny ~200 params): low MSE
  │
  │           × × ×              ← Classical LSTM (small ~2K params): moderate MSE
  │              ★               ← FourierQLSTM (small ~2K params): lower MSE
  │
  │                    ×         ← Classical LSTM (medium ~20K params): lowest classical MSE
  │                    ★         ← FourierQLSTM (medium ~20K): comparable
  └──────────────────────→ Parameter Count
        200      2K      20K
```

The prediction: FourierQLSTM's Pareto frontier should lie below (better than) all classical LSTM frontiers on spectrally concentrated data.

For TCN models, the gap at matched params will be smaller. If FourierQTCN still outperforms at matched ~1,200 params, this supports the claim that the quantum circuit itself (not just the classical adapter) provides value.

### Output

- **Figure 4 (paper)**: Pareto frontier plot. X-axis: log(params), Y-axis: test MSE. One curve per model family, one subplot per dataset. **LSTM and TCN on separate subplots** to avoid misleading cross-architecture comparisons.
- **Table (appendix)**: Decomposed parameter counts (classical vs quantum) for all models at all budgets.

### Estimated Compute

Part A: 3 budgets × 5 LSTM models × 3 datasets × 50 epochs = 45 runs × ~40-160s/epoch = ~25-100 hours.
Part B: 2 budgets × 6 TCN models × 2 datasets (ETTh1, PhysioNet EEG) × 50 epochs = ~24 hours classical + ~300 hours quantum.

Priority: Run Part A (LSTM) tiny budget first (fastest, most impactful). Run Part B only after Part A confirms the trend.

---

## Phase 4: Initialization Strategy Comparison (Theorem 2)

### Purpose

Directly test Theorem 2: FFT-seeded initialization should converge faster and reach lower loss than alternatives, especially on high-ρ(n) datasets.

### Protocol

Three initialization strategies for `freq_scale`:

| Strategy | Method | Theorem 2 Prediction |
|----------|--------|---------------------|
| **FFT-seeded** | Top-n frequencies from training data FFT | Fastest convergence, lowest loss |
| **Linspace** | Evenly spaced in [0.5, 3.0] | Moderate convergence |
| **Random** | Uniform random in [0.5, 3.0] | Slowest convergence |

### Datasets

All 4 LSTM benchmarks (Multi-Sine, Mackey-Glass, NARMA-10, Adding).

### Configuration

```bash
COMMON="--seed=2025 --n-qubits=6 --vqc-depth=2 --hidden-size=4 --window-size=8 --n-epochs=100 --batch-size=10 --lr=0.01"

# FFT-seeded
python models/FourierQLSTM.py --dataset=multisine ${COMMON} --freq-init=fft

# Linspace
python models/FourierQLSTM.py --dataset=multisine ${COMMON} --freq-init=linspace

# Random
python models/FourierQLSTM.py --dataset=multisine ${COMMON} --freq-init=random
```

Run for **100 epochs** (not 50) to clearly see convergence differences.

### Key Metrics

1. **Epochs to convergence** (defined as epoch where test MSE first drops below 110% of final MSE)
2. **Final test MSE** at epoch 100
3. **freq_scale trajectory**: Plot how freq_scale values evolve over training for each init strategy

### Expected Results

| Dataset | Init | Epochs to Converge | Final Test MSE |
|---------|------|:-:|:-:|
| Multi-Sine (ρ≈0.99) | FFT | ~10-20 | Lowest |
| Multi-Sine | Linspace | ~30-50 | Similar final |
| Multi-Sine | Random | ~40-70 | Similar final or slightly worse |
| NARMA-10 (ρ≈0.4) | FFT | ~10-20 | Slightly better |
| NARMA-10 | Linspace | ~15-25 | Similar |
| NARMA-10 | Random | ~20-30 | Similar |

**Key prediction**: The convergence speed gap between FFT and random should be **larger on high-ρ datasets** (Multi-Sine) than on low-ρ datasets (NARMA-10). This is the empirical signature of Theorem 2.

### Output

- **Table 3 (paper)**: Init strategy × dataset matrix showing (epochs to converge, final MSE)
- **Figure (appendix)**: Training curves (test MSE vs epoch) for each init strategy, overlaid per dataset. Plus freq_scale evolution plots.

### Estimated Compute

3 strategies × 4 datasets × 100 epochs × ~40s/epoch = ~13 hours.

---

## Phase 5: Qubit Scaling Experiment (Theorem 1)

### Purpose

Test Theorem 1's prediction that frequency-matched encoding gives 9^n vs 3^n accessible frequencies. More qubits should exponentially improve expressivity on multi-frequency targets.

### Protocol

Run FourierQLSTM with varying qubit counts, alongside two controls:

1. **FourierQLSTM** (RY + freq_scale × RX): 9^n frequencies (Theorem 1)
2. **Standard VQC-LSTM** (RY only, otherwise same architecture): 3^n frequencies
3. **ReLU-LSTM** at matched parameter count

### Qubit Counts

n ∈ {2, 4, 6, 8, 10}

| n_qubits | FourierQLSTM Freqs (9^n) | Standard VQC Freqs (3^n) | Approx Params (FourierQLSTM) |
|----------|:-:|:-:|:-:|
| 2 | 81 | 9 | ~65 |
| 4 | 6,561 | 81 | ~115 |
| 6 | 531,441 | 729 | ~199 |
| 8 | 43M | 6,561 | ~315 |
| 10 | 3.5B | 59,049 | ~465 |

### Dataset

**Multi-Sine (K=5)** — requires at least 5 distinct frequencies to fit exactly. With n=2 qubits, the standard VQC has only 9 total frequencies (may not include the needed 5). With n=6, it has 729 (more than enough).

Also run on **Multi-Sine (K=10)** and **Multi-Sine (K=20)** to test scaling with target complexity.

### Expected Results

Test MSE should decrease with n, with FourierQLSTM decreasing faster than Standard VQC:

```
log(Test MSE)
  ↑
  │ ×                              Standard VQC (3^n)
  │   ×
  │     ×     ×       ×
  │ ★                              FourierQLSTM (9^n)
  │   ★
  │       ★
  │           ★   ★               ← converges faster
  └──────────────────→ n_qubits
    2    4    6    8    10
```

### Output

- **Figure (paper or appendix)**: Test MSE vs n_qubits for FourierQLSTM, Standard VQC, and parameter-matched ReLU-LSTM. One subplot per K value (K=5, 10, 20).

### Estimated Compute

5 qubit counts × 2 models × 3 K values × 50 epochs × ~40-200s/epoch = ~28-140 hours.

Priority: Run K=5 first. Only run K=10 and K=20 if K=5 shows clear trends.

---

## Phase 6: Full Multivariate Benchmarks

### Purpose

Demonstrate that the approach works on standard time-series benchmarks used by NeurIPS/ICML reviewers. Currently only 1-epoch smoke tests exist for ETTh1.

### Datasets

| Dataset | Features | Task | Metric | Expected ρ(6) | VQC Advantage |
|---------|----------|------|--------|:-:|:-:|
| ETTh1 | 7 | Next-step OT prediction | RMSE | High | Moderate-Strong |
| Weather | 21 | Next-step temp prediction | RMSE | High | Moderate |
| ECL (20 channels) | 20 | Next-step load prediction | RMSE | Very High | Strong |
| PhysioNet EEG | 64 (C3/C4 selected) | Motor imagery classification | ROC-AUC | High | Strong |

PhysioNet EEG is the only **classification** task and the project's original motivation. It tests whether FourierQTCN's design principles transfer from regression to classification on intrinsically periodic biomedical signals (alpha/beta rhythms).

### Models

All 6 TCN variants: FourierQTCN, HQTCN2 (original), ReLU-TCN, Tanh-TCN, Snake-TCN, SIREN-TCN.

### Configuration

```bash
COMMON="--num-epochs=50 --seed=2025 --kernel-size=12 --dilation=3"

# ETTh1 (auto-downloads)
for model in FourierQTCN_EEG HQTCN2_EEG ReLU_TCN_EEG Tanh_TCN_EEG Snake_TCN_EEG SIREN_TCN_EEG; do
    python models/${model}.py --dataset=etth1 ${COMMON}
done

# Weather (manual download required first)
for model in FourierQTCN_EEG HQTCN2_EEG ReLU_TCN_EEG Tanh_TCN_EEG Snake_TCN_EEG SIREN_TCN_EEG; do
    python models/${model}.py --dataset=weather ${COMMON}
done

# ECL
for model in FourierQTCN_EEG HQTCN2_EEG ReLU_TCN_EEG Tanh_TCN_EEG Snake_TCN_EEG SIREN_TCN_EEG; do
    python models/${model}.py --dataset=ecl --n-channels=20 ${COMMON}
done

# PhysioNet EEG (classification — uses ROC-AUC instead of RMSE)
for model in FourierQTCN_EEG HQTCN2_EEG ReLU_TCN_EEG Tanh_TCN_EEG Snake_TCN_EEG SIREN_TCN_EEG; do
    python models/${model}.py --dataset=physionet_eeg --task=classification ${COMMON}
done
```

### Expected Results

| Model | ETTh1 RMSE | Weather RMSE | ECL RMSE | PhysioNet EEG AUC |
|-------|:-:|:-:|:-:|:-:|
| FourierQTCN | **Best** | **Best** | **Best** | **Best** |
| SIREN-TCN | Second | Second | Second | Second |
| Snake-TCN | Third | Third | Third | Third |
| ReLU-TCN | Baseline | Baseline | Baseline | Baseline |
| Tanh-TCN | ~Baseline | ~Baseline | ~Baseline | ~Baseline |
| HQTCN2 (original) | Worse than FourierQTCN | Worse | Worse | Worse |

The comparison between FourierQTCN and HQTCN2 (original quantum model without frequency-aware design) directly demonstrates the value of the proposed design principles. PhysioNet EEG additionally tests whether the advantage transfers to classification on real biomedical data.

### Output

- **Table 4 (paper)**: 6 models × 4 datasets matrix of test RMSE/AUC (mean ± std over 3 seeds)

### Estimated Compute

6 models × 4 datasets × 50 epochs. Classical: ~1 min/epoch × 50 = ~20 hours total. Quantum: ~90 min/epoch × 50 = ~150 hours per quantum model × 2 quantum models × 4 datasets = **~1,200 hours** on CPU.

**Critical**: Must use GPU for quantum models. With GPU (`lightning.gpu`), expect ~10-20× speedup → ~60-120 hours.

**Alternative**: Run ETTh1 (most cited benchmark) + PhysioNet EEG (classification, project motivation) for all models, and ECL (strongest periodicity) for quantum vs ReLU comparison only. This reduces to ~400 hours CPU / ~40 hours GPU.

---

## Phase 7: Statistical Significance Runs

### Purpose

All key results must have error bars. NeurIPS/ICML require reporting mean ± standard deviation over multiple random seeds.

### Seeds

Use 3 seeds: {2025, 2026, 2027}

### Which Experiments Need Multi-Seed

| Experiment | Seeds | Priority |
|-----------|:-----:|:-:|
| Phase 1 (Ablation) | 3 | High |
| Phase 2 (Extrapolation) on Multi-Sine | 3 | High |
| Phase 2 (Extrapolation) on ETTh1 | 3 | Medium |
| Phase 3 (Pareto) at tiny budget | 3 | High |
| Phase 4 (Init comparison) | 3 | Medium |
| Phase 6 (Multivariate) on ETTh1 | 3 | High |
| Phase 6 (Multivariate) on PhysioNet EEG | 3 | High |

### Statistical Tests

For pairwise comparisons (e.g., "FourierQLSTM vs ReLU-LSTM"):
- **Paired t-test** if results appear normally distributed (n=3 seeds)
- Report p-values; claim significance at p < 0.05
- With only 3 seeds, focus on reporting mean ± std and consistent directional trends rather than formal hypothesis tests

### Estimated Compute

Multiply Phase 1-6 budgets by seed count. This is the largest cost:
- Phase 1 × 3 seeds: ~17 hours
- Phase 2 × 3 seeds (Multi-Sine only): ~17 hours
- Phase 3 × 3 seeds (tiny budget only): ~8 hours
- Phase 4 × 3 seeds: ~39 hours
- Phase 6 × 3 seeds (ETTh1 + EEG, quantum only): ~540 hours CPU / ~54 hours GPU

---

## Phase 8: Hyperparameter Sensitivity for Baselines

### Purpose

Address the reviewer objection that SIREN and Snake underperform due to bad hyperparameters, not inherent limitations. The existing results use SIREN w0=30 (optimized for images) and Snake a_init=1.0 (default). Both may be suboptimal for time-series.

### Protocol

Sweep key hyperparameters for the two periodic classical baselines:

**SIREN-LSTM**: Sweep w0 ∈ {1.0, 5.0, 10.0, 30.0}
**Snake-LSTM**: Sweep a_init ∈ {0.1, 0.5, 1.0, 5.0}

Run on Multi-Sine and NARMA-10 with seed=2025, 50 epochs.

### Commands

```bash
# SIREN sweep
for w0 in 1.0 5.0 10.0 30.0; do
    python models/SIREN_LSTM.py --dataset=multisine --w0=${w0} --seed=2025 --n-epochs=50
    python models/SIREN_LSTM.py --dataset=narma --w0=${w0} --seed=2025 --n-epochs=50
done

# Snake sweep
for a in 0.1 0.5 1.0 5.0; do
    python models/Snake_LSTM.py --dataset=multisine --a-init=${a} --seed=2025 --n-epochs=50
    python models/Snake_LSTM.py --dataset=narma --a-init=${a} --seed=2025 --n-epochs=50
done
```

### Expected Results

SIREN with lower w0 (e.g., 1.0 or 5.0) should perform better on time-series than w0=30. If so, update the main comparison tables to use the best SIREN configuration. This is fair — we want the strongest possible baselines to make the comparison convincing.

### Output

- **Table (appendix)**: Hyperparameter sweep results for SIREN and Snake
- Use the best configuration in all main paper tables

### Estimated Compute

8 configs × 2 datasets × ~0.2s/epoch × 50 = ~2 minutes (classical). Negligible.

---

## Contingency Plans

### If FourierQLSTM does NOT outperform parameter-matched classical models (H1 fails)

**Diagnosis**: Check if freq_scale values converged to meaningful frequencies. If they stayed near initialization, the model may need longer training.

**Actions**:
1. Increase epochs to 200 or 500
2. Try learning rate warmup (lr=0.001 for 20 epochs, then 0.01)
3. Try `lightning.qubit` backend with adjoint differentiation (faster per epoch)
4. If still failing: the contribution becomes purely architectural/methodological, and the paper repositions around "we identify the right design principles even though simulation-scale advantages are marginal"

### If extrapolation experiment (H2) shows no clear VQC advantage

**Diagnosis**: Check the signal beyond the training domain. If Multi-Sine extrapolation fails, something is fundamentally wrong with the data split.

**Actions**:
1. Verify the extrapolation data is truly temporally disjoint (no data leakage from sliding windows)
2. Increase training data size (VQC may need more data to learn the Fourier coefficients)
3. Reduce to simpler targets (single sinusoid, K=1) and build complexity gradually

### If spectral concentration does NOT predict VQC advantage (H4 fails)

**Diagnosis**: The theory may be correct but confounded by optimization difficulty (barren plateaus, local minima).

**Actions**:
1. Report the negative correlation honestly
2. Add a "practical limitations" section discussing optimization barriers
3. The paper becomes: "The theory predicts advantage, but optimization challenges prevent full realization at simulation scale"

### If multivariate benchmarks (Phase 6) are too slow to complete

**Actions**:
1. Reduce to ETTh1 only (most cited, auto-downloads)
2. Reduce to 20-30 epochs (enough to show clear trends if not full convergence)
3. Use a smaller kernel_size (e.g., 6 instead of 12) to reduce the number of sliding windows
4. Report as preliminary results and note that full-scale experiments require quantum hardware

---

## Computational Budget

### Estimated Total Compute (CPU hours)

| Phase | CPU Hours | GPU Hours (est.) | Priority |
|-------|:-:|:-:|:-:|
| Phase 0 (Spectral Profiling) | 0.1 | — | P0 |
| Phase 1 (Ablation: LSTM + TCN EEG) × 3 seeds | 35 | 4 | P0 |
| Phase 2 (Extrapolation) × 3 seeds | 33 | 4 | P0 |
| Phase 3 (Pareto: LSTM + TCN w/ EEG) × 3 seeds | 75 | 8 | P1 |
| Phase 4 (Init Comparison) × 3 seeds | 39 | 4 | P1 |
| Phase 5 (Qubit Scaling) | 140 | 14 | P2 |
| Phase 6 (Multivariate + EEG) × 3 seeds | 1,200 | 120 | P1 |
| Phase 7 (Statistical Runs) | (included above) | — | — |
| Phase 8 (Baseline Sweep) | 0.1 | — | P0 |
| **Total** | **~1,390** | **~140** | — |

### Recommended Compute Strategy

1. **Run Phase 0 and Phase 8 immediately** (minutes, CPU-only)
2. **Submit Phase 1 and Phase 2 as SLURM GPU jobs** (~10 GPU-hours combined, P0 priority)
3. **Submit Phase 3 and Phase 4** once Phase 1/2 confirm positive results (~15 GPU-hours)
4. **Submit Phase 6 (ETTh1 + PhysioNet EEG)** as long-running GPU jobs (~60 GPU-hours)
5. **Phase 5** only if Phase 1 confirms frequency-matching matters (~15 GPU-hours)
6. **Add seeds** to all positive-result experiments last

### SLURM Job Template

```bash
#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J vqc-phase1-ablation

export PYTHONNOUSERSITE=1

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/VQC-PeriodicData

# Phase 1: Ablation on Multi-Sine, 3 seeds
for seed in 2025 2026 2027; do
    echo "===== Full model, seed=${seed} ====="
    python models/FourierQLSTM.py --dataset=multisine --seed=${seed} \
        --n-qubits=6 --vqc-depth=2 --hidden-size=4 --window-size=8 \
        --n-epochs=50 --batch-size=10 --lr=0.01 --freq-init=fft

    echo "===== No FFT, seed=${seed} ====="
    python models/FourierQLSTM.py --dataset=multisine --seed=${seed} \
        --n-qubits=6 --vqc-depth=2 --hidden-size=4 --window-size=8 \
        --n-epochs=50 --batch-size=10 --lr=0.01 --ablate-fft

    # ... (similarly for other ablation variants)
done
```

---

## Paper Figure and Table Map

### Main Paper (~8 pages + references)

| # | Type | Content | Source Phase |
|:-:|:----:|---------|:-:|
| Fig 1 | Plot | Spectral concentration ρ(n) curves for all 8 datasets | Phase 0 |
| Fig 2 | Diagram | FourierQLSTM architecture with 4 innovations highlighted | (existing) |
| Fig 3 | Plot | Extrapolation degradation ratio vs distance: LSTM models × Multi-Sine, TCN models × ETTh1 | Phase 2 |
| Fig 4 | Plot | Pareto frontier: test MSE vs params (LSTM subplot + TCN subplot, separated) | Phase 3 |
| Table 1 | Results | Main comparison: LSTM models × univariate datasets + TCN models × multivariate datasets (interpolation MSE/AUC, mean ± std) | Phase 7 |
| Table 2 | Results | Ablation study: 5 variants × 3 datasets (Multi-Sine, NARMA-10, PhysioNet EEG) | Phase 1 |
| Table 3 | Results | Init strategy × dataset: convergence speed + final MSE | Phase 4 |
| Table 4 | Results | Full benchmarks: 6 TCN models × ETTh1/Weather/ECL/PhysioNet EEG | Phase 6 |

### Appendix / Supplementary

| # | Type | Content | Source Phase |
|:-:|:----:|---------|:-:|
| Fig A1 | Plot | Training curves for all ablation variants | Phase 1 |
| Fig A2 | Plot | freq_scale evolution during training (FFT vs linspace vs random init) | Phase 4 |
| Fig A3 | Plot | Test MSE vs n_qubits (qubit scaling) | Phase 5 |
| Table A1 | Results | SIREN w0 sweep + Snake a_init sweep | Phase 8 |
| Table A2 | Results | Full spectral profiles (top-20 freqs) for all datasets | Phase 0 |
| Table A3 | Results | Raw extrapolation MSE at each distance | Phase 2 |
| Table A4 | Results | Decomposed parameter counts (classical vs quantum) for all models | Phase 3 |
| Table A5 | Results | Statistical significance test p-values | Phase 7 |

---

## Timeline

### Recommended Execution Order

```
Week 1: Infrastructure + Phase 0 + Phase 8
├── Day 1-2: Implement spectral_analysis.py, ablation flags, extrapolation splits
├── Day 3: Run Phase 0 (spectral profiling, minutes)
├── Day 3: Run Phase 8 (baseline HP sweep, minutes)
└── Day 4-5: Implement param_matched_configs.py, experiment runner

Week 2: Core Experiments (Phase 1 + Phase 2)
├── Day 1-3: Submit Phase 1 (ablation, ~3 GPU-hours for 1 seed)
├── Day 1-3: Submit Phase 2 Multi-Sine (extrapolation, ~6 GPU-hours)
├── Day 4-5: Analyze Phase 1+2 results, iterate if needed
└── Day 5: Submit Phase 2 ETTh1 if Multi-Sine is positive

Week 3: Efficiency + Initialization (Phase 3 + Phase 4)
├── Day 1-2: Submit Phase 3 (Pareto, tiny budget × 3 seeds)
├── Day 1-2: Submit Phase 4 (init comparison × 3 seeds)
├── Day 3-5: Analyze results, begin paper writing
└── Day 5: Decide on Phase 5 (qubit scaling) based on results

Week 4: Multivariate + Scaling (Phase 5 + Phase 6)
├── Day 1-3: Submit Phase 6 ETTh1 (all models × 3 seeds)
├── Day 1-3: Submit Phase 5 if Phase 1 supports it
├── Day 4-5: Analyze all results
└── Day 5: Identify any gaps or follow-up experiments

Week 5: Statistical Significance + Paper Writing
├── Day 1-2: Submit remaining multi-seed runs for all positive results
├── Day 3-5: Generate all figures and tables
└── Day 5: Complete first draft of results section

Week 6: Paper Completion
├── Day 1-2: Write introduction, related work, methods sections
├── Day 3: Write discussion and conclusion
├── Day 4: Internal review and revision
└── Day 5: Submit or circulate for feedback
```

### Decision Gates

After each phase, evaluate whether to proceed:

| Gate | Condition to Proceed | Fallback |
|------|---------------------|----------|
| After Phase 1 | At least 2 ablation variants degrade ≥20% | Re-examine architecture; consider combined ablations |
| After Phase 2 | FourierQLSTM EDR < 3× on Multi-Sine | Check data pipeline; simplify to K=1 sinusoid |
| After Phase 3 | FourierQLSTM on Pareto frontier for at least 1 dataset | Increase quantum model size; try different backends |
| After Phase 6 | FourierQTCN competitive on at least ETTh1 + PhysioNet EEG | Report as preliminary; focus on LSTM results |

---

## Summary of Minimum Viable Experiments

If compute or time is limited, these are the **absolute minimum** experiments for a credible submission:

1. **Phase 0** (Spectral profiling) — 5 minutes, motivates the entire paper
2. **Phase 1** (Ablation, 1 seed) — 5.5 hours, proves design choices matter
3. **Phase 2** (Extrapolation on Multi-Sine, 1 seed) — 6 hours, the key differentiating experiment
4. **Phase 3** (Pareto at tiny budget, 1 seed) — 5 hours, addresses parameter fairness
5. **Phase 8** (Baseline HP sweep) — 2 minutes, ensures fair baselines

**Total minimum**: ~17 hours CPU / ~2 hours GPU for core results with 1 seed.

For a strong submission: add 3-seed runs for Phases 1-3 and full Phase 6 on ETTh1 + PhysioNet EEG.

---

## References

- Schuld, M., Sweke, R., & Meyer, J.J. (2021). Effect of data encoding on the expressive power of variational quantum machine-learning models. *Physical Review A*, 103(3), 032430.
- Yu, Z. et al. (2024). Non-asymptotic Approximation Error Bounds of Parameterized Quantum Circuits. *NeurIPS 2024*.
- Zhao, J. et al. (2024). Quantum Implicit Neural Representations. *ICML 2024*.
- Lewis, L., Gilboa, D., & McClean, J.R. (2025). Quantum advantage for learning shallow neural networks with natural data distributions. *Nature Communications*.
- Sitzmann, V. et al. (2020). Implicit Neural Representations with Periodic Activation Functions. *NeurIPS 2020*.
- Ziyin, L. et al. (2020). Neural networks fail to learn periodic functions and how to fix it. *NeurIPS 2020*.
- Zhou, H. et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI 2021*.
- Wu, H. et al. (2021). Autoformer. *NeurIPS 2021*.
- Nie, Y. et al. (2023). PatchTST. *ICLR 2023*.

---

*Plan created: February 2026 | Target: NeurIPS 2026 or ICML 2026*
