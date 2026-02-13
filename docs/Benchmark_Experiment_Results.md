# Benchmark Experiment Results

Results from running all LSTM models on Multi-Sine, Mackey-Glass, and Adding Problem benchmarks.

**Date:** February 13, 2026
**Platform:** NERSC Perlmutter (CPU, login node)
**Quantum backend:** PennyLane `default.qubit` (statevector simulation, backprop differentiation)

---

## Experiment Configuration

### Shared Hyperparameters

| Parameter | Value |
|---|---|
| Random seed | 2025 |
| Number of qubits / MLP input dim | 6 |
| VQC depth / MLP depth | 2 |
| Hidden size | 4 |
| Output size | 1 |
| Input size | 1 (univariate) |
| Learning rate | 0.01 |
| Optimizer | Adam |
| Batch size | 10 |
| Number of epochs | 50 |
| Loss function | MSE |
| Train/test split | 67% / 33% (sequential) |

### Dataset-Specific Parameters

| Parameter | Multi-Sine | Mackey-Glass | Adding Problem |
|---|---|---|---|
| `--dataset` | `multisine` | `mackey_glass` | `adding` |
| `--n-samples` | 500 | 500 | 500 |
| `--window-size` | 8 | 8 | 50 |
| Actual input length | 8 | 8 | 100 (2 x T) |
| Sequences after windowing | 492 | 492 | 500 (no windowing) |
| Training samples | 329 | 329 | 335 |
| Test samples | 163 | 163 | 165 |
| Normalization | MinMaxScaler [-1, 1] | MinMaxScaler [-1, 1] | MinMaxScaler [-1, 1] (targets) |
| Data generation | K=5 sinusoids, noise_std=0.01 | tau=17, beta=0.2, gamma=0.1, n=10, warmup=500 | T=50, 2 marked positions per sample |

### Model-Specific Parameters

| Model | Type | Gate Mechanism | Extra Parameters |
|---|---|---|---|
| FourierQLSTM | Quantum | 4 FrequencyMatchedVQC circuits (RY + learnable freq_scale RX) | Learnable `freq_scale` per qubit |
| ReLU-LSTM | Classical | 4 ReLU MLP gates (hidden_dim=64) | `--mlp-hidden-dim=64` |
| Tanh-LSTM | Classical | 4 Tanh MLP gates (hidden_dim=64) | `--mlp-hidden-dim=64` |
| Snake-LSTM | Classical | 4 Snake MLP gates (hidden_dim=64) | `--a-init=1.0`, `--learnable-a=True` |
| SIREN-LSTM | Classical | 4 SIREN MLP gates (hidden_dim=64) | `--w0=30.0`, `--learnable-w0=True` |

---

## Parameter Counts

Parameter counts vary slightly between window_size=8 and window_size=50 because the FFT feature dimension changes (n_frequencies = window_size // 2 + 1, so FFT features = n_frequencies x 2 x input_size).

| Model | Params (window=8) | Params (window=50) |
|---|---|---|
| FourierQLSTM | 199 | — |
| ReLU-LSTM | 19,599 | 19,851 |
| Tanh-LSTM | 19,599 | 19,851 |
| Snake-LSTM | 20,111 | 20,363 |
| SIREN-LSTM | 19,607 | 19,859 |

Note: FourierQLSTM has ~100x fewer parameters than the classical models. The VQC gates have only variational rotation parameters (2 params per qubit per layer = 12 per gate) plus learnable frequency scales (6 per gate) and classical projection layers.

---

## Results

### 1. Multi-Sine (K=5)

Superposition of 5 incommensurate sinusoids: f = [0.1, 0.23, 0.37, 0.51, 0.79], amplitudes = [1.0, 0.8, 0.6, 0.4, 0.2].

| Model | Type | Params | Train MSE | Test MSE | Epoch-10 Test | Epoch-30 Test |
|---|---|---|---|---|---|---|
| **ReLU-LSTM** | Classical | 19,599 | 0.000433 | **0.000783** | 0.027745 | 0.002344 |
| FourierQLSTM | Quantum | 199 | 0.008695 | 0.009506 | 0.037912 | 0.008719 |
| Tanh-LSTM | Classical | 19,599 | 0.032260 | 0.018039 | 0.033619 | 0.022604 |
| Snake-LSTM | Classical | 20,111 | 0.102230 | 0.158275 | 0.017085 | 0.023598 |
| SIREN-LSTM | Classical | 19,607 | 0.150558 | 0.154268 | 0.154190 | 0.154638 |

**Training curves (every 10 epochs):**

| Epoch | FourierQLSTM | ReLU-LSTM | Tanh-LSTM | Snake-LSTM | SIREN-LSTM |
|---|---|---|---|---|---|
| 10 | 0.037912 | 0.027745 | 0.033619 | 0.017085 | 0.154190 |
| 20 | 0.012196 | 0.003393 | 0.027563 | 0.006961 | 0.154523 |
| 30 | 0.008719 | 0.002344 | 0.022604 | 0.023598 | 0.154638 |
| 40 | 0.007627 | 0.006388 | 0.010892 | 0.004826 | 0.154228 |
| 50 | 0.009506 | 0.000783 | 0.018039 | 0.158275 | 0.154268 |

### 2. Mackey-Glass (tau=17)

Quasi-periodic chaotic time-series from delay differential equation.

| Model | Type | Params | Train MSE | Test MSE | Epoch-10 Test | Epoch-30 Test |
|---|---|---|---|---|---|---|
| **ReLU-LSTM** | Classical | 19,599 | 0.005606 | **0.003309** | 0.066770 | 0.011086 |
| FourierQLSTM | Quantum | 199 | 0.010623 | 0.012190 | 0.037881 | 0.059906 |
| Tanh-LSTM | Classical | 19,599 | 0.241279 | 0.227769 | 0.233225 | 0.234172 |
| Snake-LSTM | Classical | 20,111 | 0.249073 | 0.234261 | 0.234256 | 0.234257 |
| SIREN-LSTM | Classical | 19,607 | 0.249612 | 0.234635 | 0.233376 | 0.234789 |

**Training curves (every 10 epochs):**

| Epoch | FourierQLSTM | ReLU-LSTM | Tanh-LSTM | Snake-LSTM | SIREN-LSTM |
|---|---|---|---|---|---|
| 10 | 0.037881 | 0.066770 | 0.233225 | 0.234256 | 0.233376 |
| 20 | 0.015451 | 0.017506 | 0.234128 | 0.234256 | 0.235880 |
| 30 | 0.059906 | 0.011086 | 0.234172 | 0.234257 | 0.234789 |
| 40 | 0.011513 | 0.007654 | 0.233904 | 0.234259 | 0.235562 |
| 50 | 0.012190 | 0.003309 | 0.227769 | 0.234261 | 0.234635 |

### 3. Adding Problem (T=50)

Long-range selective memory benchmark. Input = [signal; mask] of length 2T=100.

| Model | Type | Params | Train MSE | Test MSE | Epoch-10 Test | Epoch-30 Test |
|---|---|---|---|---|---|---|
| SIREN-LSTM | Classical | 19,859 | 0.179619 | **0.149002** | 0.150329 | 0.152825 |
| Snake-LSTM | Classical | 20,363 | 0.178918 | 0.149626 | 0.151487 | 0.151329 |
| ReLU-LSTM | Classical | 19,851 | 0.179498 | 0.149490 | 0.149490 | 0.149489 |
| Tanh-LSTM | Classical | 19,851 | 0.179499 | 0.149490 | 0.149490 | 0.149489 |
| FourierQLSTM | Quantum | — | DNF | DNF | — | — |

**DNF = Did not finish.** FourierQLSTM with window_size=50 creates 51 sliding windows per sample (input length 100, window 50). Each window requires 4 sequential VQC evaluations per sample (no batched quantum execution), making total runtime ~51x longer than window_size=8 experiments. Estimated wall-time: ~25 hours on CPU. Experiment was terminated after exceeding feasible runtime.

**Training curves (every 10 epochs):**

| Epoch | ReLU-LSTM | Tanh-LSTM | Snake-LSTM | SIREN-LSTM |
|---|---|---|---|---|
| 10 | 0.149490 | 0.149490 | 0.151487 | 0.150329 |
| 20 | 0.149489 | 0.149489 | 0.151221 | 0.149058 |
| 30 | 0.149489 | 0.149489 | 0.151329 | 0.152825 |
| 40 | 0.149489 | 0.149489 | 0.149627 | 0.149831 |
| 50 | 0.149490 | 0.149490 | 0.149626 | 0.149002 |

---

## Summary Table

Final test MSE across all experiments (lower is better). Best per dataset in **bold**.

| Model | Type | Multi-Sine | Mackey-Glass | Adding (T=50) |
|---|---|---|---|---|
| FourierQLSTM | Quantum | 0.009506 | 0.012190 | DNF |
| **ReLU-LSTM** | Classical | **0.000783** | **0.003309** | 0.149490 |
| Tanh-LSTM | Classical | 0.018039 | 0.227769 | 0.149490 |
| Snake-LSTM | Classical | 0.158275 | 0.234261 | 0.149626 |
| SIREN-LSTM | Classical | 0.154268 | 0.234635 | **0.149002** |

---

## Observations

### Multi-Sine

- **ReLU-LSTM** achieved the best final test MSE (0.000783), an order of magnitude better than FourierQLSTM (0.009506).
- **FourierQLSTM** ranked second with only 199 parameters vs. 19,599 for ReLU-LSTM — a **98x parameter efficiency advantage**. Per-parameter performance strongly favors the quantum model.
- **Tanh-LSTM** converged to a reasonable 0.018 but did not match ReLU or FourierQLSTM.
- **Snake-LSTM** showed instability: test MSE improved to 0.004826 at epoch 40 but then collapsed to 0.158275 at epoch 50, suggesting training instability with the learnable frequency parameter.
- **SIREN-LSTM** failed to learn, plateauing at ~0.154 from epoch 10 onward. The high initial w0=30.0 may cause optimization difficulties for this task.

### Mackey-Glass

- **ReLU-LSTM** again led (0.003309), with **FourierQLSTM** second (0.012190).
- Only ReLU-LSTM and FourierQLSTM learned meaningful dynamics. The other three models (Tanh, Snake, SIREN) all plateaued at ~0.23, essentially predicting near-constant output.
- FourierQLSTM showed some training instability (test MSE spiked to 0.059906 at epoch 30 before recovering), possibly due to the small number of variational parameters and sensitivity to local minima.

### Adding Problem

- All four classical models converged to nearly identical test MSE (~0.149), with negligible differences. This suggests all models learned approximately the same trivial strategy (likely predicting the mean target value) and **none solved the selective memory task** at T=50.
- The convergence to identical values from epoch 10 onward (especially ReLU and Tanh) strongly indicates a mean-prediction baseline rather than learning the marking positions.
- **FourierQLSTM could not be evaluated** due to the computational cost of per-sample VQC evaluation over long sequences. The sliding-window architecture creates 51 windows per sample at window_size=50, making quantum simulation ~51x slower than the window_size=8 experiments.

### Cross-Dataset Patterns

- **Parameter efficiency:** FourierQLSTM (199 params) achieved competitive or second-best results against classical models with ~100x more parameters on the periodic/quasi-periodic datasets.
- **Activation matters for periodic data:** Tanh, Snake, and SIREN all failed on Mackey-Glass (~0.23), while ReLU succeeded. This is surprising given that Snake and SIREN have explicit periodic structure — suggesting that the FFT preprocessing + frequency-domain cell state architecture matters more than the gate activation function alone.
- **Adding Problem baseline failure:** T=50 appears too difficult for all models in this configuration. The consistent ~0.149 test MSE across all models indicates a mean-prediction collapse. Future work should test smaller T values (e.g., T=10, T=20) or increase model capacity.

---

## Reproducibility

All experiments can be reproduced with:

```bash
cd /pscratch/sd/j/junghoon/VQC-PeriodicData
conda activate ./conda-envs/qml_eeg
export PYTHONNOUSERSITE=1

# Multi-Sine
python models/ReLU_LSTM.py    --dataset=multisine   --seed=2025
python models/Tanh_LSTM.py    --dataset=multisine   --seed=2025
python models/Snake_LSTM.py   --dataset=multisine   --seed=2025
python models/SIREN_LSTM.py   --dataset=multisine   --seed=2025
python models/FourierQLSTM.py --dataset=multisine   --seed=2025

# Mackey-Glass
python models/ReLU_LSTM.py    --dataset=mackey_glass --seed=2025
python models/Tanh_LSTM.py    --dataset=mackey_glass --seed=2025
python models/Snake_LSTM.py   --dataset=mackey_glass --seed=2025
python models/SIREN_LSTM.py   --dataset=mackey_glass --seed=2025
python models/FourierQLSTM.py --dataset=mackey_glass --seed=2025

# Adding Problem (classical only — FourierQLSTM infeasible at T=50)
python models/ReLU_LSTM.py    --dataset=adding --window-size=50 --seed=2025
python models/Tanh_LSTM.py    --dataset=adding --window-size=50 --seed=2025
python models/Snake_LSTM.py   --dataset=adding --window-size=50 --seed=2025
python models/SIREN_LSTM.py   --dataset=adding --window-size=50 --seed=2025
```

Raw output logs are saved in `results/benchmark_runs/`.
