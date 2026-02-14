#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J phase1_lstm
#SBATCH -o logs/phase1_lstm_%j.out
#SBATCH -e logs/phase1_lstm_%j.err

# Phase 1 Ablation: LSTM experiments (10 runs, ~8h total)
# Datasets: Multi-Sine + NARMA-10
# Variants: full, no_fft, no_freq_match, no_rescaled, no_fft_init

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/VQC-PeriodicData
mkdir -p logs

echo "=== Phase 1 Ablation: LSTM (10 experiments) ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

python experiments/run_phase1_ablation.py --run-all --seed=2025 --n-epochs=50 --model=FourierQLSTM

echo "Completed: $(date)"
