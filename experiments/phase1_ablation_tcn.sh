#!/bin/bash
#SBATCH -A m4807_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J phase1_tcn
#SBATCH -o logs/phase1_tcn_%A_%a.out
#SBATCH -e logs/phase1_tcn_%A_%a.err
#SBATCH --array=0-4

# Phase 1 Ablation: TCN EEG experiments (5 runs via job array)
# Each variant takes ~28h at 50 epochs on GPU
# Array index maps to variant:
#   0 = full, 1 = no_fft, 2 = no_freq_match, 3 = no_rescaled, 4 = no_fft_init

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate /pscratch/sd/j/junghoon/conda-envs/qml_eeg

cd /pscratch/sd/j/junghoon/VQC-PeriodicData
mkdir -p logs

VARIANTS=(full no_fft no_freq_match no_rescaled no_fft_init)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "=== Phase 1 Ablation: TCN EEG variant=${VARIANT} ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"

python experiments/run_phase1_ablation.py --run-all --seed=2025 --n-epochs=50 --model=FourierQTCN --variant=${VARIANT}

echo "Completed: $(date)"
