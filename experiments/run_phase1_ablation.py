"""
Phase 1 Ablation Study Runner — Tests Theorems 1, 2, 3

Runs 15 experiments (5 variants × 3 datasets) to show each of
FourierQLSTM's 4 design principles independently contributes.

Variants:
  Full          — Nothing ablated (baseline)
  A: No FFT     — FFT preprocessing → raw time-domain input
  B: No Freq-Match — RX encoding removed → RY-only (3^n vs 9^n freqs)
  C: No Rescaled Gate — (x+1)/2 → sigmoid(x)
  D: No FFT Init — FFT-seeded freq_scale → random uniform

Datasets:
  Multi-Sine (LSTM, ρ=1.00)
  NARMA-10   (LSTM, ρ=0.30)
  PhysioNet EEG (TCN, ρ=0.53)

Usage:
  python experiments/run_phase1_ablation.py --run-all --seed=2025
  python experiments/run_phase1_ablation.py --collect-only
  python experiments/run_phase1_ablation.py --run-all --n-epochs=3  # smoke test
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1_ablation"
MODELS_DIR = PROJECT_ROOT / "models"


# ─────────────────────────────────────────────────────────────────────
# Experiment Matrix
# ─────────────────────────────────────────────────────────────────────

LSTM_COMMON = (
    "--n-qubits=6 --vqc-depth=2 --hidden-size=4 "
    "--window-size=8 --batch-size=10 --lr=0.01"
)

TCN_COMMON = (
    "--n-qubits=8 --circuit-depth=2 --kernel-size=12 "
    "--dilation=3"
)


def build_experiments(seed, n_epochs_lstm, n_epochs_tcn):
    """Build the 15-experiment matrix."""
    output_dir = str(RESULTS_DIR)
    lstm_base = f"{LSTM_COMMON} --seed={seed} --n-epochs={n_epochs_lstm} --output-dir={output_dir}"
    tcn_base = f"{TCN_COMMON} --seed={seed} --num-epochs={n_epochs_tcn} --output-dir={output_dir}"

    experiments = [
        # ── LSTM × Multi-Sine ──
        {"model": "FourierQLSTM", "dataset": "multisine", "variant": "full",
         "flags": f"--dataset=multisine --freq-init=fft {lstm_base}"},
        {"model": "FourierQLSTM", "dataset": "multisine", "variant": "no_fft",
         "flags": f"--dataset=multisine --ablate-fft --freq-init=linspace {lstm_base}"},
        {"model": "FourierQLSTM", "dataset": "multisine", "variant": "no_freq_match",
         "flags": f"--dataset=multisine --ablate-freq-match --freq-init=fft {lstm_base}"},
        {"model": "FourierQLSTM", "dataset": "multisine", "variant": "no_rescaled",
         "flags": f"--dataset=multisine --ablate-rescaled --freq-init=fft {lstm_base}"},
        {"model": "FourierQLSTM", "dataset": "multisine", "variant": "no_fft_init",
         "flags": f"--dataset=multisine --freq-init=random {lstm_base}"},

        # ── LSTM × NARMA-10 ──
        {"model": "FourierQLSTM", "dataset": "narma", "variant": "full",
         "flags": f"--dataset=narma --freq-init=fft {lstm_base}"},
        {"model": "FourierQLSTM", "dataset": "narma", "variant": "no_fft",
         "flags": f"--dataset=narma --ablate-fft --freq-init=linspace {lstm_base}"},
        {"model": "FourierQLSTM", "dataset": "narma", "variant": "no_freq_match",
         "flags": f"--dataset=narma --ablate-freq-match --freq-init=fft {lstm_base}"},
        {"model": "FourierQLSTM", "dataset": "narma", "variant": "no_rescaled",
         "flags": f"--dataset=narma --ablate-rescaled --freq-init=fft {lstm_base}"},
        {"model": "FourierQLSTM", "dataset": "narma", "variant": "no_fft_init",
         "flags": f"--dataset=narma --freq-init=random {lstm_base}"},

        # ── TCN × PhysioNet EEG ──
        {"model": "FourierQTCN", "dataset": "eeg", "variant": "full",
         "flags": f"--dataset=eeg --freq-init=fft {tcn_base}"},
        {"model": "FourierQTCN", "dataset": "eeg", "variant": "no_fft",
         "flags": f"--dataset=eeg --ablate-fft --freq-init=linspace {tcn_base}"},
        {"model": "FourierQTCN", "dataset": "eeg", "variant": "no_freq_match",
         "flags": f"--dataset=eeg --ablate-freq-match --freq-init=fft {tcn_base}"},
        {"model": "FourierQTCN", "dataset": "eeg", "variant": "no_rescaled",
         "flags": f"--dataset=eeg --ablate-rescaled --freq-init=fft {tcn_base}"},
        {"model": "FourierQTCN", "dataset": "eeg", "variant": "no_fft_init",
         "flags": f"--dataset=eeg --freq-init=random {tcn_base}"},
    ]
    return experiments


# ─────────────────────────────────────────────────────────────────────
# Run Experiments
# ─────────────────────────────────────────────────────────────────────

def run_experiment(exp, dry_run=False):
    """Run a single experiment via subprocess."""
    if exp["model"] == "FourierQLSTM":
        script = str(MODELS_DIR / "FourierQLSTM.py")
    else:
        script = str(MODELS_DIR / "FourierQTCN_EEG.py")

    cmd = f"{sys.executable} {script} {exp['flags']}"
    label = f"{exp['model']}_{exp['dataset']}_{exp['variant']}"

    print(f"\n{'='*70}")
    print(f"[{label}] Running...")
    print(f"  CMD: {cmd}")
    print(f"{'='*70}")

    if dry_run:
        print("  (dry run — skipped)")
        return 0

    result = subprocess.run(
        cmd, shell=True, cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONNOUSERSITE": "1"}
    )

    if result.returncode != 0:
        print(f"  WARNING: {label} exited with code {result.returncode}")
    else:
        print(f"  {label} completed successfully.")

    return result.returncode


def run_all(experiments, dry_run=False):
    """Run all experiments sequentially."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    successes, failures = 0, 0
    for i, exp in enumerate(experiments, 1):
        print(f"\n>>> Experiment {i}/{len(experiments)}")
        rc = run_experiment(exp, dry_run=dry_run)
        if rc == 0:
            successes += 1
        else:
            failures += 1

    print(f"\n{'='*70}")
    print(f"All experiments done: {successes} succeeded, {failures} failed")
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────
# Collect Results
# ─────────────────────────────────────────────────────────────────────

def parse_final_metric(csv_path, model_type, task):
    """
    Parse the final-epoch metric from a metrics CSV.

    LSTM CSVs have columns: epoch, train_loss, test_loss
    TCN  CSVs have columns: epoch, train_loss, train_auc/rmse, valid_loss, valid_auc/rmse, test_loss, test_auc/rmse
    """
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    last = rows[-1]

    if model_type == "FourierQLSTM":
        # Return final test MSE
        return float(last.get('test_loss', 0))
    else:
        # TCN: return test AUC or test RMSE
        if task == 'classification':
            return float(last.get('test_auc', 0))
        else:
            return float(last.get('test_rmse', 0))


def collect_results(seed):
    """Collect results from all CSV files and generate summary."""
    VARIANT_LABELS = {
        "full": "Full Model",
        "no_fft": "A: No FFT",
        "no_freq_match": "B: No Freq-Match",
        "no_rescaled": "C: No Rescaled Gate",
        "no_fft_init": "D: No FFT Init",
    }
    VARIANT_ORDER = ["full", "no_fft", "no_freq_match", "no_rescaled", "no_fft_init"]

    results = {}

    for variant in VARIANT_ORDER:
        results[variant] = {}

        # LSTM Multi-Sine
        csv_path = RESULTS_DIR / f"FourierQLSTM_multisine_{variant}_seed{seed}_metrics.csv"
        results[variant]["multisine"] = parse_final_metric(str(csv_path), "FourierQLSTM", "regression")

        # LSTM NARMA
        csv_path = RESULTS_DIR / f"FourierQLSTM_narma_{variant}_seed{seed}_metrics.csv"
        results[variant]["narma"] = parse_final_metric(str(csv_path), "FourierQLSTM", "regression")

        # TCN EEG
        csv_path = RESULTS_DIR / f"FourierQTCN_eeg_{variant}_seed{seed}_metrics.csv"
        results[variant]["eeg"] = parse_final_metric(str(csv_path), "FourierQTCN", "classification")

    # ── Print Table 2 ──
    header = f"{'Variant':<25} {'Multi-Sine MSE↓':>17} {'NARMA-10 MSE↓':>15} {'EEG AUC↑':>12}"
    sep = "-" * len(header)

    lines = [
        "Phase 1 Ablation Study Results (Table 2)",
        f"Seed: {seed}",
        "",
        header,
        sep,
    ]

    summary_rows = []
    for variant in VARIANT_ORDER:
        label = VARIANT_LABELS[variant]
        ms = results[variant]["multisine"]
        nr = results[variant]["narma"]
        eeg = results[variant]["eeg"]

        ms_str = f"{ms:.6f}" if ms is not None else "N/A"
        nr_str = f"{nr:.6f}" if nr is not None else "N/A"
        eeg_str = f"{eeg:.4f}" if eeg is not None else "N/A"

        lines.append(f"{label:<25} {ms_str:>17} {nr_str:>15} {eeg_str:>12}")
        summary_rows.append({
            "variant": variant,
            "label": label,
            "multisine_mse": ms,
            "narma_mse": nr,
            "eeg_auc": eeg,
        })

    lines.append(sep)

    table_text = "\n".join(lines)
    print(f"\n{table_text}")

    # Save summary text
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = RESULTS_DIR / "ablation_summary.txt"
    with open(txt_path, 'w') as f:
        f.write(table_text + "\n")
    print(f"\nSummary saved to {txt_path}")

    # Save summary CSV
    csv_path = RESULTS_DIR / "ablation_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "label", "multisine_mse", "narma_mse", "eeg_auc"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV saved to {csv_path}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Ablation Study Runner (Table 2)"
    )
    parser.add_argument("--run-all", action="store_true",
                        help="Run all 15 experiments")
    parser.add_argument("--collect-only", action="store_true",
                        help="Only collect results from existing CSVs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Random seed (default: 2025)")
    parser.add_argument("--n-epochs", type=int, default=50,
                        help="Number of epochs for LSTM experiments (default: 50)")
    parser.add_argument("--n-epochs-tcn", type=int, default=None,
                        help="Number of epochs for TCN experiments (default: same as --n-epochs)")
    parser.add_argument("--model", type=str, default=None,
                        choices=["FourierQLSTM", "FourierQTCN"],
                        help="Only run experiments for this model type")
    parser.add_argument("--variant", type=str, default=None,
                        choices=["full", "no_fft", "no_freq_match", "no_rescaled", "no_fft_init"],
                        help="Only run this ablation variant")
    args = parser.parse_args()

    n_epochs_tcn = args.n_epochs_tcn or args.n_epochs

    if args.collect_only:
        collect_results(args.seed)
        return

    if args.run_all or args.dry_run:
        experiments = build_experiments(args.seed, args.n_epochs, n_epochs_tcn)
        # Apply filters
        if args.model:
            experiments = [e for e in experiments if e["model"] == args.model]
        if args.variant:
            experiments = [e for e in experiments if e["variant"] == args.variant]
        if not experiments:
            print("No experiments match the given filters.")
            return
        run_all(experiments, dry_run=args.dry_run)
        if not args.dry_run:
            collect_results(args.seed)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
