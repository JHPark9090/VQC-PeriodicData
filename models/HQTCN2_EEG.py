import scipy.constants  # Must be before pennylane (scipy 1.10.1 lazy-loading workaround)
import pennylane as qml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from typing import Tuple
import math, os, copy, time, random
import argparse

try:
    from Load_PhysioNet_EEG import load_eeg_ts_revised
except ImportError:
    try:
        import sys; sys.path.insert(0, os.path.dirname(__file__))
        from Load_PhysioNet_EEG import load_eeg_ts_revised
    except ImportError:
        print("Warning: Load_PhysioNet_EEG not found.")

try:
    from dataset_dispatcher import add_dataset_args, load_dataset
except ImportError:
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from dataset_dispatcher import add_dataset_args, load_dataset

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=int, default=80)
    parser.add_argument("--n-sample", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--kernel-size", type=int, default=12)
    parser.add_argument("--dilation", type=int, default=3)
    add_dataset_args(parser)
    return parser.parse_args()


print('Pennylane Version :', qml.__version__)
print('Pytorch Version :', torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on ", device)


def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG we rely on (Python, NumPy, Torch, PennyLane, CUDNN)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    qml.numpy.random.seed(seed)


class QTCN(nn.Module):
    def __init__(self, n_qubits, circuit_depth, input_dim, kernel_size, dilation=1):
        super(QTCN, self).__init__()
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        # Quantum parameters
        self.conv_params = nn.Parameter(torch.randn(circuit_depth, n_qubits, 15))
        self.pool_params = nn.Parameter(torch.randn(circuit_depth, n_qubits // 2, 3))
        # Quantum device initialization
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.quantum_circuit = qml.QNode(self.circuit, self.dev)

        self.input_channels = input_dim[1]
        self.time_steps = input_dim[2]
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.fc = nn.Linear(self.input_channels * self.kernel_size, n_qubits)

    def circuit(self, features):
        wires = list(range(self.n_qubits))
        qml.AngleEmbedding(features, wires=wires, rotation='Y')
        for layer in range(self.circuit_depth):
            self._apply_convolution(self.conv_params[layer], wires)
            self._apply_pooling(self.pool_params[layer], wires)
            wires = wires[::2]
        return qml.expval(qml.PauliZ(0))

    def _apply_convolution(self, weights, wires):
        n_wires = len(wires)
        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    qml.U3(*weights[indx, :3], wires=w)
                    qml.U3(*weights[indx + 1, 3:6], wires=wires[indx + 1])
                    qml.IsingZZ(weights[indx, 6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[indx, 7], wires=[w, wires[indx + 1]])
                    qml.IsingXX(weights[indx, 8], wires=[w, wires[indx + 1]])
                    qml.U3(*weights[indx, 9:12], wires=w)
                    qml.U3(*weights[indx + 1, 12:], wires=wires[indx + 1])

    def _apply_pooling(self, pool_weights, wires):
        n_wires = len(wires)
        assert n_wires >= 2, "Need at least two wires for pooling."
        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                measurement = qml.measure(w)
                qml.cond(measurement, qml.U3)(*pool_weights[indx // 2], wires=wires[indx - 1])

    def forward(self, x):
        batch_size, input_channels, time_steps = x.size()
        output = []
        for i in range(self.dilation * (self.kernel_size - 1), time_steps):
            indices = [i - d*self.dilation for d in range(self.kernel_size)]
            indices.reverse()
            window = x[:, :, indices].reshape(batch_size, -1)
            reduced_window = self.fc(window)
            output.append(self.quantum_circuit(reduced_window.double()).float())
        output = torch.mean(torch.stack(output, dim=1), dim=1)
        return output


################################# Calculate Running Time ########################################
def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


################################# Performance ################################
def train_perf(model, dataloader, optimizer, criterion, task='classification'):
    model.train()
    train_loss = 0.0
    all_labels = []
    all_outputs = []
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        all_labels.append(labels.cpu().numpy())
        all_outputs.append(outputs.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    if task == 'classification':
        try:
            metric = roc_auc_score(all_labels, all_outputs)
        except ValueError:
            metric = 0.5
    else:
        metric = np.sqrt(np.mean((all_labels - all_outputs) ** 2))

    return train_loss / len(dataloader), metric


def evaluate_perf(model, dataloader, criterion, task='classification'):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)

    if task == 'classification':
        try:
            metric = roc_auc_score(all_labels, all_outputs)
        except ValueError:
            metric = 0.5
    else:
        metric = np.sqrt(np.mean((all_labels - all_outputs) ** 2))

    return running_loss / len(dataloader), metric


def QuantumTCNN_run(seed, n_qubits, circuit_depth, input_dim,
                    train_loader, val_loader, test_loader, lr=0.001,
                    kernel_size=None, dilation=None, num_epochs=10,
                    checkpoint_dir=None, resume=False,
                    task='classification', dataset_name='eeg', args=None):
    print("Running on ", device)
    set_all_seeds(seed)
    print("Random Seed = ", seed)
    model = QTCN(n_qubits, circuit_depth, input_dim, kernel_size, dilation).to(device)

    # Task-specific criterion and tracking
    if task == 'classification':
        criterion = nn.BCEWithLogitsLoss()
        metric_name = 'auc'
        best_metric, is_better = 0.0, lambda new, old: new > old
    else:
        criterion = nn.MSELoss()
        metric_name = 'rmse'
        best_metric, is_better = float('inf'), lambda new, old: new < old

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(RESULTS_DIR, 'QTCN_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"q_tcnn_model_{dataset_name}_{seed}.pth")
    start_epoch = 0
    train_metrics, valid_metrics = [], []
    best_model_state = None

    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_metrics = checkpoint['train_metrics']
        valid_metrics = checkpoint['valid_metrics']
        best_metric = checkpoint.get('best_val_metric', best_metric)
        print(f"Resuming training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        train_loss, train_m = train_perf(model, train_loader, optimizer, criterion, task=task)
        train_metrics.append({'epoch': epoch + 1, 'train_loss': train_loss, f'train_{metric_name}': train_m})

        valid_loss, valid_m = evaluate_perf(model, val_loader, criterion, task=task)
        valid_metrics.append({'epoch': epoch + 1, 'valid_loss': valid_loss, f'valid_{metric_name}': valid_m})

        if is_better(valid_m, best_metric):
            best_metric = valid_m
            best_model_state = copy.deepcopy(model.state_dict())

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss: {train_loss:.4f}, {metric_name.upper()}: {train_m:.4f} | "
              f"Valid Loss: {valid_loss:.4f}, {metric_name.upper()}: {valid_m:.4f} (Best: {best_metric:.4f})")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'best_val_metric': best_metric,
        }, checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch + 1}")

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_loss, test_metric = evaluate_perf(model, test_loader, criterion, task=task)
    print(f"Test Loss: {test_loss:.4f}, {metric_name.upper()}: {test_metric:.4f}")

    metrics = []
    for i in range(len(train_metrics)):
        metrics.append({
            'epoch': i + 1,
            'train_loss': train_metrics[i]['train_loss'],
            f'train_{metric_name}': train_metrics[i][f'train_{metric_name}'],
            'valid_loss': valid_metrics[i]['valid_loss'],
            f'valid_{metric_name}': valid_metrics[i][f'valid_{metric_name}'],
            'test_loss': test_loss,
            f'test_{metric_name}': test_metric,
        })

    metrics_df = pd.DataFrame(metrics)
    metrics_dir = os.path.join(RESULTS_DIR, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    csv_filename = os.path.join(metrics_dir, f"QuantumTCNN_{dataset_name}_lr{lr}_performance_{seed}.csv")
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    return test_loss, test_metric



if __name__ == "__main__":
    args = get_args()

    # Load dataset via dispatcher
    train_loader, val_loader, test_loader, input_dim, task, scaler = load_dataset(args, device)
    print(f"Dataset: {args.dataset}, Task: {task}, Input dim: {input_dim}")

    kernel_size = args.kernel_size
    dilation = args.dilation

    # Adjust defaults for EEG frequency if using EEG dataset
    if args.dataset == 'eeg':
        if args.freq == 80:
            kernel_size, dilation = 12, 3
        elif args.freq == 4:
            kernel_size, dilation = 7, 2

    QuantumTCNN_run(
        seed=args.seed, n_qubits=8, circuit_depth=2, input_dim=input_dim,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        lr=args.lr, kernel_size=kernel_size, dilation=dilation,
        num_epochs=args.num_epochs, resume=args.resume,
        task=task, dataset_name=args.dataset, args=args
    )
