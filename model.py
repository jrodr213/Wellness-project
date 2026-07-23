"""Train a bidirectional LSTM autoencoder for wearable sensor windows.

Example:
    python3 model.py --csv sensor_data.csv --features heart_rate temperature accel_x accel_y accel_z

The script trains only a BiLSTM autoencoder and reports numerical
reconstruction errors. It does not scale data, cluster outputs, classify
anomalies, or provide medical interpretation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_EXCLUDED_COLUMNS = {"timestamp", "state", "label"}


class SensorWindowDataset(Dataset[torch.Tensor]):
    """PyTorch Dataset that returns sensor windows as float tensors."""

    def __init__(self, windows: np.ndarray) -> None:
        if windows.ndim != 3:
            raise ValueError(
                "Sensor windows must have shape "
                "[num_windows, sequence_length, number_of_sensor_features]."
            )
        self.windows = torch.from_numpy(windows.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


class BiLSTMAutoencoder(nn.Module):
    """Bidirectional LSTM encoder with a latent bottleneck and LSTM decoder."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        latent_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be greater than 0.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than 0.")
        if latent_size <= 0:
            raise ValueError("latent_size must be greater than 0.")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than 0.")

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.bottleneck = nn.Linear(hidden_size * 2, latent_size)
        self.decoder = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=False,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        _, (hidden, _) = self.encoder(x)

        # Concatenate the final forward and backward hidden states.
        final_forward = hidden[-2]
        final_backward = hidden[-1]
        encoded = torch.cat((final_forward, final_backward), dim=1)
        latent = self.bottleneck(encoded)

        repeated_latent = latent.unsqueeze(1).repeat(1, sequence_length, 1)
        decoded, _ = self.decoder(repeated_latent)
        return self.output_layer(decoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BiLSTM autoencoder on wearable sensor CSV data."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=(
            "Sensor columns to use. Accepts space-separated names, or comma-separated "
            "names such as heart_rate,temperature,accel_x."
        ),
    )
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Timestamp column to use for chronological sorting and reporting.",
    )
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--output-dir", default="model_outputs")
    return parser.parse_args()


def set_random_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_feature_args(features: list[str] | None) -> list[str] | None:
    if features is None:
        return None

    normalized: list[str] = []
    for item in features:
        normalized.extend(part.strip() for part in item.split(",") if part.strip())
    return normalized


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "window_size": args.window_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "num_layers": args.num_layers,
        "patience": args.patience,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than 0.")

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be greater than or equal to 0 and less than 1.")


def select_feature_columns(
    data: pd.DataFrame,
    requested_features: list[str] | None,
    timestamp_col: str,
) -> list[str]:
    if requested_features:
        missing = [column for column in requested_features if column not in data.columns]
        if missing:
            raise ValueError(
                "The following requested feature columns were not found in the CSV: "
                f"{missing}"
            )
        return requested_features

    excluded = set(DEFAULT_EXCLUDED_COLUMNS)
    excluded.add(timestamp_col)
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    features = [column for column in numeric_columns if column not in excluded]
    if not features:
        raise ValueError(
            "No numeric sensor columns were found. Provide columns with --features."
        )
    return features


def load_and_prepare_data(
    csv_path: str,
    requested_features: list[str] | None,
    timestamp_col: str,
) -> tuple[pd.DataFrame, list[str], bool]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    data = pd.read_csv(path)
    if data.empty:
        raise ValueError("The input CSV is empty.")

    has_timestamp = timestamp_col in data.columns
    if has_timestamp:
        data = data.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)

    feature_columns = select_feature_columns(data, requested_features, timestamp_col)

    numeric_data = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    numeric_data = numeric_data.interpolate(method="linear", limit_direction="both")

    usable_mask = ~numeric_data.isna().any(axis=1)
    prepared = data.loc[usable_mask].copy().reset_index(drop=True)
    prepared[feature_columns] = numeric_data.loc[usable_mask].to_numpy(dtype=np.float32)

    if prepared.empty:
        raise ValueError(
            "No usable rows remain after replacing infinite values and interpolating "
            "missing sensor readings."
        )
    if len(prepared) < 2:
        raise ValueError("At least two usable rows are required for training.")

    return prepared, feature_columns, has_timestamp


def create_sliding_windows(
    values: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if values.ndim != 2:
        raise ValueError("Sensor values must have shape [num_rows, num_features].")
    if len(values) < window_size:
        raise ValueError(
            f"Not enough usable rows ({len(values)}) for --window-size {window_size}."
        )

    windows: list[np.ndarray] = []
    indices: list[tuple[int, int]] = []
    for start in range(0, len(values) - window_size + 1, stride):
        end = start + window_size - 1
        windows.append(values[start : start + window_size])
        indices.append((start, end))

    if not windows:
        raise ValueError("No windows were created. Check --window-size and --stride.")

    return np.stack(windows).astype(np.float32), indices


def split_windows_chronologically(
    windows: np.ndarray,
    train_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(windows) < 2:
        raise ValueError(
            "At least two windows are required for chronological train/validation split."
        )

    train_count = int(len(windows) * train_ratio)
    train_count = max(1, min(train_count, len(windows) - 1))
    return windows[:train_count], windows[train_count:]


def make_dataloaders(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    train_loader = DataLoader(
        SensorWindowDataset(train_windows),
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        SensorWindowDataset(val_windows),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader[torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch in loader:
            batch = batch.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = int(batch.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader produced no samples.")
    return total_loss / total_samples


def train_model(
    model: BiLSTMAutoencoder,
    train_loader: DataLoader[torch.Tensor],
    val_loader: DataLoader[torch.Tensor],
    epochs: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": val_loss,
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"validation_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")

    model.load_state_dict(best_state)
    return history, best_state, best_val_loss


def save_training_history(history: list[dict[str, float]], output_dir: Path) -> None:
    history_frame = pd.DataFrame(history)
    history_frame["epoch"] = history_frame["epoch"].astype(int)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)


def save_training_plot(history: list[dict[str, float]], output_dir: Path) -> None:
    history_frame = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.plot(history_frame["epoch"], history_frame["train_loss"], label="Training loss")
    plt.plot(
        history_frame["epoch"],
        history_frame["validation_loss"],
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("BiLSTM Autoencoder Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()


def reconstruct_windows(
    model: BiLSTMAutoencoder,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        SensorWindowDataset(windows),
        batch_size=batch_size,
        shuffle=False,
    )
    reconstructed_batches: list[np.ndarray] = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstructed = model(batch).detach().cpu().numpy()
            reconstructed_batches.append(reconstructed)

    return np.concatenate(reconstructed_batches, axis=0)


def save_reconstruction_errors(
    windows: np.ndarray,
    reconstructed: np.ndarray,
    window_indices: list[tuple[int, int]],
    data: pd.DataFrame,
    feature_columns: list[str],
    timestamp_col: str,
    has_timestamp: bool,
    output_dir: Path,
) -> None:
    squared_errors = (reconstructed - windows) ** 2
    per_feature_errors = squared_errors.mean(axis=1)
    overall_errors = squared_errors.mean(axis=(1, 2))

    rows: list[dict[str, Any]] = []
    for window_number, (start_row, end_row) in enumerate(window_indices):
        row: dict[str, Any] = {
            "window_number": window_number,
            "window_start_row": start_row,
            "window_end_row": end_row,
        }
        if has_timestamp:
            row["ending_timestamp"] = data.loc[end_row, timestamp_col]
        for feature_index, feature_name in enumerate(feature_columns):
            row[f"{feature_name}_mean_reconstruction_error"] = float(
                per_feature_errors[window_number, feature_index]
            )
        row["overall_mean_reconstruction_error"] = float(overall_errors[window_number])
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / "reconstruction_errors.csv", index=False)


def build_model_config(args: argparse.Namespace, num_features: int) -> dict[str, Any]:
    return {
        "num_features": num_features,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "train_ratio": args.train_ratio,
        "stride": args.stride,
    }


def save_outputs(
    model: BiLSTMAutoencoder,
    best_state: dict[str, torch.Tensor],
    best_val_loss: float,
    history: list[dict[str, float]],
    args: argparse.Namespace,
    feature_columns: list[str],
    windows: np.ndarray,
    reconstructed: np.ndarray,
    window_indices: list[tuple[int, int]],
    prepared_data: pd.DataFrame,
    has_timestamp: bool,
    output_dir: Path,
) -> None:
    model_config = build_model_config(args, num_features=len(feature_columns))
    checkpoint = {
        "model_state_dict": best_state,
        "selected_feature_names": feature_columns,
        "window_size": args.window_size,
        "model_configuration": model_config,
        "best_validation_loss": best_val_loss,
    }

    torch.save(checkpoint, output_dir / "best_bilstm_autoencoder.pt")
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "selected_feature_names": feature_columns,
                "window_size": args.window_size,
                "model_configuration": model_config,
                "best_validation_loss": best_val_loss,
            },
            file,
            indent=2,
        )

    save_training_history(history, output_dir)
    save_training_plot(history, output_dir)
    np.save(output_dir / "original_windows.npy", windows)
    np.save(output_dir / "reconstructed_windows.npy", reconstructed)
    save_reconstruction_errors(
        windows=windows,
        reconstructed=reconstructed,
        window_indices=window_indices,
        data=prepared_data,
        feature_columns=feature_columns,
        timestamp_col=args.timestamp_col,
        has_timestamp=has_timestamp,
        output_dir=output_dir,
    )


def main() -> None:
    args = parse_args()
    args.features = normalize_feature_args(args.features)
    validate_args(args)
    set_random_seeds()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_data, feature_columns, has_timestamp = load_and_prepare_data(
        csv_path=args.csv,
        requested_features=args.features,
        timestamp_col=args.timestamp_col,
    )
    sensor_values = prepared_data[feature_columns].to_numpy(dtype=np.float32)
    windows, window_indices = create_sliding_windows(
        values=sensor_values,
        window_size=args.window_size,
        stride=args.stride,
    )
    train_windows, val_windows = split_windows_chronologically(
        windows=windows,
        train_ratio=args.train_ratio,
    )
    train_loader, val_loader = make_dataloaders(
        train_windows=train_windows,
        val_windows=val_windows,
        batch_size=args.batch_size,
    )

    device = choose_device()
    print(f"Using device: {device}")
    print(f"Selected features: {feature_columns}")
    print(
        f"Created {len(windows)} windows with shape "
        f"[batch_size, {args.window_size}, {len(feature_columns)}]."
    )

    model = BiLSTMAutoencoder(
        num_features=len(feature_columns),
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    history, best_state, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=device,
    )
    reconstructed = reconstruct_windows(
        model=model,
        windows=windows,
        batch_size=args.batch_size,
        device=device,
    )
    save_outputs(
        model=model,
        best_state=best_state,
        best_val_loss=best_val_loss,
        history=history,
        args=args,
        feature_columns=feature_columns,
        windows=windows,
        reconstructed=reconstructed,
        window_indices=window_indices,
        prepared_data=prepared_data,
        has_timestamp=has_timestamp,
        output_dir=output_dir,
    )
    print(f"Saved training outputs to: {output_dir}")


if __name__ == "__main__":
    main()
