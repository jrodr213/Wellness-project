"""Train the BiLSTM autoencoder from model.py on a user-provided CSV file.

Example:
    python3 run.py \
      --csv wellness_data.csv \
      --features heart_rate temperature accel_x accel_y accel_z capacitive \
      --window-size 50 \
      --stride 5 \
      --batch-size 32 \
      --epochs 50 \
      --output final_bilstm_autoencoder.pt

This script trains only the imported BiLSTM autoencoder architecture. It does
not scale data, cluster results, classify reconstruction errors, or provide
medical interpretation.
"""

from __future__ import annotations

import argparse
import importlib
import math
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


AUTO_FEATURE_EXCLUSIONS = {
    "timestamp",
    "state",
    "label",
    "cluster",
    "anomaly",
    "is_anomaly",
}
NUM_LSTM_LAYERS = 1
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the BiLSTM autoencoder from model.py on CSV sensor data."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=(
            "Sensor columns to train on. Accepts space-separated names or "
            "comma-separated groups."
        ),
    )
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--output", default="final_bilstm_autoencoder.pt")
    return parser.parse_args()


def normalize_features(features: list[str] | None) -> list[str] | None:
    if features is None:
        return None

    normalized: list[str] = []
    for feature_group in features:
        normalized.extend(
            feature.strip() for feature in feature_group.split(",") if feature.strip()
        )
    return normalized


def validate_args(args: argparse.Namespace) -> None:
    positive_int_args = {
        "window_size": args.window_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "latent_size": args.latent_size,
        "patience": args.patience,
    }
    for arg_name, value in positive_int_args.items():
        if value <= 0:
            raise ValueError(f"--{arg_name.replace('_', '-')} must be greater than 0.")

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be greater than 0.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be greater than or equal to 0 and less than 1.")


def import_required_modules() -> dict[str, Any]:
    model_path = Path("model.py")
    if not model_path.exists():
        raise FileNotFoundError(
            "model.py does not exist in the current directory. Run this script from "
            "the folder that contains model.py."
        )

    modules: dict[str, Any] = {}
    for module_name in ("numpy", "pandas", "torch"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Required package '{module_name}' is not installed. Activate the "
                "project virtual environment or install the required packages."
            ) from exc

    try:
        model_module = importlib.import_module("model")
    except ImportError as exc:
        raise ImportError("Could not import model.py successfully.") from exc

    for required_name in ("BiLSTMAutoencoder", "SensorWindowDataset", "choose_device"):
        if not hasattr(model_module, required_name):
            raise AttributeError(
                f"model.py is missing required reusable object: {required_name}"
            )
    modules["model"] = model_module
    return modules


def set_random_seeds(torch_module: ModuleType, numpy_module: ModuleType) -> None:
    random.seed(RANDOM_SEED)
    numpy_module.random.seed(RANDOM_SEED)
    torch_module.manual_seed(RANDOM_SEED)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(RANDOM_SEED)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False


def ensure_output_parent(output_path: Path) -> None:
    parent = output_path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if not parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {parent}")


def validate_csv_path(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"CSV path is not a file: {csv_path}")


def load_csv(csv_path: Path, pandas_module: ModuleType) -> Any:
    data = pandas_module.read_csv(csv_path)
    if data.empty:
        raise ValueError("CSV file is empty.")
    return data


def select_features(
    data: Any,
    requested_features: list[str] | None,
    timestamp_col: str,
    pandas_module: ModuleType,
    numpy_module: ModuleType,
) -> list[str]:
    if requested_features:
        missing = [feature for feature in requested_features if feature not in data.columns]
        if missing:
            raise ValueError(f"Requested feature is missing from CSV: {missing}")
        selected = requested_features
    else:
        excluded = set(AUTO_FEATURE_EXCLUSIONS)
        excluded.add(timestamp_col)
        selected = [
            column
            for column in data.select_dtypes(include=[numpy_module.number]).columns
            if column not in excluded
        ]

    if not selected:
        raise ValueError(
            "No sensor features were selected. Use --features with numeric columns."
        )

    non_numeric = [
        feature
        for feature in selected
        if not pandas_module.api.types.is_numeric_dtype(data[feature])
    ]
    if non_numeric:
        raise ValueError(f"Selected feature is not numeric: {non_numeric}")

    return selected


def print_csv_information(
    csv_path: Path,
    raw_data: Any,
    prepared_data: Any,
    selected_features: list[str],
    missing_counts: Any,
    timestamp_col: str,
) -> None:
    display_columns = selected_features.copy()
    if timestamp_col in prepared_data.columns:
        display_columns = [timestamp_col] + display_columns

    print("\nCSV information")
    print(f"CSV file path: {csv_path}")
    print(f"Number of rows: {len(raw_data)}")
    print(f"Number of columns: {len(raw_data.columns)}")
    print(f"All column names: {list(raw_data.columns)}")
    print(f"Selected sensor feature names: {selected_features}")
    print("Missing values in each selected feature:")
    print(missing_counts.to_string())
    print("First five usable rows:")
    print(prepared_data[display_columns].head().to_string(index=False))


def prepare_sensor_data(
    data: Any,
    selected_features: list[str],
    timestamp_col: str,
    numpy_module: ModuleType,
) -> tuple[Any, Any, int, int, bool]:
    if timestamp_col in data.columns:
        data = data.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
        has_timestamp = True
    else:
        data = data.reset_index(drop=True)
        has_timestamp = False

    numeric_data = data[selected_features].copy()
    unusable_before = numeric_data.isna() | numeric_data.isin(
        [numpy_module.inf, -numpy_module.inf]
    )
    rows_with_unusable_before = int(unusable_before.any(axis=1).sum())

    numeric_data = numeric_data.replace(
        [numpy_module.inf, -numpy_module.inf], numpy_module.nan
    )
    numeric_data = numeric_data.interpolate(method="linear", limit_direction="both")
    usable_mask = ~numeric_data.isna().any(axis=1)

    prepared = data.loc[usable_mask].copy().reset_index(drop=True)
    prepared[selected_features] = numeric_data.loc[usable_mask].to_numpy(dtype="float32")

    rows_removed = int(len(data) - len(prepared))
    rows_repaired = max(0, rows_with_unusable_before - rows_removed)
    if prepared.empty:
        raise ValueError(
            "No usable rows remain after replacing infinity values, interpolating "
            "missing values, and removing unusable rows."
        )

    return prepared, prepared[selected_features].to_numpy(dtype="float32"), rows_removed, rows_repaired, has_timestamp


def create_windows(
    sensor_values: Any,
    window_size: int,
    stride: int,
    numpy_module: ModuleType,
) -> Any:
    if len(sensor_values) < window_size:
        raise ValueError(
            f"Insufficient rows to create one window: {len(sensor_values)} usable rows "
            f"for window size {window_size}."
        )

    windows = [
        sensor_values[start : start + window_size]
        for start in range(0, len(sensor_values) - window_size + 1, stride)
    ]
    if not windows:
        raise ValueError("No windows were created. Check --window-size and --stride.")
    return numpy_module.stack(windows).astype("float32")


def split_windows(windows: Any, train_ratio: float) -> tuple[Any, Any]:
    if len(windows) < 2:
        raise ValueError(
            "No training or validation windows can be created because fewer than two "
            "windows are available."
        )

    train_count = int(len(windows) * train_ratio)
    train_count = max(1, min(train_count, len(windows) - 1))
    train_windows = windows[:train_count]
    validation_windows = windows[train_count:]

    if len(train_windows) == 0 or len(validation_windows) == 0:
        raise ValueError("No training or validation windows were produced.")
    return train_windows, validation_windows


def create_dataloaders(
    train_windows: Any,
    validation_windows: Any,
    batch_size: int,
    model_module: ModuleType,
    torch_module: ModuleType,
) -> tuple[Any, Any]:
    train_dataset = model_module.SensorWindowDataset(train_windows)
    validation_dataset = model_module.SensorWindowDataset(validation_windows)
    train_loader = torch_module.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = torch_module.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader


def count_trainable_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def print_model_summary(
    model: Any,
    args: argparse.Namespace,
    feature_count: int,
    parameter_count: int,
) -> None:
    print("\nModel summary")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Input feature count: {feature_count}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Latent size: {args.latent_size}")
    print(f"Dropout rate: {args.dropout}")
    print(f"Sequence length: {args.window_size}")
    print(f"Total trainable parameter count: {parameter_count}")
    print("Optimizer: Adam")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Maximum epochs: {args.epochs}")
    print(f"Patience: {args.patience}")


def run_one_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    torch_module: ModuleType,
    optimizer: Any | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0
    context = torch_module.enable_grad() if is_training else torch_module.no_grad()

    with context:
        for batch in loader:
            batch = batch.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                phase = "training" if is_training else "validation"
                raise ValueError(f"NaN or infinite {phase} loss detected.")

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = int(batch.shape[0])
            total_loss += loss_value * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader produced no samples.")
    return total_loss / total_samples


def train_with_early_stopping(
    model: Any,
    train_loader: Any,
    validation_loader: Any,
    args: argparse.Namespace,
    selected_features: list[str],
    feature_count: int,
    output_path: Path,
    torch_module: ModuleType,
    device: Any,
) -> dict[str, Any]:
    criterion = torch_module.nn.MSELoss()
    optimizer = torch_module.optim.Adam(model.parameters(), lr=args.learning_rate)

    training_history: list[float] = []
    validation_history: list[float] = []
    best_validation_loss = float("inf")
    best_training_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            torch_module=torch_module,
            optimizer=optimizer,
        )
        validation_loss = run_one_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            torch_module=torch_module,
        )

        training_history.append(train_loss)
        validation_history.append(validation_loss)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_training_loss = train_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "selected_feature_names": selected_features,
                "input_feature_count": feature_count,
                "window_size": args.window_size,
                "stride": args.stride,
                "hidden_size": args.hidden_size,
                "latent_size": args.latent_size,
                "dropout": args.dropout,
                "num_layers": NUM_LSTM_LAYERS,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "training_loss_history": training_history.copy(),
                "validation_loss_history": validation_history.copy(),
                "model_class_name": model.__class__.__name__,
            }
            torch_module.save(checkpoint, output_path)
            print(f"  Validation improved. Saved checkpoint to {output_path}")
        else:
            epochs_without_improvement += 1
            print(
                "  Validation did not improve. Early-stopping counter: "
                f"{epochs_without_improvement}/{args.patience}"
            )
            if epochs_without_improvement >= args.patience:
                early_stopped = True
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    return {
        "optimizer": optimizer,
        "training_history": training_history,
        "validation_history": validation_history,
        "best_epoch": best_epoch,
        "best_training_loss": best_training_loss,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(training_history),
        "early_stopped": early_stopped,
    }


def reload_and_validate_checkpoint(
    output_path: Path,
    args: argparse.Namespace,
    feature_count: int,
    validation_loader: Any,
    model_module: ModuleType,
    torch_module: ModuleType,
    device: Any,
) -> tuple[bool, tuple[int, ...], tuple[int, ...], float]:
    checkpoint = torch_module.load(output_path, map_location=device)
    reloaded_model = model_module.BiLSTMAutoencoder(
        num_features=feature_count,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_layers=NUM_LSTM_LAYERS,
        dropout=args.dropout,
    ).to(device)
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])
    reloaded_model.eval()

    criterion = torch_module.nn.MSELoss()
    validation_batch = next(iter(validation_loader)).to(device)
    with torch_module.no_grad():
        reconstructed_batch = reloaded_model(validation_batch)
        batch_loss = float(criterion(reconstructed_batch, validation_batch).item())

    if not math.isfinite(batch_loss):
        raise ValueError("NaN or infinite reconstruction loss detected after reload.")

    print("\nReload verification")
    print(f"Input tensor shape: {tuple(validation_batch.shape)}")
    print(f"Reconstructed tensor shape: {tuple(reconstructed_batch.shape)}")
    print(f"Reconstruction loss for validation batch: {batch_loss:.6f}")

    return (
        True,
        tuple(validation_batch.shape),
        tuple(reconstructed_batch.shape),
        batch_loss,
    )


def print_final_summary(
    train_result: dict[str, Any],
    output_path: Path,
    checkpoint_reloaded: bool,
) -> None:
    file_size = output_path.stat().st_size
    print("\nFinal summary")
    print(f"Early stopping occurred: {train_result['early_stopped']}")
    print(f"Number of epochs completed: {train_result['epochs_completed']}")
    print(f"Best epoch: {train_result['best_epoch']}")
    print(f"Best training loss at that point: {train_result['best_training_loss']:.6f}")
    print(f"Best validation loss: {train_result['best_validation_loss']:.6f}")
    print(f"Final model file path: {output_path}")
    print(f"Final model file size: {file_size} bytes")
    print(f"Checkpoint reloaded successfully: {checkpoint_reloaded}")


def main() -> None:
    args = parse_args()
    args.features = normalize_features(args.features)

    try:
        validate_args(args)
        modules = import_required_modules()
        numpy_module = modules["numpy"]
        pandas_module = modules["pandas"]
        torch_module = modules["torch"]
        model_module = modules["model"]

        set_random_seeds(torch_module=torch_module, numpy_module=numpy_module)

        csv_path = Path(args.csv)
        output_path = Path(args.output)
        validate_csv_path(csv_path)
        ensure_output_parent(output_path)

        raw_data = load_csv(csv_path=csv_path, pandas_module=pandas_module)
        selected_features = select_features(
            data=raw_data,
            requested_features=args.features,
            timestamp_col=args.timestamp_col,
            pandas_module=pandas_module,
            numpy_module=numpy_module,
        )
        missing_counts = raw_data[selected_features].isna().sum()
        prepared_data, sensor_values, rows_removed, rows_repaired, _ = prepare_sensor_data(
            data=raw_data,
            selected_features=selected_features,
            timestamp_col=args.timestamp_col,
            numpy_module=numpy_module,
        )
        print_csv_information(
            csv_path=csv_path,
            raw_data=raw_data,
            prepared_data=prepared_data,
            selected_features=selected_features,
            missing_counts=missing_counts,
            timestamp_col=args.timestamp_col,
        )
        print("\nData preparation")
        print(f"Rows repaired by interpolation: {rows_repaired}")
        print(f"Rows removed after preparation: {rows_removed}")
        print("Features were not standardized or normalized.")

        windows = create_windows(
            sensor_values=sensor_values,
            window_size=args.window_size,
            stride=args.stride,
            numpy_module=numpy_module,
        )
        print("\nWindowing")
        print(f"Window tensor shape: {windows.shape}")
        print(f"Total number of windows: {len(windows)}")

        train_windows, validation_windows = split_windows(
            windows=windows,
            train_ratio=args.train_ratio,
        )
        print("\nChronological split")
        print(f"Training windows: {len(train_windows)}")
        print(f"Validation windows: {len(validation_windows)}")

        train_loader, validation_loader = create_dataloaders(
            train_windows=train_windows,
            validation_windows=validation_windows,
            batch_size=args.batch_size,
            model_module=model_module,
            torch_module=torch_module,
        )

        device = model_module.choose_device()
        print(f"\nSelected device: {device}")

        feature_count = len(selected_features)
        model = model_module.BiLSTMAutoencoder(
            num_features=feature_count,
            hidden_size=args.hidden_size,
            latent_size=args.latent_size,
            num_layers=NUM_LSTM_LAYERS,
            dropout=args.dropout,
        ).to(device)
        parameter_count = count_trainable_parameters(model)
        print_model_summary(
            model=model,
            args=args,
            feature_count=feature_count,
            parameter_count=parameter_count,
        )

        print("\nTraining")
        train_result = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            args=args,
            selected_features=selected_features,
            feature_count=feature_count,
            output_path=output_path,
            torch_module=torch_module,
            device=device,
        )
        checkpoint_reloaded, _, _, _ = reload_and_validate_checkpoint(
            output_path=output_path,
            args=args,
            feature_count=feature_count,
            validation_loader=validation_loader,
            model_module=model_module,
            torch_module=torch_module,
            device=device,
        )
        print_final_summary(
            train_result=train_result,
            output_path=output_path,
            checkpoint_reloaded=checkpoint_reloaded,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
