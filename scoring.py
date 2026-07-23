"""Score CSV sensor windows with a trained BiLSTM autoencoder checkpoint.

Example:
    python3 scoring.py \
      --csv wellness_data.csv \
      --model final_bilstm_autoencoder.pt \
      --output-csv reconstruction_scores.csv

The script reconstructs chronological windows and saves numerical mean squared
reconstruction errors. It does not train the model, update weights, scale data,
cluster scores, apply thresholds, classify windows, or provide medical labels.
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


RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score CSV windows with a trained BiLSTM autoencoder checkpoint."
    )
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--model", required=True, help="Path to the .pt checkpoint.")
    parser.add_argument(
        "--output-csv",
        default="reconstruction_scores.csv",
        help="Path where reconstruction scores will be saved.",
    )
    parser.add_argument(
        "--timestamp-col",
        default="timestamp",
        help="Timestamp column used for chronological sorting and reporting.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def import_required_modules() -> dict[str, Any]:
    if not Path("model.py").exists():
        raise FileNotFoundError(
            "model.py does not exist in the current directory. Run scoring.py from "
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


def validate_paths(csv_path: Path, checkpoint_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV file does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Input CSV path is not a file: {csv_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")


def ensure_output_parent(output_path: Path) -> None:
    parent = output_path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if not parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {parent}")


def load_checkpoint(
    checkpoint_path: Path,
    torch_module: ModuleType,
    device: Any,
) -> dict[str, Any]:
    try:
        checkpoint = torch_module.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise ValueError(f"Failed to load checkpoint: {checkpoint_path}") from exc

    if not isinstance(checkpoint, dict):
        raise ValueError("Incompatible checkpoint: expected a dictionary.")
    return checkpoint


def require_checkpoint_value(
    checkpoint: dict[str, Any],
    key: str,
    fallback_config: dict[str, Any] | None = None,
) -> Any:
    if key in checkpoint:
        return checkpoint[key]
    if fallback_config is not None and key in fallback_config:
        return fallback_config[key]
    raise ValueError(f"Missing checkpoint metadata: {key}")


def extract_checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    model_config = checkpoint.get("model_configuration")
    if model_config is not None and not isinstance(model_config, dict):
        raise ValueError("Incompatible checkpoint: model_configuration must be a dict.")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError("Missing checkpoint metadata: model_state_dict")

    selected_features = checkpoint.get("selected_feature_names")
    if selected_features is None:
        selected_features = checkpoint.get("feature_columns")
    if not isinstance(selected_features, list) or not selected_features:
        raise ValueError("Missing checkpoint metadata: selected feature names.")

    fallback = model_config or {}
    input_feature_count = checkpoint.get("input_feature_count")
    if input_feature_count is None:
        input_feature_count = fallback.get("num_features", len(selected_features))

    window_size = require_checkpoint_value(checkpoint, "window_size", fallback)
    stride = require_checkpoint_value(checkpoint, "stride", fallback)
    hidden_size = require_checkpoint_value(checkpoint, "hidden_size", fallback)
    latent_size = require_checkpoint_value(checkpoint, "latent_size", fallback)
    dropout = require_checkpoint_value(checkpoint, "dropout", fallback)
    num_layers = checkpoint.get("num_layers", fallback.get("num_layers", 1))
    model_class_name = checkpoint.get("model_class_name", "BiLSTMAutoencoder")

    config = {
        "selected_features": [str(feature) for feature in selected_features],
        "input_feature_count": int(input_feature_count),
        "window_size": int(window_size),
        "stride": int(stride),
        "hidden_size": int(hidden_size),
        "latent_size": int(latent_size),
        "dropout": float(dropout),
        "num_layers": int(num_layers),
        "model_state_dict": state_dict,
        "model_class_name": str(model_class_name),
    }

    if config["input_feature_count"] != len(config["selected_features"]):
        raise ValueError(
            "Incompatible checkpoint: input_feature_count does not match the number "
            "of selected feature names."
        )
    if config["window_size"] <= 0 or config["stride"] <= 0:
        raise ValueError("Incompatible checkpoint: window size and stride must be > 0.")
    if config["hidden_size"] <= 0 or config["latent_size"] <= 0:
        raise ValueError("Incompatible checkpoint: hidden and latent sizes must be > 0.")
    if config["num_layers"] <= 0:
        raise ValueError("Incompatible checkpoint: num_layers must be > 0.")

    return config


def load_csv(csv_path: Path, pandas_module: ModuleType) -> Any:
    data = pandas_module.read_csv(csv_path)
    if data.empty:
        raise ValueError("Input CSV is empty.")
    return data


def validate_feature_columns(
    data: Any,
    selected_features: list[str],
    pandas_module: ModuleType,
) -> None:
    missing = [feature for feature in selected_features if feature not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")

    nonnumeric = [
        feature
        for feature in selected_features
        if not pandas_module.api.types.is_numeric_dtype(data[feature])
    ]
    if nonnumeric:
        raise ValueError(f"Nonnumeric features in CSV: {nonnumeric}")


def prepare_sensor_data(
    data: Any,
    selected_features: list[str],
    timestamp_col: str,
    numpy_module: ModuleType,
) -> tuple[Any, Any, bool]:
    if timestamp_col in data.columns:
        sorted_data = data.sort_values(timestamp_col, kind="mergesort").reset_index(
            drop=True
        )
        has_timestamp = True
    else:
        sorted_data = data.reset_index(drop=True)
        has_timestamp = False

    numeric_data = sorted_data[selected_features].copy()
    numeric_data = numeric_data.replace(
        [numpy_module.inf, -numpy_module.inf], numpy_module.nan
    )
    numeric_data = numeric_data.interpolate(method="linear", limit_direction="both")
    usable_mask = ~numeric_data.isna().any(axis=1)

    prepared = sorted_data.loc[usable_mask].copy().reset_index(drop=True)
    prepared[selected_features] = numeric_data.loc[usable_mask].to_numpy(dtype="float32")

    if prepared.empty:
        raise ValueError(
            "No usable rows remain after replacing infinity values, interpolating "
            "missing values, and removing invalid selected feature values."
        )

    sensor_values = prepared[selected_features].to_numpy(dtype="float32")
    return prepared, sensor_values, has_timestamp


def create_windows_with_indices(
    sensor_values: Any,
    window_size: int,
    stride: int,
    model_module: ModuleType,
) -> tuple[Any, list[tuple[int, int]]]:
    try:
        windows, indices = model_module.create_sliding_windows(
            sensor_values,
            window_size,
            stride,
        )
    except ValueError as exc:
        raise ValueError(
            "Insufficient rows to create a window or invalid window settings: "
            f"{exc}"
        ) from exc

    if len(windows) == 0:
        raise ValueError("No windows were generated.")
    return windows, indices


def build_model(
    config: dict[str, Any],
    model_module: ModuleType,
    torch_module: ModuleType,
    device: Any,
) -> Any:
    if config["model_class_name"] != "BiLSTMAutoencoder":
        raise ValueError(
            "Incompatible checkpoint: expected model class BiLSTMAutoencoder, found "
            f"{config['model_class_name']}."
        )

    model = model_module.BiLSTMAutoencoder(
        num_features=config["input_feature_count"],
        hidden_size=config["hidden_size"],
        latent_size=config["latent_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    try:
        model.load_state_dict(config["model_state_dict"])
    except Exception as exc:
        raise ValueError("Failed to load the checkpoint weights into the model.") from exc

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Model parameters should not require gradients while scoring.")

    return model


def reconstruct_windows(
    model: Any,
    windows: Any,
    batch_size: int,
    model_module: ModuleType,
    torch_module: ModuleType,
    device: Any,
) -> Any:
    dataset = model_module.SensorWindowDataset(windows)
    loader = torch_module.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    reconstructed_batches: list[Any] = []

    with torch_module.no_grad():
        for batch in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            if tuple(reconstructed.shape) != tuple(batch.shape):
                raise ValueError(
                    "Model input or output shape mismatch: "
                    f"input {tuple(batch.shape)}, reconstructed "
                    f"{tuple(reconstructed.shape)}."
                )
            reconstructed_batches.append(reconstructed.detach().cpu())

    reconstructed_all = torch_module.cat(reconstructed_batches, dim=0)
    if tuple(reconstructed_all.shape) != tuple(windows.shape):
        raise ValueError(
            "Model input or output shape mismatch after batching: "
            f"input {tuple(windows.shape)}, reconstructed "
            f"{tuple(reconstructed_all.shape)}."
        )
    return reconstructed_all


def calculate_scores(
    windows: Any,
    reconstructed: Any,
    torch_module: ModuleType,
) -> tuple[Any, Any]:
    original = torch_module.as_tensor(windows, dtype=torch_module.float32)
    squared_error = (original - reconstructed) ** 2
    feature_scores = squared_error.mean(dim=1)
    final_scores = squared_error.mean(dim=(1, 2))

    if not torch_module.isfinite(feature_scores).all() or not torch_module.isfinite(
        final_scores
    ).all():
        raise ValueError("NaN or infinite reconstruction scores detected.")

    return feature_scores.cpu().numpy(), final_scores.cpu().numpy()


def build_results_frame(
    feature_scores: Any,
    final_scores: Any,
    window_indices: list[tuple[int, int]],
    prepared_data: Any,
    selected_features: list[str],
    timestamp_col: str,
    has_timestamp: bool,
    pandas_module: ModuleType,
) -> Any:
    rows: list[dict[str, Any]] = []
    for window_number, (start_row, end_row) in enumerate(window_indices):
        row: dict[str, Any] = {
            "window_number": window_number,
            "window_start_row": start_row,
            "window_end_row": end_row,
        }
        if has_timestamp:
            row["window_start_timestamp"] = prepared_data.loc[start_row, timestamp_col]
            row["window_end_timestamp"] = prepared_data.loc[end_row, timestamp_col]

        for feature_index, feature_name in enumerate(selected_features):
            row[f"{feature_name}_mse"] = float(
                feature_scores[window_number, feature_index]
            )
        row["final_reconstruction_score"] = float(final_scores[window_number])
        rows.append(row)

    return pandas_module.DataFrame(rows)


def print_scoring_inputs(
    csv_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: Any,
    config: dict[str, Any],
    input_row_count: int,
    usable_row_count: int,
    windows: Any,
) -> None:
    print("Scoring inputs")
    print(f"CSV path: {csv_path}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Output path: {output_path}")
    print(f"Selected device: {device}")
    print(f"Selected features: {config['selected_features']}")
    print(f"Window size: {config['window_size']}")
    print(f"Stride: {config['stride']}")
    print(f"Input row count: {input_row_count}")
    print(f"Usable row count: {usable_row_count}")
    print(f"Number of generated windows: {len(windows)}")
    print(f"Complete tensor shape: {windows.shape}")


def print_summary(final_scores: Any, output_path: Path, numpy_module: ModuleType) -> None:
    print("\nScoring summary")
    print(f"Number of scored windows: {len(final_scores)}")
    print(f"Minimum final reconstruction score: {float(numpy_module.min(final_scores)):.12g}")
    print(f"Average final reconstruction score: {float(numpy_module.mean(final_scores)):.12g}")
    print(f"Median final reconstruction score: {float(numpy_module.median(final_scores)):.12g}")
    print(f"Maximum final reconstruction score: {float(numpy_module.max(final_scores)):.12g}")
    print(f"Standard deviation of final reconstruction scores: {float(numpy_module.std(final_scores)):.12g}")
    print(f"Output CSV path: {output_path}")
    print("Confirmation: no training or weight updates occurred.")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        print("Error: --batch-size must be greater than 0.", file=sys.stderr)
        raise SystemExit(1)

    try:
        modules = import_required_modules()
        numpy_module = modules["numpy"]
        pandas_module = modules["pandas"]
        torch_module = modules["torch"]
        model_module = modules["model"]

        set_random_seeds(torch_module=torch_module, numpy_module=numpy_module)

        csv_path = Path(args.csv)
        checkpoint_path = Path(args.model)
        output_path = Path(args.output_csv)
        validate_paths(csv_path=csv_path, checkpoint_path=checkpoint_path)
        ensure_output_parent(output_path)

        device = model_module.choose_device()
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            torch_module=torch_module,
            device=device,
        )
        config = extract_checkpoint_config(checkpoint)

        raw_data = load_csv(csv_path=csv_path, pandas_module=pandas_module)
        validate_feature_columns(
            data=raw_data,
            selected_features=config["selected_features"],
            pandas_module=pandas_module,
        )
        prepared_data, sensor_values, has_timestamp = prepare_sensor_data(
            data=raw_data,
            selected_features=config["selected_features"],
            timestamp_col=args.timestamp_col,
            numpy_module=numpy_module,
        )
        windows, window_indices = create_windows_with_indices(
            sensor_values=sensor_values,
            window_size=config["window_size"],
            stride=config["stride"],
            model_module=model_module,
        )
        if tuple(windows.shape[1:]) != (
            config["window_size"],
            config["input_feature_count"],
        ):
            raise ValueError(
                "Model input or output shape mismatch: generated windows have shape "
                f"{tuple(windows.shape)}, but checkpoint expects "
                f"[number_of_windows, {config['window_size']}, "
                f"{config['input_feature_count']}]."
            )

        print_scoring_inputs(
            csv_path=csv_path,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            device=device,
            config=config,
            input_row_count=len(raw_data),
            usable_row_count=len(prepared_data),
            windows=windows,
        )

        model = build_model(
            config=config,
            model_module=model_module,
            torch_module=torch_module,
            device=device,
        )
        reconstructed = reconstruct_windows(
            model=model,
            windows=windows,
            batch_size=args.batch_size,
            model_module=model_module,
            torch_module=torch_module,
            device=device,
        )
        feature_scores, final_scores = calculate_scores(
            windows=windows,
            reconstructed=reconstructed,
            torch_module=torch_module,
        )
        results = build_results_frame(
            feature_scores=feature_scores,
            final_scores=final_scores,
            window_indices=window_indices,
            prepared_data=prepared_data,
            selected_features=config["selected_features"],
            timestamp_col=args.timestamp_col,
            has_timestamp=has_timestamp,
            pandas_module=pandas_module,
        )
        results.to_csv(output_path, index=False, float_format="%.12g")
        print_summary(
            final_scores=final_scores,
            output_path=output_path,
            numpy_module=numpy_module,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
