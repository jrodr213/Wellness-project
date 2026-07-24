"""Group reconstruction-score windows with a Gaussian Mixture Model.

Example:
    python3 sorting.py \
      --input-csv reconstruction_scores.csv \
      --output-csv grouped_reconstruction_scores.csv \
      --score-column final_reconstruction_score

This script reads the CSV created by scoring.py, fits Gaussian Mixture Models
only on log-transformed reconstruction scores, and writes grouped numerical
results. It does not run the autoencoder, load model checkpoints, retrain
weights, use thresholds, or assign medical/health labels.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_SCORE_COLUMN = "final_reconstruction_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group reconstruction scores using a Gaussian Mixture Model."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the reconstruction-score CSV produced by scoring.py.",
    )
    parser.add_argument(
        "--output-csv",
        default="grouped_reconstruction_scores.csv",
        help="Path where grouped reconstruction scores will be saved.",
    )
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument(
        "--plot",
        default="gmm_score_groups.png",
        help="Path where the GMM density plot will be saved.",
    )
    return parser.parse_args()


def import_required_modules() -> dict[str, Any]:
    if not Path("scoring.py").exists():
        raise FileNotFoundError(
            "scoring.py does not exist in the current directory. Run sorting.py from "
            "the folder that contains scoring.py."
        )

    modules: dict[str, Any] = {}
    for module_name in ("numpy", "pandas"):
        try:
            modules[module_name] = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Required package '{module_name}' is not installed. Activate the "
                "project virtual environment or install the required packages."
            ) from exc

    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        modules["pyplot"] = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(
            "Required package 'matplotlib' is not installed. Activate the project "
            "virtual environment or install matplotlib."
        ) from exc

    try:
        sklearn_mixture = importlib.import_module("sklearn.mixture")
        modules["GaussianMixture"] = sklearn_mixture.GaussianMixture
    except ImportError as exc:
        raise ImportError(
            "Required package 'scikit-learn' is not installed. Install it with "
            "'python3 -m pip install scikit-learn' or use the project virtual "
            "environment after installing it there."
        ) from exc

    return modules


def set_random_state(seed: int, numpy_module: ModuleType) -> None:
    random.seed(seed)
    numpy_module.random.seed(seed)


def validate_args(args: argparse.Namespace) -> None:
    if args.min_components <= 0:
        raise ValueError("--min-components must be greater than 0.")
    if args.max_components < args.min_components:
        raise ValueError("--max-components must be >= --min-components.")
    if args.n_init <= 0:
        raise ValueError("--n-init must be greater than 0.")


def ensure_output_parent(path: Path) -> None:
    parent = path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if not parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {parent}")


def validate_paths(input_path: Path, output_path: Path, plot_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input CSV path is not a file: {input_path}")
    ensure_output_parent(output_path)
    ensure_output_parent(plot_path)
    ensure_output_parent(output_path.parent / "gmm_model_selection.csv")
    ensure_output_parent(output_path.parent / "gmm_group_summary.csv")
    ensure_output_parent(output_path.parent / "reconstruction_groups_over_time.png")


def load_scores(input_path: Path, pandas_module: ModuleType) -> Any:
    data = pandas_module.read_csv(input_path)
    if data.empty:
        raise ValueError("Input CSV is empty.")
    return data


def read_scoring_py_score_hints() -> list[str]:
    scoring_path = Path("scoring.py")
    if not scoring_path.exists():
        return []
    text = scoring_path.read_text(encoding="utf-8", errors="ignore")
    hints: list[str] = []
    for candidate in (
        "final_reconstruction_score",
        "overall_mean_reconstruction_error",
        "reconstruction_score",
    ):
        if candidate in text:
            hints.append(candidate)
    return hints


def likely_score_columns(columns: list[str]) -> list[str]:
    likely_terms = ("score", "mse", "reconstruction", "error")
    return [
        column
        for column in columns
        if any(term in column.lower() for term in likely_terms)
    ]


def validate_score_column(
    data: Any,
    score_column: str,
    pandas_module: ModuleType,
    numpy_module: ModuleType,
) -> Any:
    if score_column not in data.columns:
        hints = sorted(set(read_scoring_py_score_hints() + likely_score_columns(list(data.columns))))
        if score_column == DEFAULT_SCORE_COLUMN:
            raise ValueError(
                f"Score column '{score_column}' is missing. Likely reconstruction-score "
                f"columns from scoring.py or the CSV are: {hints or 'none found'}."
            )
        raise ValueError(
            f"Score column '{score_column}' is missing. Available columns are: "
            f"{list(data.columns)}. Likely reconstruction-score columns are: "
            f"{hints or 'none found'}."
        )

    numeric_scores = pandas_module.to_numeric(data[score_column], errors="coerce")
    if numeric_scores.isna().any() and data[score_column].notna().any():
        raise ValueError(f"Score column contains nonnumeric data: {score_column}")
    if numeric_scores.isna().any():
        raise ValueError(f"Scores contain NaN values in column: {score_column}")

    scores = numeric_scores.to_numpy(dtype="float64")
    if not numpy_module.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinity.")
    if (scores < 0).any():
        raise ValueError("Scores contain negative values.")
    if numpy_module.all(scores == scores[0]):
        raise ValueError("All scores are identical; GMM grouping is not meaningful.")

    return scores


def validate_component_counts(
    number_of_windows: int,
    min_components: int,
    max_components: int,
) -> None:
    if number_of_windows < min_components:
        raise ValueError(
            "Too few windows for the requested component counts: "
            f"{number_of_windows} windows for minimum {min_components} components."
        )
    if number_of_windows < max_components:
        raise ValueError(
            "Too few windows for the requested component counts: "
            f"{number_of_windows} windows for maximum {max_components} components."
        )


def print_fit_inputs(
    input_path: Path,
    output_path: Path,
    score_column: str,
    scores: Any,
    min_components: int,
    max_components: int,
    numpy_module: ModuleType,
) -> None:
    print("GMM grouping inputs")
    print(f"Input CSV path: {input_path}")
    print(f"Output CSV path: {output_path}")
    print(f"Selected score column: {score_column}")
    print(f"Number of windows: {len(scores)}")
    print(f"Minimum score: {float(numpy_module.min(scores)):.12g}")
    print(f"Mean score: {float(numpy_module.mean(scores)):.12g}")
    print(f"Median score: {float(numpy_module.median(scores)):.12g}")
    print(f"Maximum score: {float(numpy_module.max(scores)):.12g}")
    print(f"Candidate component range: {min_components}-{max_components}")


def fit_candidate_models(
    x_values: Any,
    min_components: int,
    max_components: int,
    random_state: int,
    n_init: int,
    gaussian_mixture_class: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected_model: Any | None = None
    selected_bic = float("inf")

    for components in range(min_components, max_components + 1):
        try:
            model = gaussian_mixture_class(
                n_components=components,
                covariance_type="full",
                random_state=random_state,
                n_init=n_init,
                reg_covar=1e-6,
            )
            model.fit(x_values)
            bic = float(model.bic(x_values))
            aic = float(model.aic(x_values))
        except Exception as exc:
            raise ValueError(
                f"GMM fitting fails for {components} groups: {exc}"
            ) from exc

        if bic < selected_bic:
            selected_bic = bic
            selected_model = model

        rows.append(
            {
                "number_of_groups": components,
                "bic": bic,
                "aic": aic,
                "selected": False,
            }
        )

    if selected_model is None:
        raise ValueError("GMM fitting fails: no candidate model was selected.")

    selected_components = int(selected_model.n_components)
    for row in rows:
        row["selected"] = row["number_of_groups"] == selected_components
    return selected_model, rows


def print_model_selection(selection_rows: list[dict[str, Any]]) -> None:
    print("\nModel-selection results")
    for row in selection_rows:
        print(
            f"Groups: {row['number_of_groups']} | BIC: {row['bic']:.6f} | "
            f"AIC: {row['aic']:.6f}"
        )
    selected = next(row for row in selection_rows if row["selected"])
    print(f"Selected number of groups by lowest BIC: {selected['number_of_groups']}")


def create_order_mapping(raw_labels: Any, scores: Any, n_components: int, numpy_module: ModuleType) -> dict[int, int]:
    raw_means: list[tuple[int, float]] = []
    for raw_group in range(n_components):
        group_scores = scores[raw_labels == raw_group]
        mean_score = (
            float(numpy_module.mean(group_scores))
            if len(group_scores) > 0
            else float("inf")
        )
        raw_means.append((raw_group, mean_score))

    raw_means.sort(key=lambda item: item[1])
    return {raw_group: ordered_group for ordered_group, (raw_group, _) in enumerate(raw_means)}


def apply_ordering(
    raw_labels: Any,
    raw_probabilities: Any,
    scores: Any,
    numpy_module: ModuleType,
) -> tuple[Any, Any, Any, dict[int, int]]:
    n_components = raw_probabilities.shape[1]
    mapping = create_order_mapping(
        raw_labels=raw_labels,
        scores=scores,
        n_components=n_components,
        numpy_module=numpy_module,
    )
    ordered_labels = numpy_module.array([mapping[int(label)] for label in raw_labels], dtype=int)
    ordered_probabilities = numpy_module.zeros_like(raw_probabilities)
    for raw_group, ordered_group in mapping.items():
        ordered_probabilities[:, ordered_group] = raw_probabilities[:, raw_group]

    probability_sums = ordered_probabilities.sum(axis=1)
    if not numpy_module.allclose(probability_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Probability rows do not sum to approximately 1.")

    group_confidence = ordered_probabilities[
        numpy_module.arange(len(ordered_labels)),
        ordered_labels,
    ]
    return ordered_labels, ordered_probabilities, group_confidence, mapping


def build_output_frame(
    data: Any,
    score_column: str,
    log_scores: Any,
    ordered_labels: Any,
    ordered_probabilities: Any,
    group_confidence: Any,
) -> Any:
    output = data.copy()
    output["_original_reconstruction_score_copy"] = output[score_column]
    output["score_group"] = ordered_labels.astype(int)
    output["group_confidence"] = group_confidence
    output["log_reconstruction_score"] = log_scores
    for group_index in range(ordered_probabilities.shape[1]):
        output[f"probability_group_{group_index}"] = ordered_probabilities[:, group_index]
    output = output.drop(columns=["_original_reconstruction_score_copy"])
    return output


def build_selection_frame(
    selection_rows: list[dict[str, Any]],
    pandas_module: ModuleType,
) -> Any:
    return pandas_module.DataFrame(selection_rows)


def covariance_variance(model: Any, raw_group: int) -> float:
    covariance = model.covariances_[raw_group]
    return float(covariance.reshape(-1)[0])


def build_group_summary(
    output: Any,
    score_column: str,
    mapping: dict[int, int],
    selected_model: Any,
    pandas_module: ModuleType,
) -> Any:
    inverse_mapping = {ordered: raw for raw, ordered in mapping.items()}
    total_windows = len(output)
    rows: list[dict[str, Any]] = []

    for ordered_group in sorted(output["score_group"].unique()):
        group_rows = output[output["score_group"] == ordered_group]
        scores = group_rows[score_column]
        raw_group = inverse_mapping[int(ordered_group)]
        rows.append(
            {
                "score_group": int(ordered_group),
                "number_of_windows": int(len(group_rows)),
                "percentage_of_windows": float(len(group_rows) / total_windows * 100.0),
                "minimum_score": float(scores.min()),
                "mean_score": float(scores.mean()),
                "median_score": float(scores.median()),
                "maximum_score": float(scores.max()),
                "standard_deviation": float(scores.std(ddof=0)),
                "mean_group_confidence": float(group_rows["group_confidence"].mean()),
                "gaussian_mean_log_space": float(selected_model.means_[raw_group, 0]),
                "gaussian_variance_log_space": covariance_variance(
                    selected_model,
                    raw_group,
                ),
                "gaussian_mixture_weight": float(selected_model.weights_[raw_group]),
            }
        )

    return pandas_module.DataFrame(rows)


def print_group_summary(summary: Any) -> None:
    print("\nFinal group summary")
    for _, row in summary.iterrows():
        print(
            f"Group {int(row['score_group'])}: "
            f"{int(row['number_of_windows'])} windows "
            f"({row['percentage_of_windows']:.2f}%) | "
            f"avg score {row['mean_score']:.12g} | "
            f"min {row['minimum_score']:.12g} | "
            f"max {row['maximum_score']:.12g} | "
            f"avg confidence {row['mean_group_confidence']:.4f}"
        )


def print_low_confidence_warning(group_confidence: Any, threshold: float = 0.60) -> None:
    low_count = int((group_confidence < threshold).sum())
    if low_count > 0:
        print(
            f"Warning: {low_count} windows have maximum group probability below "
            f"{threshold:.2f}. Assigned groups were not changed."
        )
    else:
        print(f"No windows have maximum group probability below {threshold:.2f}.")


def plot_gmm_density(
    log_scores: Any,
    selected_model: Any,
    mapping: dict[int, int],
    plot_path: Path,
    numpy_module: ModuleType,
    pyplot: Any,
) -> None:
    x_grid = numpy_module.linspace(log_scores.min(), log_scores.max(), 600).reshape(-1, 1)
    total_density = numpy_module.exp(selected_model.score_samples(x_grid))

    pyplot.figure(figsize=(9, 5))
    pyplot.hist(log_scores, bins="auto", density=True, alpha=0.35, label="Window scores")

    for raw_group, ordered_group in sorted(mapping.items(), key=lambda item: item[1]):
        mean = selected_model.means_[raw_group, 0]
        variance = covariance_variance(selected_model, raw_group)
        weight = selected_model.weights_[raw_group]
        component_density = (
            weight
            * (1.0 / numpy_module.sqrt(2.0 * numpy_module.pi * variance))
            * numpy_module.exp(-0.5 * ((x_grid[:, 0] - mean) ** 2) / variance)
        )
        pyplot.plot(x_grid[:, 0], component_density, label=f"Ordered group {ordered_group}")

    pyplot.plot(x_grid[:, 0], total_density, linestyle="--", label="Total mixture")
    pyplot.title("Gaussian Mixture Groups for log1p Reconstruction Scores")
    pyplot.xlabel("log1p reconstruction score")
    pyplot.ylabel("Density")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig(plot_path, dpi=150)
    pyplot.close()


def should_use_log_axis(scores: Any, numpy_module: ModuleType) -> bool:
    positive_scores = scores[scores > 0]
    if len(positive_scores) == 0:
        return False
    return float(numpy_module.max(positive_scores) / numpy_module.min(positive_scores)) > 100.0


def plot_groups_over_time(
    scores: Any,
    ordered_labels: Any,
    output_path: Path,
    numpy_module: ModuleType,
    pyplot: Any,
) -> None:
    positions = numpy_module.arange(len(scores))
    pyplot.figure(figsize=(10, 5))
    for group in sorted(set(int(label) for label in ordered_labels)):
        mask = ordered_labels == group
        pyplot.scatter(positions[mask], scores[mask], s=18, label=f"Group {group}")

    if should_use_log_axis(scores, numpy_module):
        pyplot.yscale("log")
    pyplot.title("Reconstruction Groups Over Time")
    pyplot.xlabel("Chronological window position")
    pyplot.ylabel("Original reconstruction score")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.savefig(output_path, dpi=150)
    pyplot.close()


def main() -> None:
    args = parse_args()

    try:
        validate_args(args)
        modules = import_required_modules()
        numpy_module = modules["numpy"]
        pandas_module = modules["pandas"]
        pyplot = modules["pyplot"]
        gaussian_mixture_class = modules["GaussianMixture"]

        set_random_state(seed=args.random_state, numpy_module=numpy_module)

        input_path = Path(args.input_csv)
        output_path = Path(args.output_csv)
        plot_path = Path(args.plot)
        selection_path = output_path.parent / "gmm_model_selection.csv"
        summary_path = output_path.parent / "gmm_group_summary.csv"
        over_time_plot_path = output_path.parent / "reconstruction_groups_over_time.png"

        validate_paths(
            input_path=input_path,
            output_path=output_path,
            plot_path=plot_path,
        )

        data = load_scores(input_path=input_path, pandas_module=pandas_module)
        scores = validate_score_column(
            data=data,
            score_column=args.score_column,
            pandas_module=pandas_module,
            numpy_module=numpy_module,
        )
        validate_component_counts(
            number_of_windows=len(scores),
            min_components=args.min_components,
            max_components=args.max_components,
        )

        # log1p compresses extremely large scores and can make strongly
        # right-skewed reconstruction scores easier for the Gaussian Mixture
        # Model to separate while preserving score order.
        log_scores = numpy_module.log1p(scores)
        x_values = log_scores.reshape(-1, 1)

        print_fit_inputs(
            input_path=input_path,
            output_path=output_path,
            score_column=args.score_column,
            scores=scores,
            min_components=args.min_components,
            max_components=args.max_components,
            numpy_module=numpy_module,
        )

        selected_model, selection_rows = fit_candidate_models(
            x_values=x_values,
            min_components=args.min_components,
            max_components=args.max_components,
            random_state=args.random_state,
            n_init=args.n_init,
            gaussian_mixture_class=gaussian_mixture_class,
        )
        print_model_selection(selection_rows)

        raw_labels = selected_model.predict(x_values)
        raw_probabilities = selected_model.predict_proba(x_values)
        ordered_labels, ordered_probabilities, group_confidence, mapping = apply_ordering(
            raw_labels=raw_labels,
            raw_probabilities=raw_probabilities,
            scores=scores,
            numpy_module=numpy_module,
        )

        output = build_output_frame(
            data=data,
            score_column=args.score_column,
            log_scores=log_scores,
            ordered_labels=ordered_labels,
            ordered_probabilities=ordered_probabilities,
            group_confidence=group_confidence,
        )
        selection_frame = build_selection_frame(
            selection_rows=selection_rows,
            pandas_module=pandas_module,
        )
        summary = build_group_summary(
            output=output,
            score_column=args.score_column,
            mapping=mapping,
            selected_model=selected_model,
            pandas_module=pandas_module,
        )

        output.to_csv(output_path, index=False, float_format="%.12g")
        selection_frame.to_csv(selection_path, index=False, float_format="%.12g")
        summary.to_csv(summary_path, index=False, float_format="%.12g")

        plot_gmm_density(
            log_scores=log_scores,
            selected_model=selected_model,
            mapping=mapping,
            plot_path=plot_path,
            numpy_module=numpy_module,
            pyplot=pyplot,
        )
        plot_groups_over_time(
            scores=scores,
            ordered_labels=ordered_labels,
            output_path=over_time_plot_path,
            numpy_module=numpy_module,
            pyplot=pyplot,
        )

        print_group_summary(summary)
        print_low_confidence_warning(group_confidence)
        print("\nSaved files")
        print(f"Grouped output CSV: {output_path}")
        print(f"Model-selection CSV: {selection_path}")
        print(f"Group summary CSV: {summary_path}")
        print(f"GMM density plot: {plot_path}")
        print(f"Groups-over-time plot: {over_time_plot_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
