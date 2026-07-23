# Wearable Wellness Sensor BiLSTM Autoencoder

## 1. Project Overview

This project processes time-series data collected from a wearable wellness-monitoring system. The software works with chronological sensor readings stored in a CSV file and uses a PyTorch bidirectional LSTM autoencoder to learn patterns in overlapping windows of sensor data.

The project software:

1. loads chronological sensor data from a CSV file;
2. organizes it into overlapping time windows;
3. trains a PyTorch bidirectional LSTM autoencoder;
4. reconstructs each sensor window;
5. calculates reconstruction-error scores;
6. groups similar scores using a Gaussian Mixture Model.

Higher reconstruction error means the model reconstructed a window less accurately. A higher error may indicate that the window is different from patterns the model learned, but it does not by itself identify a medical condition.

## 2. Sensor Data

The code supports CSV files with user-selected numeric sensor columns. Possible columns may include:

- `timestamp`
- `heart_rate`
- `temperature`
- `accel_x`
- `accel_y`
- `accel_z`
- `capacitive`
- `state`
- `label`

Not every column is required. Each row represents one sampling time. Selected sensor features must be numeric. Rows should be arranged chronologically, although the scripts sort by the timestamp column when that column exists. The exact sensor columns used by the model depend on the supplied CSV and the `--features` command-line argument.

Small example CSV:

```csv
timestamp,heart_rate,temperature,accel_x,accel_y,accel_z,capacitive,state,label
2026-01-01T09:00:00,72,36.7,0.01,-0.03,0.98,214,resting,
2026-01-01T09:00:01,73,36.7,0.02,-0.02,1.01,216,resting,
2026-01-01T09:00:02,75,36.8,0.05,-0.01,1.04,219,walking,
2026-01-01T09:00:03,78,36.8,0.08,0.01,1.08,223,walking,
```

## 3. Processing Pipeline

```text
Sensor CSV
    ↓
Data cleaning and chronological ordering
    ↓
Overlapping time windows
    ↓
BiLSTM autoencoder training
    ↓
Window reconstruction
    ↓
Reconstruction-error scoring
    ↓
Gaussian Mixture Model grouping
    ↓
Grouped CSV results and plots
```

`Sensor CSV`: The input file contains rows of timestamped sensor readings.

`Data cleaning and chronological ordering`: The scripts sort by the timestamp column when present, replace positive and negative infinity with missing values, interpolate missing selected sensor readings, and remove rows that still cannot be used.

`Overlapping time windows`: The cleaned sensor readings are divided into fixed-length chronological windows. With the defaults, each window contains 50 rows and the next window starts 5 rows later.

`BiLSTM autoencoder training`: The model learns to reconstruct the same sensor windows it receives as input.

`Window reconstruction`: After training, the model recreates each input window from its compressed latent representation.

`Reconstruction-error scoring`: The scoring script compares each original window to its reconstruction using squared error.

`Gaussian Mixture Model grouping`: The sorting script groups windows based only on the final reconstruction score.

`Grouped CSV results and plots`: The final outputs include grouped score CSV files, model-selection summaries, group summaries, and plots.

## 4. Project Files

### `model.py`

`model.py` defines the reusable PyTorch model and helper functions. Its main model class is `BiLSTMAutoencoder`.

The architecture contains:

- a bidirectional LSTM encoder;
- forward and backward processing over each input sequence;
- a combined hidden representation from the final forward and backward encoder states;
- a linear bottleneck layer that creates a compressed latent vector;
- a decoder LSTM that receives the latent vector repeated across the sequence length;
- a final linear output layer that reconstructs the original number of sensor features.

`model.py` can also train the model directly and save outputs under `model_outputs/` by default. The newer project workflow uses `run.py` for training.

### `run.py`

`run.py` is the main training script. It:

- loads the training CSV;
- selects features from `--features`, or automatically selects numeric columns when `--features` is not supplied;
- excludes `timestamp`, `state`, `label`, `cluster`, `anomaly`, and `is_anomaly` from automatic feature selection;
- validates that selected features exist and are numeric;
- sorts data by the timestamp column when it exists;
- replaces infinite values, interpolates missing selected sensor values, and removes rows that remain unusable;
- creates chronological overlapping windows;
- splits windows chronologically into training and validation sets;
- trains the autoencoder using Adam and mean squared reconstruction loss;
- uses early stopping based on validation loss;
- saves the best model checkpoint to the path supplied by `--output`.

Training uses each input window as its own reconstruction target. The training loss is PyTorch `MSELoss`, which averages squared differences between reconstructed values and original values.

### `scoring.py`

`scoring.py` loads a trained checkpoint and calculates reconstruction scores for all available windows in a CSV. It:

- loads the trained `.pt` checkpoint;
- reads the selected feature order, input feature count, window size, stride, hidden size, latent size, dropout, number of LSTM layers, and model weights from the checkpoint;
- recreates windows using the saved feature order and configuration;
- rebuilds the `BiLSTMAutoencoder` from `model.py`;
- runs the model in evaluation mode with gradients disabled;
- reconstructs every window without changing model weights;
- calculates feature-level and final reconstruction scores;
- saves the scores to a CSV.

The scoring formula in `scoring.py` is:

```text
squared_error = (original - reconstructed) ** 2
feature_score_for_feature_j = mean(squared_error over all timestamps for feature j)
final_reconstruction_score = mean(squared_error over all timestamps and all selected features)
```

In plain English, the script squares the difference between every original sensor value and reconstructed sensor value. It then averages those squared differences across time for each feature, and across both time and features for the final window score.

Training loss and scoring are closely related but used at different times:

- During training, `run.py` uses `MSELoss` to update model weights.
- After training, `scoring.py` calculates squared reconstruction scores with no weight updates.

`scoring.py` does not square the final MSE again.

### `sorting.py`

`sorting.py` reads the reconstruction-score CSV produced by `scoring.py` and groups windows using a Gaussian Mixture Model. It:

- reads the score CSV;
- validates the requested score column;
- keeps the original reconstruction scores unchanged;
- creates `log_reconstruction_score = np.log1p(original_score)` for fitting;
- fits candidate Gaussian Mixture Models from `--min-components` through `--max-components`;
- computes BIC and AIC for every candidate model;
- selects the model with the lowest BIC;
- assigns raw GMM labels and membership probabilities;
- reorders groups so group `0` has the lowest average original reconstruction score, group `1` has the next-lowest average, and so on;
- saves grouped results, model-selection data, group summaries, and plots.

The GMM groups are mathematical score groups. They are not medical classifications.

### Other Project Files

| File | Purpose |
| --- | --- |
| `README.md` | Project documentation. |
| `requirements.txt` | Third-party Python dependencies needed by the scripts. |

No sample CSV files or generated output directories were present when this README was written.

## 5. Model Architecture

The main training workflow uses `run.py`, which creates `BiLSTMAutoencoder` from `model.py` with one LSTM layer.

| Setting | Current Default |
| --- | --- |
| Input shape | `[batch_size, sequence_length, number_of_sensor_features]` |
| Window size / sequence length | `50` |
| Stride | `5` |
| Number of selected features | Depends on `--features` or automatic numeric-column selection |
| Hidden size | `64` |
| Latent size | `16` |
| Number of LSTM layers | `1` |
| Bidirectional setting | Encoder is bidirectional; decoder is not bidirectional |
| Dropout | Argument default is `0.2`, but actual LSTM dropout is `0.0` when there is only one LSTM layer |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Batch size | `32` |
| Maximum epochs | `50` |
| Early-stopping patience | `8` |
| Training ratio | `0.8` |
| Validation ratio | `0.2` |

The model input shape is:

```text
[batch size, sequence length, number of sensor features]
```

`Sequence length` is the number of chronological rows in one window. With the default window size, each input window contains 50 rows.

`Batch size` is the number of windows processed before a training update. With the default batch size, the model processes up to 32 windows per training step.

`Number of sensor features` is the number of selected numeric sensor columns, such as `heart_rate`, `temperature`, and accelerometer columns.

Bidirectional processing means the encoder has one LSTM direction that reads the sequence forward and another that reads the sequence backward. The code combines the final forward and backward hidden states before passing them through the latent bottleneck.

The code accepts a dropout value, but PyTorch LSTM dropout is only applied between stacked LSTM layers. Because the current training script uses one LSTM layer, the model sets actual LSTM dropout to `0.0`.

## 6. Training Process

Training starts by reading the CSV and selecting numeric sensor features. The selected readings are sorted chronologically when the timestamp column exists. The script then creates overlapping windows, where each window is a small chronological sequence.

The windows are split chronologically: the first portion is used for training and the final portion is used for validation. The validation windows are not randomly mixed into the training portion.

For each training batch:

1. The input window is passed through the BiLSTM encoder.
2. The encoder output is compressed into a latent vector.
3. The decoder reconstructs the full window.
4. The reconstructed window is compared with the original input window.
5. `MSELoss` calculates the mean squared reconstruction error.
6. Backpropagation computes gradients.
7. Adam updates the model weights.

Adam is an optimizer that uses gradient history and adaptive update sizes to adjust model parameters during training.

At the end of each epoch, validation loss is calculated without weight updates. Early stopping tracks validation loss. When validation loss improves, `run.py` saves a new best checkpoint. If validation loss does not improve for the configured patience value, training stops early.

## 7. Requirements

The project imports these third-party libraries:

- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`

Install them with:

```bash
python3 -m pip install -r requirements.txt
```

If using the project virtual environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 8. Expected CSV Format

Example CSV:

```csv
timestamp,heart_rate,temperature,accel_x,accel_y,accel_z,capacitive,state,label
2026-01-01T09:00:00,72,36.7,0.01,-0.03,0.98,214,resting,
2026-01-01T09:00:01,73,36.7,0.02,-0.02,1.01,216,resting,
2026-01-01T09:00:02,75,36.8,0.05,-0.01,1.04,219,walking,
2026-01-01T09:00:03,78,36.8,0.08,0.01,1.08,223,walking,
```

The timestamp column defaults to `timestamp`. You can change it with `--timestamp-col`.

Feature selection works in two ways:

- Provide features manually with `--features heart_rate temperature accel_x accel_y accel_z capacitive`.
- Omit `--features` and let `run.py` select numeric columns automatically while excluding metadata columns.

Selected feature columns must be numeric. The scripts replace positive and negative infinity with missing values, interpolate missing selected sensor values in chronological order, and remove rows that still contain unusable selected values.

The number of usable rows must be at least the window size. In practice, at least two generated windows are needed for training because `run.py` requires both a training set and validation set.

The project does not normalize or standardize sensor values. Scoring uses the same unscaled feature format used during training.

## 9. How to Train the Model

Use `run.py` to train the model:

```bash
python3 run.py \
  --csv wellness_data.csv \
  --features heart_rate temperature accel_x accel_y accel_z capacitive \
  --window-size 50 \
  --stride 5 \
  --batch-size 32 \
  --epochs 50 \
  --output final_bilstm_autoencoder.pt
```

Important options:

| Option | Meaning |
| --- | --- |
| `--csv` | Path to the training CSV file. |
| `--features` | Numeric sensor columns to use. If omitted, numeric columns are selected automatically. |
| `--timestamp-col` | Timestamp column used for chronological sorting. Default: `timestamp`. |
| `--window-size` | Number of rows per window. Default: `50`. |
| `--stride` | Number of rows to move forward between windows. Default: `5`. |
| `--train-ratio` | Chronological fraction of windows used for training. Default: `0.8`. |
| `--batch-size` | Number of windows per training batch. Default: `32`. |
| `--epochs` | Maximum number of training epochs. Default: `50`. |
| `--learning-rate` | Adam learning rate. Default: `0.001`. |
| `--hidden-size` | LSTM hidden size. Default: `64`. |
| `--latent-size` | Bottleneck latent size. Default: `16`. |
| `--dropout` | Dropout argument passed to the model. With one LSTM layer, actual LSTM dropout is `0.0`. |
| `--patience` | Early-stopping patience. Default: `8`. |
| `--output` | Path for the best checkpoint. Default: `final_bilstm_autoencoder.pt`. |

## 10. How to Calculate Reconstruction Scores

Use `scoring.py` after training:

```bash
python3 scoring.py \
  --csv wellness_data.csv \
  --model final_bilstm_autoencoder.pt \
  --output-csv reconstruction_scores.csv
```

`scoring.py` creates a CSV with:

- `window_number`
- `window_start_row`
- `window_end_row`
- `window_start_timestamp`, when the timestamp column exists
- `window_end_timestamp`, when the timestamp column exists
- one feature-level MSE column for every selected feature, such as `heart_rate_mse`
- `final_reconstruction_score`

The final reconstruction-score formula is:

```text
final_reconstruction_score =
    mean((original - reconstructed) ** 2 over all timestamps and all selected features)
```

This is the mean squared reconstruction error for the complete window. It is not squared again.

## 11. How to Group Reconstruction Scores

Use `sorting.py` after creating `reconstruction_scores.csv`:

```bash
python3 sorting.py \
  --input-csv reconstruction_scores.csv \
  --output-csv grouped_reconstruction_scores.csv \
  --score-column final_reconstruction_score
```

`sorting.py` keeps all original columns and adds:

- `score_group`
- `group_confidence`
- `log_reconstruction_score`
- `probability_group_0`
- `probability_group_1`
- additional `probability_group_N` columns when more groups are selected

The groups are reordered by the average original reconstruction score. Group `0` has the lowest average original reconstruction score, group `1` has the next-lowest average, and so on.

## 12. Output Files

The scripts can create the following important outputs:

| Output File | Created By | Contents |
| --- | --- | --- |
| `final_bilstm_autoencoder.pt` | `run.py` | Best model checkpoint saved during training. The path is configurable with `--output`. |
| `reconstruction_scores.csv` | `scoring.py` | Chronological per-window reconstruction scores. The path is configurable with `--output-csv`. |
| `grouped_reconstruction_scores.csv` | `sorting.py` | Original score CSV plus GMM group assignments and group probabilities. The path is configurable with `--output-csv`. |
| `gmm_model_selection.csv` | `sorting.py` | BIC/AIC table for candidate GMM component counts. Saved beside the grouped output CSV. |
| `gmm_group_summary.csv` | `sorting.py` | One-row-per-group summary statistics. Saved beside the grouped output CSV. |
| `gmm_score_groups.png` | `sorting.py` | Histogram of `log1p` reconstruction scores with fitted Gaussian component curves. The path is configurable with `--plot`. |
| `reconstruction_groups_over_time.png` | `sorting.py` | Chronological scatter plot of reconstruction scores colored by group. Saved beside the grouped output CSV. |

`model.py` can also be run directly. When used directly, it saves these files in `model_outputs/` by default:

| Output File | Contents |
| --- | --- |
| `best_bilstm_autoencoder.pt` | Best checkpoint from `model.py` training. |
| `training_history.csv` | Epoch-level training and validation loss history. |
| `training_loss.png` | Training/validation loss plot. |
| `model_config.json` | Selected feature names, window size, model configuration, and best validation loss. |
| `original_windows.npy` | NumPy array of original windows. |
| `reconstructed_windows.npy` | NumPy array of reconstructed windows. |
| `reconstruction_errors.csv` | Per-window numerical reconstruction errors from `model.py`. |

`run.py` does not save `training_history.csv` or `training_loss.png`; it stores training and validation loss history inside the checkpoint.

## 13. Device Selection

The code selects the compute device in this order:

1. CUDA for compatible NVIDIA GPUs;
2. Apple Silicon MPS when available;
3. CPU as the fallback.

`run.py` and `scoring.py` print the selected device when they run. `sorting.py` does not use PyTorch model computation and does not select a CUDA/MPS/CPU device.

## 14. Example Project Workflow

1. Prepare the sensor CSV.

2. Train the model:

```bash
python3 run.py \
  --csv wellness_data.csv \
  --features heart_rate temperature accel_x accel_y accel_z capacitive \
  --window-size 50 \
  --stride 5 \
  --batch-size 32 \
  --epochs 50 \
  --output final_bilstm_autoencoder.pt
```

3. Score all windows:

```bash
python3 scoring.py \
  --csv wellness_data.csv \
  --model final_bilstm_autoencoder.pt \
  --output-csv reconstruction_scores.csv
```

4. Group the scores:

```bash
python3 sorting.py \
  --input-csv reconstruction_scores.csv \
  --output-csv grouped_reconstruction_scores.csv \
  --score-column final_reconstruction_score
```

5. Review the output CSV files and plots.

## 15. Interpretation of Results

Low reconstruction scores indicate the model reconstructed those patterns more accurately.

High reconstruction scores indicate greater mismatch with patterns learned during training. A high score can result from unusual activity, noise, sensor movement, sensor failure, poor training coverage, or a genuinely different pattern.

GMM probabilities describe mathematical group membership based on reconstruction scores. Score groups require external labels or observations before being assigned behavioral meanings.

## 16. Limitations

- Results depend on training-data quality and coverage.
- Overlapping windows are not fully independent.
- A BiLSTM requires the complete window before producing its representation.
- The project does not normalize sensor values, so sensors with different numerical ranges can influence reconstruction loss differently.
- High reconstruction error is not automatically a health warning.
- The Gaussian Mixture Model assumes the score distribution can be represented as a mixture of Gaussian components after the `log1p` transform.
- Small datasets may cause overfitting or unstable grouping.

## 17. Reproducibility

The scripts include several reproducibility features:

- `run.py` uses random seeds for Python, NumPy, and PyTorch.
- `scoring.py` also sets random seeds, although it does not update model weights.
- `sorting.py` uses a configurable random state for Gaussian Mixture Model fitting.
- The trained checkpoint stores selected feature order.
- The trained checkpoint stores model configuration, including window size, stride, hidden size, latent size, dropout, number of LSTM layers, batch size, and learning rate.
- The train/validation split is chronological.
- The checkpoint stores the best epoch, best validation loss, training-loss history, validation-loss history, optimizer state, model state dictionary, and model class name.

## 18. Project Structure

Actual current project files and recommended locations:

```text
Wearable Wellness Sensor BiLSTM Autoencoder/
├── README.md
├── requirements.txt
├── model.py
├── run.py
├── scoring.py
├── sorting.py
├── data/
│   └── wellness_data.csv
├── models/
│   └── final_bilstm_autoencoder.pt
└── outputs/
    ├── reconstruction_scores.csv
    ├── grouped_reconstruction_scores.csv
    ├── gmm_model_selection.csv
    ├── gmm_group_summary.csv
    ├── gmm_score_groups.png
    └── reconstruction_groups_over_time.png
```

The `data/`, `models/`, and `outputs/` folders are recommended organization locations. They were not present in the current project folder when this README was written.

## 19. Disclaimer

This project is an experimental engineering and machine-learning system. Its reconstruction scores and groups are not medical diagnoses and should not be used as a substitute for professional medical evaluation.
