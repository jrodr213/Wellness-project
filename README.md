# Wearable Wellness Monitoring and Anomaly-Detection System

This project is an end-to-end experimental wellness monitoring workflow. It combines ESP32 firmware, live data collection, CSV logging, and a Python machine-learning pipeline for later anomaly-style analysis. It is an engineering prototype, not a medical device.

## Workflow Summary

The system works in four stages:

1. The ESP32 collects sensor measurements, conditions the signals, prints structured serial output, and can send JSON packets over Wi-Fi.
2. `Hardware_code/hardware_run.py` reads the ESP32 serial stream, follows the firmware `FIELDS`/`DATA` protocol, and saves live measurements to CSV.
3. The ML scripts in `ML_code/` train and run a BiLSTM autoencoder on collected CSV data.
4. Reconstruction scores are grouped with a Gaussian Mixture Model to explore patterns in the recorded data.

```text
ESP32 firmware
  -> serial DATA rows and optional HTTP JSON
  -> hardware_run.py CSV recording
  -> ML_code training and scoring
  -> grouped reconstruction-score outputs
```


## Firmware

Firmware source lives in `Hardware_code/src/main.cpp` and builds with PlatformIO. On startup it initializes the sensors, starts the continuous pulse-sampling task, prepares Wi-Fi if configured, and calibrates the capacitive touch baseline.

After startup, the capacitive foil acts as a start/stop control:

1. Wait for a confirmed touch to start a measurement session.
2. Require release so the same held touch cannot stop the session.
3. During the active session, collect and output one complete measurement every configured interval.
4. Stop the session on a second confirmed touch.
5. Require release before returning to the idle state.

The `temperature_f` field is the Kalman-filtered thermistor temperature used by downstream analysis. The firmware also appends `raw_temperature_f` for comparison.

Serial output remains machine-readable:

- `INFO,...` for normal status messages
- `WARNING,...` for recoverable configuration or calibration notices
- `ERROR,...` for failed operations
- `FIELDS,...` for the current CSV column order
- `DATA,...` for measurement rows

Wi-Fi credentials and the server endpoint are read from `Hardware_code/src/secrets.h`. A template is provided in `Hardware_code/src/secrets.example.h`, and the real secrets file is ignored by Git.

## Build And Run

Compile firmware:

```bash
cd Hardware_code
pio run
```

Upload firmware:

```bash
cd Hardware_code
pio run --target upload
```

Monitor firmware output:

```bash
cd Hardware_code
pio device monitor --baud 115200
```

Create Wi-Fi configuration:

```bash
cd Hardware_code/src
cp secrets.example.h secrets.h
```

Then edit `secrets.h` with your local Wi-Fi name, password, and server URL.

## Record Data

List available serial ports:

```bash
cd Hardware_code
../.venv/bin/python hardware_run.py --list-ports
```

Record CSV data from the ESP32:

```bash
cd Hardware_code
../.venv/bin/python hardware_run.py \
  --port /dev/cu.usbserial-0001 \
  --baud 115200 \
  --output data/wellness_data.csv
```

The logger waits for the firmware `FIELDS` line, parses matching `DATA` rows, ignores status lines, handles blank optional fields, flushes rows regularly, and preserves partial recordings on Ctrl-C.

## Analyze Data

Install Python dependencies:

```bash
cd ML_code
python3 -m pip install -r requirements.txt
```

Train the autoencoder:

```bash
cd ML_code
../.venv/bin/python run.py \
  --csv ../Hardware_code/data/wellness_data.csv \
  --features temperature_f bpm capacitive_filtered_us touch_status accel_x_g accel_y_g accel_z_g movement_intensity_g \
  --window-size 50 \
  --stride 5 \
  --batch-size 32 \
  --epochs 50 \
  --output final_bilstm_autoencoder.pt
```

Generate reconstruction scores:

```bash
cd ML_code
../.venv/bin/python scoring.py \
  --csv ../Hardware_code/data/wellness_data.csv \
  --model final_bilstm_autoencoder.pt \
  --output-csv reconstruction_scores.csv
```

Group the scores:

```bash
cd ML_code
../.venv/bin/python sorting.py \
  --input-csv reconstruction_scores.csv \
  --output-csv grouped_reconstruction_scores.csv \
  --score-column final_reconstruction_score
```

## Project Layout

```text
Hardware_code/
  platformio.ini
  hardware_run.py
  src/
    main.cpp
    secrets.example.h

ML_code/
  model.py
  run.py
  scoring.py
  sorting.py
  requirements.txt
```

## Notes

This repository is meant for experimental signal collection and ML exploration. The resulting scores and groups are mathematical analysis outputs, not medical conclusions.
