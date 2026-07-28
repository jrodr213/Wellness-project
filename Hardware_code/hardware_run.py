"""Live serial CSV collector for the wearable wellness ESP32 firmware.

Example:
    python3 hardware_run.py \
      --port /dev/cu.usbserial-0001 \
      --baud 115200 \
      --output data/wellness_data.csv

The ESP32 firmware runs continuously. This script starts saving rows only after
the first distinct capacitive touch and stops before saving the second distinct
touch. The firmware emits calibrated/processed values; this script records
those values as received and does not simulate sensor data.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TextIO


BOOT_PREFIXES = (
    "rst:",
    "ets ",
    "ESP-ROM:",
    "load:",
    "entry ",
    "configsip:",
)

TEXT_FIELDS = {"movement_status"}


class CollectorState(Enum):
    CALIBRATING = "CALIBRATING"
    WAITING_FOR_START = "WAITING_FOR_START"
    RECORDING = "RECORDING"
    FINISHED = "FINISHED"


@dataclass
class Thresholds:
    baseline_median: float
    baseline_mad: float
    touch_threshold: float
    release_threshold: float


@dataclass
class RunStats:
    samples_saved: int = 0
    malformed_rows: int = 0
    first_esp32_timestamp: str | None = None
    final_esp32_timestamp: str | None = None
    recording_started_monotonic: float | None = None
    recording_finished_monotonic: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect live ESP32 DATA rows into a CSV after touch start/stop."
    )
    parser.add_argument("--port", default="/dev/cu.usbserial-0001")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--output", default="data/wellness_data.csv")
    parser.add_argument("--capacitive-field", default="capacitive_filtered_us")
    parser.add_argument("--touch-threshold", type=float, default=None)
    parser.add_argument("--release-threshold", type=float, default=None)
    parser.add_argument(
        "--touch-direction",
        choices=("above", "below"),
        default="above",
    )
    parser.add_argument("--calibration-samples", type=int, default=50)
    parser.add_argument("--debounce-ms", type=int, default=750)
    parser.add_argument("--serial-timeout", type=float, default=2)
    parser.add_argument("--reset-wait", type=float, default=2)
    parser.add_argument("--reconnect-delay", type=float, default=3)
    parser.add_argument("--status-interval", type=float, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--print-all-samples", action="store_true")
    return parser.parse_args()


def list_serial_ports() -> None:
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        candidates = sorted(set(glob.glob("/dev/cu.*") + glob.glob("/dev/tty.*")))
        print(
            "pyserial is not installed for this Python interpreter; showing basic "
            "/dev serial candidates instead."
        )
        print("Install pyserial with: python3 -m pip install pyserial")
        if not candidates:
            print("No serial port candidates found.")
            return
        for candidate in candidates:
            print(candidate)
        return

    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    for port in ports:
        print(f"{port.device}\t{port.description}")


def validate_args(args: argparse.Namespace) -> None:
    if args.baud <= 0:
        raise ValueError("--baud must be greater than 0.")
    if args.calibration_samples <= 0:
        raise ValueError("--calibration-samples must be greater than 0.")
    if args.debounce_ms < 0:
        raise ValueError("--debounce-ms must be greater than or equal to 0.")
    if args.serial_timeout <= 0:
        raise ValueError("--serial-timeout must be greater than 0.")
    if args.reset_wait < 0:
        raise ValueError("--reset-wait must be greater than or equal to 0.")
    if args.reconnect_delay <= 0:
        raise ValueError("--reconnect-delay must be greater than 0.")
    if args.status_interval <= 0:
        raise ValueError("--status-interval must be greater than 0.")


def open_serial(args: argparse.Namespace):
    try:
        import serial
        from serial import SerialException
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required to open the ESP32 serial port. Install it with: "
            "python3 -m pip install pyserial"
        ) from exc

    while True:
        try:
            connection = serial.Serial(
                port=args.port,
                baudrate=args.baud,
                timeout=args.serial_timeout,
            )
            break
        except SerialException as exc:
            print(
                f"Could not open serial port {args.port}: {exc}. "
                f"Retrying in {args.reconnect_delay:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(args.reconnect_delay)

    print(f"Opened {args.port} at {args.baud} baud.")
    print(f"Waiting {args.reset_wait:.1f}s for possible ESP32 reset...")
    time.sleep(args.reset_wait)
    connection.reset_input_buffer()
    return connection


def is_serial_exception(exc: BaseException) -> bool:
    return exc.__class__.__name__ == "SerialException"


def close_serial_quietly(connection) -> None:
    try:
        connection.close()
    except Exception:
        pass


def reconnect_serial(connection, args: argparse.Namespace, expected_fields: list[str]):
    close_serial_quietly(connection)
    print(
        f"Serial connection lost. Reconnecting to {args.port} every "
        f"{args.reconnect_delay:.1f}s...",
        file=sys.stderr,
    )
    connection = open_serial(args)
    fields = wait_for_fields(connection)
    if fields != expected_fields:
        raise RuntimeError(
            "Firmware FIELDS changed after reconnect; restart the collector so the "
            "CSV header matches the DATA rows."
        )
    print("Serial reconnected.")
    return connection


def resolve_output_path(path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10000):
        candidate = path.with_name(f"{stem}_{index:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a numbered output filename for {path}.")


def is_ignored_line(line: str) -> bool:
    if not line:
        return True
    return line.startswith(BOOT_PREFIXES)


def parse_fields(line: str) -> list[str] | None:
    if not line.startswith("FIELDS,"):
        return None
    fields = line.split(",")[1:]
    if not fields:
        raise ValueError("FIELDS line did not contain any columns.")
    return fields


def parse_data_line(line: str, fields: list[str]) -> dict[str, str] | None:
    if not line.startswith("DATA,"):
        return None
    values = line.split(",")[1:]
    if len(values) != len(fields):
        raise ValueError(
            f"DATA row has {len(values)} values but expected {len(fields)}."
        )
    return dict(zip(fields, values))


def numeric_value(row: dict[str, str], field: str) -> float:
    if field not in row:
        raise ValueError(f"Required field is missing from DATA row: {field}")
    value = row[field].strip()
    if value == "":
        raise ValueError(f"Required field is blank in DATA row: {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Field {field} contains NaN or infinity.")
    return number


def validate_numeric_row(row: dict[str, str]) -> None:
    for field, value in row.items():
        if field in TEXT_FIELDS:
            continue
        text = value.strip()
        if text == "":
            continue
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"Field {field} contains nonnumeric data: {text}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Field {field} contains NaN or infinity.")


def median_absolute_deviation(values: list[float], median_value: float) -> float:
    deviations = [abs(value - median_value) for value in values]
    return float(statistics.median(deviations))


def calculate_thresholds(values: list[float], args: argparse.Namespace) -> Thresholds:
    baseline = float(statistics.median(values))
    mad = median_absolute_deviation(values, baseline)
    spread = max(5.0, mad * 8.0, abs(baseline) * 0.50)
    release_spread = max(2.0, mad * 4.0, abs(baseline) * 0.25)

    if args.touch_direction == "above":
        automatic_touch = baseline + spread
        automatic_release = min(baseline + release_spread, automatic_touch - 1.0)
    else:
        automatic_touch = baseline - spread
        automatic_release = max(baseline - release_spread, automatic_touch + 1.0)

    touch_threshold = (
        float(args.touch_threshold)
        if args.touch_threshold is not None
        else automatic_touch
    )
    release_threshold = (
        float(args.release_threshold)
        if args.release_threshold is not None
        else automatic_release
    )

    if args.touch_direction == "above" and release_threshold >= touch_threshold:
        raise ValueError("For touch-direction above, release threshold must be lower.")
    if args.touch_direction == "below" and release_threshold <= touch_threshold:
        raise ValueError("For touch-direction below, release threshold must be higher.")

    return Thresholds(
        baseline_median=baseline,
        baseline_mad=mad,
        touch_threshold=touch_threshold,
        release_threshold=release_threshold,
    )


def is_touched(value: float, thresholds: Thresholds, direction: str) -> bool:
    if direction == "above":
        return value >= thresholds.touch_threshold
    return value <= thresholds.touch_threshold


def is_released(value: float, thresholds: Thresholds, direction: str) -> bool:
    if direction == "above":
        return value <= thresholds.release_threshold
    return value >= thresholds.release_threshold


def wait_for_fields(connection) -> list[str]:
    print("Waiting for FIELDS line from ESP32...")
    while True:
        raw = connection.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if is_ignored_line(line):
            continue
        if line.startswith("INFO,"):
            print(line)
            continue
        if line.startswith("WARNING,"):
            print(line)
            continue
        if line.startswith("ERROR,"):
            print(line, file=sys.stderr)
            continue
        fields = parse_fields(line)
        if fields is not None:
            print(f"Received FIELDS: {fields}")
            return fields
        print(f"Skipping non-protocol line before FIELDS: {line}")


def read_next_data_row(
    connection,
    fields: list[str],
    stats: RunStats,
    print_protocol: bool = True,
) -> dict[str, str] | None:
    raw = connection.readline()
    if not raw:
        return None
    line = raw.decode("utf-8", errors="ignore").strip()
    if is_ignored_line(line):
        return None
    if line.startswith("INFO,"):
        if print_protocol:
            print(line)
        return None
    if line.startswith("WARNING,"):
        if print_protocol:
            print(line)
        return None
    if line.startswith("ERROR,"):
        if print_protocol:
            print(line, file=sys.stderr)
        return None
    if line.startswith("FIELDS,"):
        return None
    try:
        row = parse_data_line(line, fields)
        if row is None:
            stats.malformed_rows += 1
            return None
        validate_numeric_row(row)
        return row
    except ValueError as exc:
        stats.malformed_rows += 1
        print(f"Malformed row skipped: {exc}", file=sys.stderr)
        return None


def calibrate(
    connection,
    fields: list[str],
    args: argparse.Namespace,
    stats: RunStats,
) -> Thresholds:
    if args.capacitive_field not in fields:
        raise ValueError(
            f"Capacitive field {args.capacitive_field!r} was not found in FIELDS: "
            f"{fields}"
        )

    print("Do not touch the foil. Collecting capacitive baseline samples...")
    samples: list[float] = []
    while len(samples) < args.calibration_samples:
        row = read_next_data_row(connection, fields, stats)
        if row is None:
            continue
        try:
            cap_value = numeric_value(row, args.capacitive_field)
        except ValueError as exc:
            stats.malformed_rows += 1
            print(f"Calibration row skipped: {exc}", file=sys.stderr)
            continue
        samples.append(cap_value)
        print(
            f"Calibration {len(samples)}/{args.calibration_samples}: "
            f"{args.capacitive_field}={cap_value:.3f}"
        )

    thresholds = calculate_thresholds(samples, args)
    print("Capacitive calibration complete.")
    print(f"Baseline median: {thresholds.baseline_median:.3f}")
    print(f"Baseline MAD: {thresholds.baseline_mad:.3f}")
    print(f"Touch threshold: {thresholds.touch_threshold:.3f}")
    print(f"Release threshold: {thresholds.release_threshold:.3f}")
    print("Waiting for capacitive touch to start recording...")
    return thresholds


def create_writer(
    output_path: Path,
    fields: list[str],
) -> tuple[TextIO, csv.DictWriter[str]]:
    file_obj = output_path.open("w", newline="", encoding="utf-8")
    output_fields = fields + [
        "host_timestamp_iso",
        "recording_elapsed_seconds",
        "sample_number",
    ]
    writer: csv.DictWriter[str] = csv.DictWriter(file_obj, fieldnames=output_fields)
    writer.writeheader()
    file_obj.flush()
    return file_obj, writer


def display_status(
    row: dict[str, str],
    stats: RunStats,
    args: argparse.Namespace,
    state: CollectorState,
) -> None:
    cap = row.get(args.capacitive_field, "")
    cap_valid = row.get("capacitive_valid", "")
    therm_mv = row.get("thermistor_millivolts", "")
    temp = row.get("temperature_f", "")
    bpm = row.get("bpm", "")
    touch = row.get("touch_status", "")
    movement = row.get("movement_status", "")
    mpu = row.get("mpu_connected", "")
    accel = ""
    if row.get("kalman_accel_x_g", "") and row.get("kalman_accel_y_g", "") and row.get("kalman_accel_z_g", ""):
        accel = (
            f", kalman_accel_g=({row['kalman_accel_x_g']},"
            f"{row['kalman_accel_y_g']},{row['kalman_accel_z_g']})"
        )

    elapsed = 0.0
    if stats.recording_started_monotonic is not None:
        elapsed = time.monotonic() - stats.recording_started_monotonic

    parts = [
        f"state={state.value}",
        f"samples={stats.samples_saved}",
        f"elapsed={elapsed:.1f}s",
        f"cap={cap}",
        f"cap_valid={cap_valid}",
        f"touch={touch}",
        f"thermistor_mV={therm_mv}",
        f"mpu={mpu}",
    ]
    if temp:
        parts.append(f"temperature_f={temp}")
    if bpm:
        parts.append(f"bpm={bpm}")
    if movement:
        parts.append(f"movement={movement}")
    print(", ".join(parts) + accel)


def save_recording_row(
    writer: csv.DictWriter[str],
    file_obj: TextIO,
    row: dict[str, str],
    stats: RunStats,
) -> None:
    now = datetime.now(timezone.utc)
    if stats.recording_started_monotonic is None:
        stats.recording_started_monotonic = time.monotonic()
    elapsed = time.monotonic() - stats.recording_started_monotonic
    sample_number = stats.samples_saved + 1

    output_row = dict(row)
    output_row["host_timestamp_iso"] = now.isoformat()
    output_row["recording_elapsed_seconds"] = f"{elapsed:.6f}"
    output_row["sample_number"] = str(sample_number)
    writer.writerow(output_row)
    file_obj.flush()

    stats.samples_saved = sample_number
    timestamp = row.get("timestamp_ms", "")
    if stats.first_esp32_timestamp is None:
        stats.first_esp32_timestamp = timestamp
    stats.final_esp32_timestamp = timestamp


def print_final_report(
    stop_reason: str,
    stats: RunStats,
    output_path: Path,
) -> None:
    if stats.recording_finished_monotonic is None:
        stats.recording_finished_monotonic = time.monotonic()

    if stats.recording_started_monotonic is not None:
        duration = stats.recording_finished_monotonic - stats.recording_started_monotonic
    else:
        duration = 0.0
    sample_rate = stats.samples_saved / duration if duration > 0 else 0.0
    file_size = output_path.stat().st_size if output_path.exists() else 0

    print("\nCollection summary")
    print(f"Stop reason: {stop_reason}")
    print(f"Samples saved: {stats.samples_saved}")
    print(f"Malformed rows skipped: {stats.malformed_rows}")
    print(f"Duration: {duration:.3f} seconds")
    print(f"Approximate sampling rate: {sample_rate:.3f} samples/second")
    print(f"First ESP32 timestamp: {stats.first_esp32_timestamp or ''}")
    print(f"Final ESP32 timestamp: {stats.final_esp32_timestamp or ''}")
    print(f"Final CSV path: {output_path}")
    print(f"Final file size: {file_size} bytes")


def collect(args: argparse.Namespace) -> None:
    output_path = resolve_output_path(Path(args.output), args.overwrite)
    stats = RunStats()
    state = CollectorState.CALIBRATING
    stop_reason = "not_started"
    file_obj: TextIO | None = None
    writer: csv.DictWriter[str] | None = None
    released = True
    last_touch_time_ms = -float("inf")
    last_status_time = 0.0

    connection = open_serial(args)
    try:
        fields = wait_for_fields(connection)
        thresholds = calibrate(connection, fields, args, stats)
        file_obj, writer = create_writer(output_path, fields)
        state = CollectorState.WAITING_FOR_START

        while state != CollectorState.FINISHED:
            try:
                row = read_next_data_row(connection, fields, stats)
            except Exception as exc:
                if not is_serial_exception(exc):
                    raise
                connection = reconnect_serial(connection, args, fields)
                released = True
                continue
            if row is None:
                continue

            cap_value = numeric_value(row, args.capacitive_field)
            now = time.monotonic()
            now_ms = now * 1000.0
            touched = is_touched(cap_value, thresholds, args.touch_direction)
            released_now = is_released(cap_value, thresholds, args.touch_direction)

            if released_now:
                released = True

            should_print = args.print_all_samples or (now - last_status_time >= args.status_interval)
            if should_print:
                display_status(row, stats, args, state)
                last_status_time = now

            if touched and released and now_ms - last_touch_time_ms >= args.debounce_ms:
                last_touch_time_ms = now_ms
                released = False

                if state == CollectorState.WAITING_FOR_START:
                    stats.recording_started_monotonic = time.monotonic()
                    state = CollectorState.RECORDING
                    print("Recording started. Touch the capacitive sensor again to stop.")
                    continue

                if state == CollectorState.RECORDING:
                    state = CollectorState.FINISHED
                    stats.recording_finished_monotonic = time.monotonic()
                    stop_reason = "second capacitive touch"
                    break

            if state == CollectorState.RECORDING:
                if writer is None or file_obj is None:
                    raise RuntimeError("CSV writer was not initialized.")
                save_recording_row(writer, file_obj, row, stats)

        if stop_reason == "not_started":
            stop_reason = "finished"
    except KeyboardInterrupt:
        stop_reason = "Control-C"
        stats.recording_finished_monotonic = time.monotonic()
        print("\nControl-C received. Preserving partial recording.")
    except Exception as exc:
        if not is_serial_exception(exc):
            raise
        stop_reason = f"serial disconnection: {exc}"
        stats.recording_finished_monotonic = time.monotonic()
        print(f"\nSerial connection issue. Preserving partial recording: {exc}", file=sys.stderr)
    finally:
        if file_obj is not None:
            file_obj.flush()
            file_obj.close()
        connection.close()
        print_final_report(stop_reason, stats, output_path)


def main() -> None:
    args = parse_args()
    if args.list_ports:
        list_serial_ports()
        return

    try:
        validate_args(args)
        collect(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
