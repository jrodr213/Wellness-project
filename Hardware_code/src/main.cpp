#include <Arduino.h>
#include <HTTPClient.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <Wire.h>
#include <freertos/task.h>
#include <math.h>
#include <string.h>

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID ""
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD ""
#endif

#ifndef SERVER_URL
#define SERVER_URL ""
#endif

const int THERMISTOR_PIN = 35;
const int PULSE_PIN = 34;
const int CAP_SEND_PIN = 25;
const int CAP_RECEIVE_PIN = 27;
const int MPU_SDA_PIN = 21;
const int MPU_SCL_PIN = 22;
const uint8_t MPU_ADDRESS = 0x68;
const int SERIAL_BAUD = 115200;

const int ADC_MAX_VALUE = 4095;
const float ADC_SUPPLY_VOLTS = 3.3f;
const float ADC_SUPPLY_MILLIVOLTS = ADC_SUPPLY_VOLTS * 1000.0f;

const int THERMISTOR_SAMPLES = 16;
// Confirmed divider: 3.3 V -> fixed resistor -> LM358-buffered node -> NTC || 10 k -> GND.
const bool THERMISTOR_FIXED_RESISTOR_TO_3V3 = true;
const float LM358_VOLTAGE_GAIN = 1.0f;
const float THERMISTOR_FIXED_RESISTOR_OHMS = 10000.0f;
const float THERMISTOR_PARALLEL_RESISTOR_OHMS = 10000.0f;
const float THERMISTOR_NOMINAL_RESISTANCE_OHMS = 10000.0f;
const float THERMISTOR_NOMINAL_TEMPERATURE_C = 25.0f;
const float THERMISTOR_BETA_K = 3950.0f;
// TODO: Replace gain and offset only after measured thermistor calibration values are available.
const float TEMPERATURE_CALIBRATION_GAIN = 1.0f;
const float TEMPERATURE_CALIBRATION_OFFSET_F = 0.0f;

const unsigned long PULSE_SAMPLE_INTERVAL_MS = 5;
const unsigned long PULSE_REFRACTORY_MS = 250;
const unsigned long PULSE_MIN_VALID_INTERVAL_MS = 300;
const unsigned long PULSE_MAX_VALID_INTERVAL_MS = 1500;
const unsigned long PULSE_SIGNAL_TIMEOUT_MS = 4000;
const float PULSE_CALIBRATION_OFFSET_BPM = -4.0f;
const int PULSE_INTERVAL_HISTORY = 5;
const float PULSE_BASELINE_ALPHA = 0.01f;
const float PULSE_FILTER_ALPHA = 0.25f;
const float PULSE_NOISE_ALPHA = 0.05f;
const float PULSE_MIN_THRESHOLD_ADC = 30.0f;
const float PULSE_THRESHOLD_NOISE_MULTIPLIER = 1.8f;

const int CAPACITIVE_SAMPLES = 30;
const unsigned long CAPACITIVE_TIMEOUT_US = 30000;
const float CAPACITIVE_FILTER_ALPHA = 0.20f;
const int CAPACITIVE_BASELINE_SAMPLES = 50;
const int CAPACITIVE_BASELINE_MAX_ATTEMPTS = CAPACITIVE_BASELINE_SAMPLES * 10;
const float CAPACITIVE_MIN_TOUCH_INCREASE_US = 8.0f;
const float CAPACITIVE_TOUCH_INCREASE_RATIO = 0.50f;

// X uses measured calibration; Y and Z remain datasheet/default conversion until measured.
const float ACCEL_X_OFFSET_COUNTS = 103.0f;
const float ACCEL_X_COUNTS_PER_G = 16137.0f;
const float ACCEL_Y_OFFSET_COUNTS = 0.0f;
const float ACCEL_Y_COUNTS_PER_G = 16384.0f;
const float ACCEL_Z_OFFSET_COUNTS = 0.0f;
const float ACCEL_Z_COUNTS_PER_G = 16384.0f;
const float KALMAN_PROCESS_NOISE = 0.01f;
const float KALMAN_MEASUREMENT_NOISE = 0.08f;
const float KALMAN_INITIAL_ERROR = 1.0f;
const float MOVEMENT_THRESHOLD_G = 0.12f;
const int MOVEMENT_STATUS_CONFIRMATION_SAMPLES = 3;

const unsigned long TOUCH_POLL_INTERVAL_MS = 50;
const int TOUCH_CONFIRMATION_SAMPLES = 2;
const int RELEASE_CONFIRMATION_SAMPLES = 2;
const unsigned long WIFI_RECONNECT_INTERVAL_MS = 10000;
const unsigned long WIFI_CONNECT_ATTEMPT_TIMEOUT_MS = 8000;
const unsigned long HTTP_TRANSMIT_INTERVAL_MS = 2000;
const unsigned long MEASUREMENT_INTERVAL_MS = HTTP_TRANSMIT_INTERVAL_MS;

enum MeasurementState {
    WAITING_TO_START,
    WAITING_FOR_START_RELEASE,
    MEASURING,
    WAITING_FOR_STOP_RELEASE
};

MPU6050 mpu(MPU_ADDRESS);
bool mpuConnected = false;
bool fieldsPrinted = false;
bool wifiConfiguredMessagePrinted = false;
bool serverConfiguredMessagePrinted = false;
bool wifiWasConnected = false;
bool wifiConnectInProgress = false;
unsigned long lastTouchPollMs = 0;
unsigned long lastWifiAttemptMs = 0;
unsigned long wifiAttemptStartedMs = 0;
unsigned long lastHttpTransmitMs = 0;
MeasurementState measurementState = WAITING_TO_START;
int touchConfirmationCount = 0;
int releaseConfirmationCount = 0;
unsigned long lastMeasurementMs = 0;

struct TemperatureSummary {
    bool valid;
    float averageRaw;
    float averageMilliVolts;
    float lowerResistanceOhms;
    float thermistorResistanceOhms;
    float temperatureF;
};

struct PulseSummary {
    int rawAdc;
    float baseline;
    float filtered;
    float threshold;
    bool bpmValid;
    float bpm;
};

struct CapacitiveSummary {
    bool valid;
    unsigned long rawUs;
    float filteredUs;
    float baselineUs;
    float thresholdUs;
    bool touched;
    bool timeout;
    int timeoutSamples;
};

struct MpuSummary {
    bool valid;
    int16_t accelXRaw;
    int16_t accelYRaw;
    int16_t accelZRaw;
    int16_t gyroXRaw;
    int16_t gyroYRaw;
    int16_t gyroZRaw;
    float accelXG;
    float accelYG;
    float accelZG;
    float kalmanAccelXG;
    float kalmanAccelYG;
    float kalmanAccelZG;
    float filteredMagnitudeG;
    float movementIntensityG;
    bool moving;
};

class OneDimensionalKalmanFilter {
public:
    OneDimensionalKalmanFilter(float processNoise, float measurementNoise, float initialError)
        : q(processNoise),
          r(measurementNoise),
          p(initialError),
          x(0.0f),
          initialized(false) {}

    float update(float measurement) {
        if (!initialized) {
            x = measurement;
            initialized = true;
            return x;
        }

        p += q;
        const float gain = p / (p + r);
        x += gain * (measurement - x);
        p *= (1.0f - gain);
        return x;
    }

private:
    float q;
    float r;
    float p;
    float x;
    bool initialized;
};

OneDimensionalKalmanFilter accelXKalman(
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_INITIAL_ERROR
);
OneDimensionalKalmanFilter accelYKalman(
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_INITIAL_ERROR
);
OneDimensionalKalmanFilter accelZKalman(
    KALMAN_PROCESS_NOISE,
    KALMAN_MEASUREMENT_NOISE,
    KALMAN_INITIAL_ERROR
);

float pulseBaseline = 0.0f;
float pulseFiltered = 0.0f;
float previousPulseFiltered = 0.0f;
float pulseNoise = 0.0f;
float pulseThreshold = PULSE_MIN_THRESHOLD_ADC;
bool pulseInitialized = false;
unsigned long lastPulseSampleMs = 0;
unsigned long lastBeatMs = 0;
unsigned long pulseIntervalsMs[PULSE_INTERVAL_HISTORY] = {0, 0, 0, 0, 0};
int pulseIntervalCount = 0;
int pulseIntervalIndex = 0;
float correctedBpm = 0.0f;
bool bpmValid = false;
int lastPulseRaw = 0;
portMUX_TYPE pulseStateMux = portMUX_INITIALIZER_UNLOCKED;
TaskHandle_t pulseTaskHandle = nullptr;

float capacitiveBaselineUs = 0.0f;
float capacitiveFilteredUs = 0.0f;
float capacitiveThresholdUs = 0.0f;
bool capacitiveBaselineReady = false;

bool movementMoving = false;
int movementAboveCount = 0;
int movementBelowCount = 0;

bool hasConfiguredWifi() {
    return strlen(WIFI_SSID) > 0;
}

bool hasConfiguredServer() {
    return strlen(SERVER_URL) > 0;
}

unsigned long elapsedMillis(unsigned long now, unsigned long previous) {
    return now - previous;
}

uint32_t readMilliVoltsCompat(int pin, int rawValue) {
#if defined(ARDUINO_ARCH_ESP32)
    return analogReadMilliVolts(pin);
#else
    return static_cast<uint32_t>((static_cast<float>(rawValue) / ADC_MAX_VALUE) * ADC_SUPPLY_MILLIVOLTS);
#endif
}

TemperatureSummary readTemperatureSummary() {
    uint32_t rawSum = 0;
    uint32_t milliVoltSum = 0;
    int validCount = 0;

    for (int i = 0; i < THERMISTOR_SAMPLES; i++) {
        const int raw = analogRead(THERMISTOR_PIN);
        if (raw > 0 && raw < ADC_MAX_VALUE) {
            rawSum += static_cast<uint32_t>(raw);
            milliVoltSum += readMilliVoltsCompat(THERMISTOR_PIN, raw);
            validCount++;
        }
        delayMicroseconds(250);
    }

    if (validCount == 0) {
        return {false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    const float averageRaw = static_cast<float>(rawSum) / static_cast<float>(validCount);
    const float averageMilliVolts =
        static_cast<float>(milliVoltSum) / static_cast<float>(validCount);
    const float adcVoltage = averageMilliVolts / 1000.0f;

    if (!THERMISTOR_FIXED_RESISTOR_TO_3V3 || LM358_VOLTAGE_GAIN <= 0.0f) {
        return {false, averageRaw, averageMilliVolts, 0.0f, 0.0f, 0.0f};
    }

    const float dividerNodeVoltage = adcVoltage / LM358_VOLTAGE_GAIN;
    const float denominator = ADC_SUPPLY_VOLTS - dividerNodeVoltage;

    if (dividerNodeVoltage <= 0.0f ||
        dividerNodeVoltage >= ADC_SUPPLY_VOLTS ||
        denominator <= 0.0f) {
        return {false, averageRaw, averageMilliVolts, 0.0f, 0.0f, 0.0f};
    }

    const float lowerResistanceOhms =
        THERMISTOR_FIXED_RESISTOR_OHMS * dividerNodeVoltage / denominator;
    if (!isfinite(lowerResistanceOhms) || lowerResistanceOhms <= 0.0f) {
        return {false, averageRaw, averageMilliVolts, lowerResistanceOhms, 0.0f, 0.0f};
    }

    const float thermistorReciprocal =
        (1.0f / lowerResistanceOhms) - (1.0f / THERMISTOR_PARALLEL_RESISTOR_OHMS);
    if (!isfinite(thermistorReciprocal) || thermistorReciprocal <= 0.0f) {
        return {false, averageRaw, averageMilliVolts, lowerResistanceOhms, 0.0f, 0.0f};
    }

    const float thermistorResistanceOhms = 1.0f / thermistorReciprocal;
    const float nominalTemperatureK = THERMISTOR_NOMINAL_TEMPERATURE_C + 273.15f;
    const float logRatio =
        logf(thermistorResistanceOhms / THERMISTOR_NOMINAL_RESISTANCE_OHMS);
    const float inverseTemperatureK =
        (1.0f / nominalTemperatureK) + (logRatio / THERMISTOR_BETA_K);

    if (!isfinite(logRatio) || !isfinite(inverseTemperatureK) || inverseTemperatureK <= 0.0f) {
        return {
            false,
            averageRaw,
            averageMilliVolts,
            lowerResistanceOhms,
            thermistorResistanceOhms,
            0.0f
        };
    }

    const float temperatureC = (1.0f / inverseTemperatureK) - 273.15f;
    const float temperatureF = ((temperatureC * 9.0f / 5.0f) + 32.0f) *
        TEMPERATURE_CALIBRATION_GAIN + TEMPERATURE_CALIBRATION_OFFSET_F;

    if (!isfinite(temperatureF)) {
        return {
            false,
            averageRaw,
            averageMilliVolts,
            lowerResistanceOhms,
            thermistorResistanceOhms,
            0.0f
        };
    }

    return {
        true,
        averageRaw,
        averageMilliVolts,
        lowerResistanceOhms,
        thermistorResistanceOhms,
        temperatureF
    };
}

void resetPulseHistoryLocked() {
    for (int i = 0; i < PULSE_INTERVAL_HISTORY; i++) {
        pulseIntervalsMs[i] = 0;
    }
    pulseIntervalCount = 0;
    pulseIntervalIndex = 0;
    correctedBpm = 0.0f;
    bpmValid = false;
}

void acceptPulseIntervalLocked(unsigned long intervalMs) {
    pulseIntervalsMs[pulseIntervalIndex] = intervalMs;
    pulseIntervalIndex = (pulseIntervalIndex + 1) % PULSE_INTERVAL_HISTORY;
    if (pulseIntervalCount < PULSE_INTERVAL_HISTORY) {
        pulseIntervalCount++;
    }

    if (pulseIntervalCount < PULSE_INTERVAL_HISTORY) {
        bpmValid = false;
        return;
    }

    unsigned long intervalSum = 0;
    for (int i = 0; i < PULSE_INTERVAL_HISTORY; i++) {
        intervalSum += pulseIntervalsMs[i];
    }

    const float averageIntervalMs =
        static_cast<float>(intervalSum) / static_cast<float>(PULSE_INTERVAL_HISTORY);
    correctedBpm = max(
        0.0f,
        (60000.0f / averageIntervalMs) + PULSE_CALIBRATION_OFFSET_BPM
    );
    bpmValid = true;
}

void updatePulseStateFromSample(int rawAdc, unsigned long now) {
    portENTER_CRITICAL(&pulseStateMux);
    lastPulseRaw = rawAdc;
    lastPulseSampleMs = now;

    if (!pulseInitialized) {
        pulseBaseline = static_cast<float>(rawAdc);
        pulseFiltered = 0.0f;
        previousPulseFiltered = 0.0f;
        pulseNoise = PULSE_MIN_THRESHOLD_ADC;
        pulseThreshold = PULSE_MIN_THRESHOLD_ADC;
        pulseInitialized = true;
        portEXIT_CRITICAL(&pulseStateMux);
        return;
    }

    if (lastBeatMs != 0 && elapsedMillis(now, lastBeatMs) > PULSE_SIGNAL_TIMEOUT_MS) {
        resetPulseHistoryLocked();
        lastBeatMs = 0;
    }

    pulseBaseline += PULSE_BASELINE_ALPHA * (static_cast<float>(rawAdc) - pulseBaseline);
    const float baselineRemoved = static_cast<float>(rawAdc) - pulseBaseline;
    previousPulseFiltered = pulseFiltered;
    pulseFiltered += PULSE_FILTER_ALPHA * (baselineRemoved - pulseFiltered);
    pulseNoise += PULSE_NOISE_ALPHA * (fabsf(pulseFiltered) - pulseNoise);
    pulseThreshold = max(PULSE_MIN_THRESHOLD_ADC, pulseNoise * PULSE_THRESHOLD_NOISE_MULTIPLIER);

    const bool crossedThreshold =
        previousPulseFiltered <= pulseThreshold && pulseFiltered > pulseThreshold;
    const bool rising = pulseFiltered > previousPulseFiltered;
    if (!crossedThreshold || !rising) {
        portEXIT_CRITICAL(&pulseStateMux);
        return;
    }

    if (lastBeatMs == 0) {
        resetPulseHistoryLocked();
        lastBeatMs = now;
        portEXIT_CRITICAL(&pulseStateMux);
        return;
    }

    const unsigned long intervalMs = elapsedMillis(now, lastBeatMs);
    if (intervalMs < PULSE_REFRACTORY_MS || intervalMs < PULSE_MIN_VALID_INTERVAL_MS) {
        portEXIT_CRITICAL(&pulseStateMux);
        return;
    }

    if (intervalMs > PULSE_MAX_VALID_INTERVAL_MS) {
        resetPulseHistoryLocked();
        lastBeatMs = now;
        portEXIT_CRITICAL(&pulseStateMux);
        return;
    }

    acceptPulseIntervalLocked(intervalMs);
    lastBeatMs = now;
    portEXIT_CRITICAL(&pulseStateMux);
}

void pulseSamplingTask(void *parameter) {
    (void)parameter;
    TickType_t lastWakeTime = xTaskGetTickCount();

    while (true) {
        const int rawAdc = analogRead(PULSE_PIN);
        updatePulseStateFromSample(rawAdc, millis());
        vTaskDelayUntil(&lastWakeTime, pdMS_TO_TICKS(PULSE_SAMPLE_INTERVAL_MS));
    }
}

void startPulseSamplingTask() {
    // A separate task keeps 5 ms pulse sampling alive during HTTP and slower sensor operations.
    const BaseType_t created = xTaskCreatePinnedToCore(
        pulseSamplingTask,
        "pulse_sampling",
        4096,
        nullptr,
        1,
        &pulseTaskHandle,
        1
    );

    if (created == pdPASS) {
        Serial.println("INFO,pulse_sampling_task_started");
    } else {
        Serial.println("ERROR,pulse_sampling_task_failed");
    }
}

PulseSummary readPulseSummary() {
    portENTER_CRITICAL(&pulseStateMux);
    const PulseSummary summary = {
        lastPulseRaw,
        pulseBaseline,
        pulseFiltered,
        pulseThreshold,
        bpmValid,
        correctedBpm
    };
    portEXIT_CRITICAL(&pulseStateMux);
    return summary;
}

unsigned long readCapacitiveSensor() {
    pinMode(CAP_SEND_PIN, OUTPUT);
    digitalWrite(CAP_SEND_PIN, LOW);

    pinMode(CAP_RECEIVE_PIN, OUTPUT);
    digitalWrite(CAP_RECEIVE_PIN, LOW);

    delayMicroseconds(10);

    pinMode(CAP_RECEIVE_PIN, INPUT);
    digitalWrite(CAP_SEND_PIN, HIGH);

    const unsigned long startTime = micros();

    while (digitalRead(CAP_RECEIVE_PIN) == LOW) {
        if (micros() - startTime >= CAPACITIVE_TIMEOUT_US) {
            break;
        }
    }

    const unsigned long chargeTime = micros() - startTime;

    digitalWrite(CAP_SEND_PIN, LOW);
    pinMode(CAP_RECEIVE_PIN, OUTPUT);
    digitalWrite(CAP_RECEIVE_PIN, LOW);

    return chargeTime;
}

CapacitiveSummary readCapacitiveSummary() {
    unsigned long validSum = 0;
    int validCount = 0;
    int timeoutCount = 0;

    for (int i = 0; i < CAPACITIVE_SAMPLES; i++) {
        const unsigned long reading = readCapacitiveSensor();
        if (reading >= CAPACITIVE_TIMEOUT_US) {
            timeoutCount++;
        } else {
            validSum += reading;
            validCount++;
        }
        delayMicroseconds(250);
    }

    if (validCount == 0 || !capacitiveBaselineReady) {
        return {
            false,
            CAPACITIVE_TIMEOUT_US,
            capacitiveFilteredUs,
            capacitiveBaselineUs,
            capacitiveThresholdUs,
            false,
            timeoutCount > 0,
            timeoutCount
        };
    }

    const unsigned long rawUs = validSum / static_cast<unsigned long>(validCount);
    capacitiveFilteredUs +=
        CAPACITIVE_FILTER_ALPHA * (static_cast<float>(rawUs) - capacitiveFilteredUs);
    const bool timeout = timeoutCount > 0;
    const bool touched = !timeout && capacitiveFilteredUs >= capacitiveThresholdUs;

    return {
        true,
        rawUs,
        capacitiveFilteredUs,
        capacitiveBaselineUs,
        capacitiveThresholdUs,
        touched,
        timeout,
        timeoutCount
    };
}

void calibrateCapacitiveBaseline() {
    Serial.println("INFO,capacitive_baseline_calibration,do_not_touch_foil");
    unsigned long sum = 0;
    int count = 0;
    int attempts = 0;

    while (count < CAPACITIVE_BASELINE_SAMPLES && attempts < CAPACITIVE_BASELINE_MAX_ATTEMPTS) {
        attempts++;
        const unsigned long reading = readCapacitiveSensor();
        if (reading < CAPACITIVE_TIMEOUT_US) {
            sum += reading;
            count++;
        }
        delay(10);
    }

    if (count < CAPACITIVE_BASELINE_SAMPLES) {
        capacitiveBaselineReady = false;
        capacitiveBaselineUs = 0.0f;
        capacitiveFilteredUs = 0.0f;
        capacitiveThresholdUs = 0.0f;
        Serial.print("ERROR,capacitive_baseline_calibration_failed,valid_samples,");
        Serial.print(count);
        Serial.print(",attempts,");
        Serial.println(attempts);
        return;
    }

    capacitiveBaselineUs = static_cast<float>(sum) / static_cast<float>(count);
    capacitiveFilteredUs = capacitiveBaselineUs;
    capacitiveThresholdUs = capacitiveBaselineUs +
        max(CAPACITIVE_MIN_TOUCH_INCREASE_US, capacitiveBaselineUs * CAPACITIVE_TOUCH_INCREASE_RATIO);
    capacitiveBaselineReady = true;

    Serial.print("INFO,capacitive_baseline_us,");
    Serial.println(capacitiveBaselineUs, 2);
    Serial.print("INFO,capacitive_touch_threshold_us,");
    Serial.println(capacitiveThresholdUs, 2);
}

bool scanForMpuAddress() {
    Wire.beginTransmission(MPU_ADDRESS);
    return Wire.endTransmission() == 0;
}

void initializeMpu() {
    if (!scanForMpuAddress()) {
        mpuConnected = false;
        Serial.println("ERROR,mpu_not_detected_at_0x68");
        return;
    }

    mpu.initialize();
    mpuConnected = mpu.testConnection();

    if (mpuConnected) {
        Serial.println("INFO,mpu_detected,1");
    } else {
        Serial.println("ERROR,mpu_not_detected_at_0x68");
    }
}

float calibrateAcceleration(int16_t rawValue, float offsetCounts, float countsPerG) {
    if (countsPerG == 0.0f) {
        return 0.0f;
    }
    return (static_cast<float>(rawValue) + offsetCounts) / countsPerG;
}

MpuSummary readMpuSummary() {
    MpuSummary summary = {
        false,
        0,
        0,
        0,
        0,
        0,
        0,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        movementMoving
    };

    if (!mpuConnected) {
        return summary;
    }

    mpu.getMotion6(
        &summary.accelXRaw,
        &summary.accelYRaw,
        &summary.accelZRaw,
        &summary.gyroXRaw,
        &summary.gyroYRaw,
        &summary.gyroZRaw
    );

    summary.valid = true;
    summary.accelXG = calibrateAcceleration(
        summary.accelXRaw,
        ACCEL_X_OFFSET_COUNTS,
        ACCEL_X_COUNTS_PER_G
    );
    summary.accelYG = calibrateAcceleration(
        summary.accelYRaw,
        ACCEL_Y_OFFSET_COUNTS,
        ACCEL_Y_COUNTS_PER_G
    );
    summary.accelZG = calibrateAcceleration(
        summary.accelZRaw,
        ACCEL_Z_OFFSET_COUNTS,
        ACCEL_Z_COUNTS_PER_G
    );
    summary.kalmanAccelXG = accelXKalman.update(summary.accelXG);
    summary.kalmanAccelYG = accelYKalman.update(summary.accelYG);
    summary.kalmanAccelZG = accelZKalman.update(summary.accelZG);
    summary.filteredMagnitudeG = sqrtf(
        summary.kalmanAccelXG * summary.kalmanAccelXG +
        summary.kalmanAccelYG * summary.kalmanAccelYG +
        summary.kalmanAccelZG * summary.kalmanAccelZG
    );
    summary.movementIntensityG = fabsf(summary.filteredMagnitudeG - 1.0f);

    if (summary.movementIntensityG >= MOVEMENT_THRESHOLD_G) {
        movementAboveCount++;
        movementBelowCount = 0;
        if (movementAboveCount >= MOVEMENT_STATUS_CONFIRMATION_SAMPLES) {
            movementMoving = true;
        }
    } else {
        movementBelowCount++;
        movementAboveCount = 0;
        if (movementBelowCount >= MOVEMENT_STATUS_CONFIRMATION_SAMPLES) {
            movementMoving = false;
        }
    }

    summary.moving = movementMoving;
    return summary;
}

void handleWifi(unsigned long now) {
    if (!hasConfiguredWifi()) {
        if (!wifiConfiguredMessagePrinted) {
            Serial.println("WARNING,wifi_not_configured");
            wifiConfiguredMessagePrinted = true;
        }
        return;
    }

    if (WiFi.status() == WL_CONNECTED) {
        if (!wifiWasConnected) {
            wifiWasConnected = true;
            wifiConnectInProgress = false;
            Serial.print("INFO,wifi_connected,ip,");
            Serial.println(WiFi.localIP());
        }
        return;
    }

    if (wifiWasConnected) {
        Serial.println("WARNING,wifi_disconnected");
    }
    wifiWasConnected = false;

    if (wifiConnectInProgress) {
        if (elapsedMillis(now, wifiAttemptStartedMs) < WIFI_CONNECT_ATTEMPT_TIMEOUT_MS) {
            return;
        }
        wifiConnectInProgress = false;
        WiFi.disconnect(false);
        Serial.println("WARNING,wifi_connect_timeout");
    }

    if (lastWifiAttemptMs != 0 &&
        elapsedMillis(now, lastWifiAttemptMs) < WIFI_RECONNECT_INTERVAL_MS) {
        return;
    }

    lastWifiAttemptMs = now;
    wifiAttemptStartedMs = now;
    wifiConnectInProgress = true;
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.println("INFO,wifi_connecting");
}

void appendJsonFloat(String &json, const char *name, bool valid, float value, int decimals) {
    json += "\"";
    json += name;
    json += "\":";
    if (valid && isfinite(value)) {
        json += String(value, decimals);
    } else {
        json += "null";
    }
}

void appendJsonInt(String &json, const char *name, int value) {
    json += "\"";
    json += name;
    json += "\":";
    json += String(value);
}

void appendJsonUnsignedLong(String &json, const char *name, unsigned long value) {
    json += "\"";
    json += name;
    json += "\":";
    json += String(value);
}

void appendJsonBool(String &json, const char *name, bool value) {
    json += "\"";
    json += name;
    json += "\":";
    json += value ? "true" : "false";
}

String buildMeasurementJson(
    unsigned long timestampMs,
    const TemperatureSummary &temperature,
    const PulseSummary &pulse,
    const CapacitiveSummary &capacitive,
    const MpuSummary &mpuSummary
) {
    String json = "{";
    appendJsonUnsignedLong(json, "timestamp_ms", timestampMs);
    json += ",";
    appendJsonFloat(json, "temperature_f", temperature.valid, temperature.temperatureF, 2);
    json += ",";
    appendJsonBool(json, "temperature_valid", temperature.valid);
    json += ",";
    appendJsonFloat(json, "bpm", pulse.bpmValid, pulse.bpm, 2);
    json += ",";
    appendJsonBool(json, "bpm_valid", pulse.bpmValid);
    json += ",";
    appendJsonBool(json, "touch_status", capacitive.touched);
    json += ",";
    appendJsonBool(json, "capacitive_valid", capacitive.valid);
    json += ",";
    appendJsonFloat(json, "capacitive_filtered_us", capacitive.valid, capacitive.filteredUs, 2);
    json += ",";
    appendJsonBool(json, "capacitive_timeout", capacitive.timeout);
    json += ",";
    appendJsonInt(json, "capacitive_timeout_count", capacitive.timeoutSamples);
    json += ",";
    appendJsonFloat(json, "accel_x_g", mpuSummary.valid, mpuSummary.accelXG, 5);
    json += ",";
    appendJsonFloat(json, "accel_y_g", mpuSummary.valid, mpuSummary.accelYG, 5);
    json += ",";
    appendJsonFloat(json, "accel_z_g", mpuSummary.valid, mpuSummary.accelZG, 5);
    json += ",";
    appendJsonFloat(json, "kalman_accel_x_g", mpuSummary.valid, mpuSummary.kalmanAccelXG, 5);
    json += ",";
    appendJsonFloat(json, "kalman_accel_y_g", mpuSummary.valid, mpuSummary.kalmanAccelYG, 5);
    json += ",";
    appendJsonFloat(json, "kalman_accel_z_g", mpuSummary.valid, mpuSummary.kalmanAccelZG, 5);
    json += ",";
    appendJsonFloat(json, "movement_intensity_g", mpuSummary.valid, mpuSummary.movementIntensityG, 5);
    json += ",\"movement_status\":";
    if (mpuSummary.valid) {
        json += mpuSummary.moving ? "\"MOVING\"" : "\"STILL\"";
    } else {
        json += "null";
    }
    json += ",";
    appendJsonBool(json, "mpu_connected", mpuSummary.valid);
    json += "}";
    return json;
}

bool transmitMeasurements(
    unsigned long now,
    const TemperatureSummary &temperature,
    const PulseSummary &pulse,
    const CapacitiveSummary &capacitive,
    const MpuSummary &mpuSummary,
    bool forceTransmit = false
) {
    if (!forceTransmit && elapsedMillis(now, lastHttpTransmitMs) < HTTP_TRANSMIT_INTERVAL_MS) {
        return false;
    }
    lastHttpTransmitMs = now;

    if (!hasConfiguredServer()) {
        if (!serverConfiguredMessagePrinted) {
            Serial.println("WARNING,server_url_not_configured");
            serverConfiguredMessagePrinted = true;
        }
        return false;
    }

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("ERROR,http_skipped_wifi_disconnected");
        return false;
    }

    HTTPClient http;
    if (!http.begin(SERVER_URL)) {
        Serial.println("ERROR,http_begin_failed");
        return false;
    }
    http.setTimeout(750);

    const String json = buildMeasurementJson(now, temperature, pulse, capacitive, mpuSummary);
    http.addHeader("Content-Type", "application/json");
    const int httpStatus = http.POST(json);
    const bool sent = httpStatus >= 200 && httpStatus < 300;
    if (sent) {
        Serial.print("INFO,http_status,");
        Serial.println(httpStatus);
    } else if (httpStatus > 0) {
        Serial.print("ERROR,http_status,");
        Serial.println(httpStatus);
    } else {
        Serial.print("ERROR,http_post_failed,");
        Serial.println(httpStatus);
    }
    http.end();
    return sent;
}

void printFields() {
    Serial.println(
        "FIELDS,timestamp_ms,"
        "temperature_valid,temperature_f,thermistor_adc_raw,thermistor_millivolts,"
        "thermistor_lower_resistance_ohms,thermistor_resistance_ohms,"
        "pulse_raw_adc,pulse_filtered,pulse_baseline,pulse_threshold,bpm_valid,bpm,"
        "capacitive_valid,capacitive_raw_us,capacitive_filtered_us,capacitive_baseline_us,"
        "capacitive_threshold_us,touch_status,capacitive_timeout,capacitive_timeout_count,"
        "mpu_connected,accel_x_raw,accel_y_raw,accel_z_raw,"
        "gyro_x_raw,gyro_y_raw,gyro_z_raw,"
        "accel_x_g,accel_y_g,accel_z_g,"
        "kalman_accel_x_g,kalman_accel_y_g,kalman_accel_z_g,"
        "accel_magnitude_g,movement_intensity_g,movement_status"
    );
    fieldsPrinted = true;
}

void printFloatOrBlank(bool valid, float value, int decimals) {
    if (valid && isfinite(value)) {
        Serial.print(value, decimals);
    }
}

void printMpuCsvValues(const MpuSummary &summary) {
    Serial.print(summary.valid ? 1 : 0);
    Serial.print(',');
    if (summary.valid) {
        Serial.print(summary.accelXRaw);
    }
    Serial.print(',');
    if (summary.valid) {
        Serial.print(summary.accelYRaw);
    }
    Serial.print(',');
    if (summary.valid) {
        Serial.print(summary.accelZRaw);
    }
    Serial.print(',');
    if (summary.valid) {
        Serial.print(summary.gyroXRaw);
    }
    Serial.print(',');
    if (summary.valid) {
        Serial.print(summary.gyroYRaw);
    }
    Serial.print(',');
    if (summary.valid) {
        Serial.print(summary.gyroZRaw);
    }
    Serial.print(',');
    printFloatOrBlank(summary.valid, summary.accelXG, 5);
    Serial.print(',');
    printFloatOrBlank(summary.valid, summary.accelYG, 5);
    Serial.print(',');
    printFloatOrBlank(summary.valid, summary.accelZG, 5);
    Serial.print(',');
    printFloatOrBlank(summary.valid, summary.kalmanAccelXG, 5);
    Serial.print(',');
    printFloatOrBlank(summary.valid, summary.kalmanAccelYG, 5);
    Serial.print(',');
    printFloatOrBlank(summary.valid, summary.kalmanAccelZG, 5);
    Serial.print(',');
    printFloatOrBlank(summary.valid, summary.filteredMagnitudeG, 5);
    Serial.print(',');
    printFloatOrBlank(summary.valid, summary.movementIntensityG, 5);
    Serial.print(',');
    if (summary.valid) {
        Serial.print(summary.moving ? "MOVING" : "STILL");
    }
}

void printData(
    unsigned long timestampMs,
    const TemperatureSummary &temperature,
    const PulseSummary &pulse,
    const CapacitiveSummary &capacitive,
    const MpuSummary &mpuSummary
) {
    Serial.print("DATA,");
    Serial.print(timestampMs);
    Serial.print(',');
    Serial.print(temperature.valid ? 1 : 0);
    Serial.print(',');
    printFloatOrBlank(temperature.valid, temperature.temperatureF, 2);
    Serial.print(',');
    printFloatOrBlank(temperature.valid, temperature.averageRaw, 2);
    Serial.print(',');
    printFloatOrBlank(temperature.valid, temperature.averageMilliVolts, 2);
    Serial.print(',');
    printFloatOrBlank(temperature.valid, temperature.lowerResistanceOhms, 2);
    Serial.print(',');
    printFloatOrBlank(temperature.valid, temperature.thermistorResistanceOhms, 2);
    Serial.print(',');
    Serial.print(pulse.rawAdc);
    Serial.print(',');
    Serial.print(pulse.filtered, 2);
    Serial.print(',');
    Serial.print(pulse.baseline, 2);
    Serial.print(',');
    Serial.print(pulse.threshold, 2);
    Serial.print(',');
    Serial.print(pulse.bpmValid ? 1 : 0);
    Serial.print(',');
    printFloatOrBlank(pulse.bpmValid, pulse.bpm, 2);
    Serial.print(',');
    Serial.print(capacitive.valid ? 1 : 0);
    Serial.print(',');
    if (capacitive.valid) {
        Serial.print(capacitive.rawUs);
    }
    Serial.print(',');
    printFloatOrBlank(capacitive.valid, capacitive.filteredUs, 2);
    Serial.print(',');
    printFloatOrBlank(capacitiveBaselineReady, capacitive.baselineUs, 2);
    Serial.print(',');
    printFloatOrBlank(capacitiveBaselineReady, capacitive.thresholdUs, 2);
    Serial.print(',');
    Serial.print(capacitive.touched ? 1 : 0);
    Serial.print(',');
    Serial.print(capacitive.timeout ? 1 : 0);
    Serial.print(',');
    Serial.print(capacitive.timeoutSamples);
    Serial.print(',');
    printMpuCsvValues(mpuSummary);
    Serial.println();
}

void collectAndSendMeasurement(
    unsigned long timestampMs,
    const CapacitiveSummary &triggerCapacitive
) {
    const TemperatureSummary temperature = readTemperatureSummary();
    const PulseSummary pulse = readPulseSummary();
    const MpuSummary mpuSummary = readMpuSummary();

    printData(timestampMs, temperature, pulse, triggerCapacitive, mpuSummary);

    const bool sent = transmitMeasurements(
        timestampMs,
        temperature,
        pulse,
        triggerCapacitive,
        mpuSummary,
        true
    );
    if (sent) {
        Serial.println("INFO,measurement_sent");
    } else {
        Serial.println("ERROR,measurement_send_failed");
    }
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    delay(300);

    analogReadResolution(12);
#if defined(ARDUINO_ARCH_ESP32)
    analogSetPinAttenuation(THERMISTOR_PIN, ADC_11db);
    analogSetPinAttenuation(PULSE_PIN, ADC_11db);
#endif

    pinMode(THERMISTOR_PIN, INPUT);
    pinMode(PULSE_PIN, INPUT);

    Serial.println("INFO,firmware_started");
    Serial.println("WARNING,thermistor_beta_and_temperature_fit_are_provisional");
    Serial.println("WARNING,accel_yz_calibration_uses_default_scale_and_offset");
    Serial.println("INFO,thermistor_pin,35");
    Serial.println("INFO,pulse_pin,34");
    Serial.println("INFO,capacitive_send_pin,25");
    Serial.println("INFO,capacitive_receive_pin,27");
    Serial.println("INFO,mpu_sda_pin,21");
    Serial.println("INFO,mpu_scl_pin,22");
    Serial.println("INFO,mpu_address,0x68");

    startPulseSamplingTask();

    Wire.begin(MPU_SDA_PIN, MPU_SCL_PIN);
    Serial.println("INFO,i2c_mode,explicit_sda_21_scl_22");
    initializeMpu();
    handleWifi(millis());
    calibrateCapacitiveBaseline();
    handleWifi(millis());
    Serial.println("INFO,waiting_to_start");
}

void loop() {
    const unsigned long now = millis();
    handleWifi(now);

    if (!fieldsPrinted) {
        printFields();
    }

    if (elapsedMillis(now, lastTouchPollMs) < TOUCH_POLL_INTERVAL_MS) {
        return;
    }
    lastTouchPollMs = now;

    const CapacitiveSummary capacitive = readCapacitiveSummary();
    const bool currentTouchState = capacitive.valid && capacitive.touched;

    if (measurementState == WAITING_TO_START) {
        releaseConfirmationCount = 0;

        if (currentTouchState) {
            touchConfirmationCount++;

            if (touchConfirmationCount >= TOUCH_CONFIRMATION_SAMPLES) {
                Serial.println("INFO,start_touch_detected");
                measurementState = WAITING_FOR_START_RELEASE;
                touchConfirmationCount = 0;
                releaseConfirmationCount = 0;
                Serial.println("INFO,waiting_for_start_release");
            }
        } else {
            touchConfirmationCount = 0;
        }
    } else if (measurementState == WAITING_FOR_START_RELEASE) {
        touchConfirmationCount = 0;

        if (!currentTouchState) {
            releaseConfirmationCount++;
            if (releaseConfirmationCount >= RELEASE_CONFIRMATION_SAMPLES) {
                Serial.println("INFO,measurement_session_started");
                measurementState = MEASURING;
                releaseConfirmationCount = 0;
                lastMeasurementMs = now - MEASUREMENT_INTERVAL_MS;
            }
        } else {
            releaseConfirmationCount = 0;
        }
    } else if (measurementState == MEASURING) {
        releaseConfirmationCount = 0;

        if (currentTouchState) {
            touchConfirmationCount++;
            if (touchConfirmationCount >= TOUCH_CONFIRMATION_SAMPLES) {
                Serial.println("INFO,stop_touch_detected");
                Serial.println("INFO,measurement_session_stopped");
                measurementState = WAITING_FOR_STOP_RELEASE;
                touchConfirmationCount = 0;
                releaseConfirmationCount = 0;
                Serial.println("INFO,waiting_for_stop_release");
            }
        } else {
            touchConfirmationCount = 0;
        }

        if (measurementState == MEASURING &&
            elapsedMillis(now, lastMeasurementMs) >= MEASUREMENT_INTERVAL_MS) {
            collectAndSendMeasurement(now, capacitive);
            lastMeasurementMs = now;
        }
    } else if (measurementState == WAITING_FOR_STOP_RELEASE) {
        touchConfirmationCount = 0;

        if (!currentTouchState) {
            releaseConfirmationCount++;
            if (releaseConfirmationCount >= RELEASE_CONFIRMATION_SAMPLES) {
                measurementState = WAITING_TO_START;
                releaseConfirmationCount = 0;
                Serial.println("INFO,ready_for_next_session");
                Serial.println("INFO,waiting_to_start");
            }
        } else {
            releaseConfirmationCount = 0;
        }
    }
}
