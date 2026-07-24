#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>

// Confirmed hardware connections:
// - Move the LM358 thermistor output wire from GPIO23 to GPIO35.
// - GPIO35 is used only as an analog input.
// - Pulse output remains on GPIO34.
// - Capacitive circuit remains GPIO25 through 1 MΩ to GPIO27 and foil.
// - MPU-6050 communicates over I2C at address 0x68.
const int THERMISTOR_PIN = 35;   // LM358-buffered thermistor analog output
const int PULSE_PIN = 34;        // Pulse-sensor analog signal
const int CAP_SEND_PIN = 25;     // One side of the 1-megaohm resistor
const int CAP_RECEIVE_PIN = 27;  // Other resistor side plus foil pad
const uint8_t MPU_ADDRESS = 0x68;

const int THERMISTOR_SAMPLES = 16;
const int PULSE_SAMPLES = 100;
const int PULSE_SAMPLE_DELAY_MS = 5;
const int CAPACITIVE_SAMPLES = 30;

MPU6050 mpu(MPU_ADDRESS);
bool mpuConnected = false;
bool fieldsPrinted = false;

struct AnalogSummary {
    bool valid;
    float averageRaw;
    float averageMilliVolts;
};

struct PulseSummary {
    float average;
    int minimum;
    int maximum;
    int change;
};

struct CapacitiveSummary {
    unsigned long averageUs;
    unsigned long minimumUs;
    unsigned long maximumUs;
};

unsigned long readCapacitiveSensor() {
    const unsigned long TIMEOUT_US = 30000;

    pinMode(CAP_SEND_PIN, OUTPUT);
    digitalWrite(CAP_SEND_PIN, LOW);

    pinMode(CAP_RECEIVE_PIN, OUTPUT);
    digitalWrite(CAP_RECEIVE_PIN, LOW);

    delayMicroseconds(10);

    pinMode(CAP_RECEIVE_PIN, INPUT);
    digitalWrite(CAP_SEND_PIN, HIGH);

    const unsigned long startTime = micros();

    while (digitalRead(CAP_RECEIVE_PIN) == LOW) {
        if (micros() - startTime >= TIMEOUT_US) {
            break;
        }
    }

    const unsigned long chargeTime = micros() - startTime;

    digitalWrite(CAP_SEND_PIN, LOW);
    pinMode(CAP_RECEIVE_PIN, OUTPUT);
    digitalWrite(CAP_RECEIVE_PIN, LOW);

    return chargeTime;
}

uint32_t readMilliVoltsCompat(int pin, int rawValue) {
#if defined(ARDUINO_ARCH_ESP32)
    return analogReadMilliVolts(pin);
#else
    return static_cast<uint32_t>((static_cast<float>(rawValue) / 4095.0f) * 3300.0f);
#endif
}

AnalogSummary readThermistorSummary() {
    uint32_t rawSum = 0;
    uint32_t milliVoltSum = 0;
    int validCount = 0;

    for (int i = 0; i < THERMISTOR_SAMPLES; i++) {
        const int raw = analogRead(THERMISTOR_PIN);
        if (raw > 0 && raw < 4095) {
            rawSum += static_cast<uint32_t>(raw);
            milliVoltSum += readMilliVoltsCompat(THERMISTOR_PIN, raw);
            validCount++;
        }
        delayMicroseconds(250);
    }

    if (validCount == 0) {
        return {false, 0.0f, 0.0f};
    }

    return {
        true,
        static_cast<float>(rawSum) / static_cast<float>(validCount),
        static_cast<float>(milliVoltSum) / static_cast<float>(validCount),
    };
}

PulseSummary readPulseSummary() {
    uint32_t sum = 0;
    int minimumValue = 4095;
    int maximumValue = 0;

    for (int i = 0; i < PULSE_SAMPLES; i++) {
        const int raw = analogRead(PULSE_PIN);
        sum += static_cast<uint32_t>(raw);
        if (raw < minimumValue) {
            minimumValue = raw;
        }
        if (raw > maximumValue) {
            maximumValue = raw;
        }
        delay(PULSE_SAMPLE_DELAY_MS);
    }

    return {
        static_cast<float>(sum) / static_cast<float>(PULSE_SAMPLES),
        minimumValue,
        maximumValue,
        maximumValue - minimumValue,
    };
}

CapacitiveSummary readCapacitiveSummary() {
    unsigned long sum = 0;
    unsigned long minimumValue = ULONG_MAX;
    unsigned long maximumValue = 0;

    for (int i = 0; i < CAPACITIVE_SAMPLES; i++) {
        const unsigned long reading = readCapacitiveSensor();
        sum += reading;
        if (reading < minimumValue) {
            minimumValue = reading;
        }
        if (reading > maximumValue) {
            maximumValue = reading;
        }
        delayMicroseconds(250);
    }

    return {
        sum / static_cast<unsigned long>(CAPACITIVE_SAMPLES),
        minimumValue,
        maximumValue,
    };
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

void printFields() {
    Serial.println(
        "FIELDS,timestamp_ms,thermistor_adc_raw,thermistor_millivolts,"
        "pulse_average,pulse_minimum,pulse_maximum,pulse_change,"
        "capacitive_average_us,capacitive_minimum_us,capacitive_maximum_us,"
        "mpu_connected,accel_x_raw,accel_y_raw,accel_z_raw,"
        "gyro_x_raw,gyro_y_raw,gyro_z_raw"
    );
    fieldsPrinted = true;
}

void printFloatOrBlank(bool valid, float value, int decimals) {
    if (valid) {
        Serial.print(value, decimals);
    }
}

void printMpuValues() {
    int16_t accelX = 0;
    int16_t accelY = 0;
    int16_t accelZ = 0;
    int16_t gyroX = 0;
    int16_t gyroY = 0;
    int16_t gyroZ = 0;

    if (mpuConnected) {
        mpu.getMotion6(&accelX, &accelY, &accelZ, &gyroX, &gyroY, &gyroZ);
        Serial.print(accelX);
        Serial.print(',');
        Serial.print(accelY);
        Serial.print(',');
        Serial.print(accelZ);
        Serial.print(',');
        Serial.print(gyroX);
        Serial.print(',');
        Serial.print(gyroY);
        Serial.print(',');
        Serial.print(gyroZ);
    } else {
        Serial.print(",,,,,");
    }
}

void setup() {
    Serial.begin(115200);
    delay(300);

    analogReadResolution(12);
#if defined(ARDUINO_ARCH_ESP32)
    analogSetPinAttenuation(THERMISTOR_PIN, ADC_11db);
    analogSetPinAttenuation(PULSE_PIN, ADC_11db);
#endif

    pinMode(THERMISTOR_PIN, INPUT);
    pinMode(PULSE_PIN, INPUT);

    Serial.println("INFO,firmware_started");
    Serial.println("INFO,thermistor_pin,35");
    Serial.println("INFO,pulse_pin,34");
    Serial.println("INFO,capacitive_send_pin,25");
    Serial.println("INFO,capacitive_receive_pin,27");
    Serial.println("INFO,mpu_address,0x68");

    Wire.begin();
    Serial.println("INFO,i2c_mode,default");
    initializeMpu();
}

void loop() {
    if (!fieldsPrinted) {
        printFields();
    }

    const unsigned long timestampMs = millis();
    const AnalogSummary thermistor = readThermistorSummary();
    const PulseSummary pulse = readPulseSummary();
    const CapacitiveSummary capacitive = readCapacitiveSummary();

    Serial.print("DATA,");
    Serial.print(timestampMs);
    Serial.print(',');
    printFloatOrBlank(thermistor.valid, thermistor.averageRaw, 2);
    Serial.print(',');
    printFloatOrBlank(thermistor.valid, thermistor.averageMilliVolts, 2);
    Serial.print(',');
    Serial.print(pulse.average, 2);
    Serial.print(',');
    Serial.print(pulse.minimum);
    Serial.print(',');
    Serial.print(pulse.maximum);
    Serial.print(',');
    Serial.print(pulse.change);
    Serial.print(',');
    Serial.print(capacitive.averageUs);
    Serial.print(',');
    Serial.print(capacitive.minimumUs);
    Serial.print(',');
    Serial.print(capacitive.maximumUs);
    Serial.print(',');
    Serial.print(mpuConnected ? 1 : 0);
    Serial.print(',');
    printMpuValues();
    Serial.println();
}
