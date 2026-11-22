#include "sensors.h"
#include "config.h"
#include <DHT.h>

static DHT dht(DHT_PIN, DHT_TYPE);
static float lastValidTemp = 30.0f;
static float lastValidHum = 70.0f;
static int dhtFailStreak = 0;

void initSensors() {
    dht.begin();
    pinMode(RAIN_DIGITAL_PIN, INPUT);
    pinMode(PIR_MOTION_PIN, INPUT);
    pinMode(ESTOP_BUTTON_PIN, INPUT_PULLUP);
}

SensorPacket readAllSensors() {
    SensorPacket pkt;

    // 1. DHT11 with 3-strike fault logic matching dashboard
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (isnan(t) || isnan(h)) {
        dhtFailStreak++;
    } else {
        dhtFailStreak = 0;
        lastValidTemp = t;
        lastValidHum = h;
    }
    pkt.tempC = lastValidTemp;
    pkt.humidityPct = lastValidHum;
    pkt.dhtFailStreak = dhtFailStreak;
    pkt.dhtHealthy = (dhtFailStreak < 3);

    // 2. Digital Rain Sensor (LOW = wet)
    pkt.rainDetected = (digitalRead(RAIN_DIGITAL_PIN) == LOW);

    // 3. Analog UV & LDR (12-bit ADC: 0 - 4095)
    int rawUv = analogRead(UV_ANALOG_PIN);
    int rawLdr = analogRead(LDR_ANALOG_PIN);
    pkt.uvIndex = (rawUv / 4095.0f) * 11.0f;
    pkt.lightLux = ((4095 - rawLdr) / 4095.0f) * 1000.0f;

    // 4. PIR & Physical E-Stop
    pkt.pirMotion = (digitalRead(PIR_MOTION_PIN) == HIGH);
    pkt.hwEstopPressed = (digitalRead(ESTOP_BUTTON_PIN) == LOW);

    return pkt;
}