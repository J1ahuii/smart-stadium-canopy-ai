#include <Arduino.h>
#include <ArduinoJson.h>
#include "config.h"
#include "sensors.h"
#include "actuator.h"

unsigned long lastSendMs = 0;

void handleIncomingCommand() {
    if (!Serial.available()) return;

    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, line);
    if (err) return;

    const char* cmd = doc["cmd"] | "";
    if (strcmp(cmd, "ESTOP") == 0) {
        bool active = doc["active"] | false;
        emergencyStopActuator(active);
    } else if (strcmp(cmd, "MOVE") == 0) {
        float target = doc["target_pos"] | 0.0f;
        setCanopyPosition(target);
    }
}

void sendTelemetryJson(const SensorPacket& pkt) {
    JsonDocument doc;
    doc["device"] = "esp32_stadium_node";
    doc["temp_c"] = round(pkt.tempC * 10.0f) / 10.0f;
    doc["humidity"] = round(pkt.humidityPct * 10.0f) / 10.0f;
    doc["dht_healthy"] = pkt.dhtHealthy;
    doc["dht_fail_streak"] = pkt.dhtFailStreak;
    doc["rain_wet"] = pkt.rainDetected;
    doc["uv_index"] = round(pkt.uvIndex * 10.0f) / 10.0f;
    doc["light_lux"] = round(pkt.lightLux);
    doc["pir_motion"] = pkt.pirMotion;
    doc["hw_estop"] = pkt.hwEstopPressed;
    doc["confirmed_pos"] = getConfirmedPosition();
    doc["actuator_locked"] = isActuatorLocked();

    serializeJson(doc, Serial);
    Serial.println();
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    initSensors();
    initActuator();
}

void loop() {
    handleIncomingCommand();

    unsigned long now = millis();
    if (now - lastSendMs >= TELEMETRY_INTERVAL_MS) {
        lastSendMs = now;
        SensorPacket pkt = readAllSensors();

        if (pkt.hwEstopPressed) {
            emergencyStopActuator(true);
        }

        sendTelemetryJson(pkt);
    }
}