#pragma once
#include <Arduino.h>

struct SensorPacket {
    float tempC;
    float humidityPct;
    bool dhtHealthy;
    int dhtFailStreak;
    bool rainDetected;
    float uvIndex;
    float lightLux;
    bool pirMotion;
    bool hwEstopPressed;
};

void initSensors();
SensorPacket readAllSensors();