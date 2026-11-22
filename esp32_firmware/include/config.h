#pragma once
#include <Arduino.h>

// USB Serial Baud Rate (must match Python PC bridge)
#define SERIAL_BAUD 115200

// I2C Pins for PCA9685 Servo Driver
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define PCA9685_ADDR 0x40
#define CANOPY_SERVO_CH 0      // Channel 0 on PCA9685 (add more if using dual servos)
#define SERVO_MIN_PULSE 120    // ~0 degrees (0% Closed / Fully Open)
#define SERVO_MAX_PULSE 520    // ~180 degrees (100% Closed)

// Local Stadium Sensor Pins
#define DHT_PIN 4
#define DHT_TYPE DHT11
#define RAIN_DIGITAL_PIN 27    // Digital Rain Module (LOW = Wet on most modules)
#define UV_ANALOG_PIN 34       // ADC1 Analog UV Sensor
#define LDR_ANALOG_PIN 35      // ADC1 Analog Light Sensor
#define PIR_MOTION_PIN 26      // PIR Spectator Occupancy Sensor
#define ESTOP_BUTTON_PIN 25    // Physical Emergency Stop (Active LOW with PULLUP)

// Telemetry Interval
#define TELEMETRY_INTERVAL_MS 1000