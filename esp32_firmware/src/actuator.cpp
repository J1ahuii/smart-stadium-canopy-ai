#include "actuator.h"
#include "config.h"
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

static Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_ADDR);
static float currentPosPct = 0.0f;
static bool estopLocked = false;

void initActuator() {
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    pwm.begin();
    pwm.setOscillatorFrequency(27000000);
    pwm.setPWMFreq(50); // Standard 50 Hz servo frequency
    delay(10);
    setCanopyPosition(0.0f);
}

void setCanopyPosition(float targetPct) {
    if (estopLocked) return;

    targetPct = constrain(targetPct, 0.0f, 100.0f);
    uint16_t pulse = map((long)targetPct, 0, 100, SERVO_MIN_PULSE, SERVO_MAX_PULSE);
    pwm.setPWM(CANOPY_SERVO_CH, 0, pulse);
    currentPosPct = targetPct;
}

void emergencyStopActuator(bool active) {
    estopLocked = active;
    if (estopLocked) {
        // Cut PWM signal on emergency stop so servo stops torque immediately
        pwm.setPWM(CANOPY_SERVO_CH, 0, 0);
    }
}

float getConfirmedPosition() {
    return currentPosPct;
}

bool isActuatorLocked() {
    return estopLocked;
}