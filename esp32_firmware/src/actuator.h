#pragma once
#include <Arduino.h>

void initActuator();
void setCanopyPosition(float targetPct);
void emergencyStopActuator(bool active);
float getConfirmedPosition();
bool isActuatorLocked();