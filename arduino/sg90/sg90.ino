/**
 * Dual SG90 eye servos — TRACK / IDLE / FRENZY / DEAD
 * Serial line: "TRACK 90.0 88.5\n" | "IDLE ..." | "FRENZY ..." | "DEAD ..."
 * PWM pins: LEFT 9, RIGHT 10 (Arduino Uno)
 */
#include <Servo.h>

Servo leftEye;
Servo rightEye;

const int PIN_LEFT = 9;
const int PIN_RIGHT = 10;

String buf;
unsigned long frenzyEnd = 0;
unsigned long deadUntil = 0;
float targetL = 90, targetR = 90;
float curL = 90, curR = 90;
const float EASE = 0.35f;
const int FRENZY_MS = 2500;

void setup() {
  Serial.begin(115200);
  leftEye.attach(PIN_LEFT);
  rightEye.attach(PIN_RIGHT);
  leftEye.write(90);
  rightEye.write(90);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      handleLine(buf);
      buf = "";
    } else if (c != '\r') {
      buf += c;
    }
  }

  unsigned long now = millis();
  if (now < deadUntil) {
    curL += (90 - curL) * EASE;
    curR += (90 - curR) * EASE;
    leftEye.write((int)curL);
    rightEye.write((int)curR);
    return;
  }

  if (now < frenzyEnd) {
    curL = 90 + random(-18, 18);
    curR = 90 + random(-18, 18);
    leftEye.write((int)curL);
    rightEye.write((int)curR);
    delay(40);
    return;
  }

  curL += (targetL - curL) * EASE;
  curR += (targetR - curR) * EASE;
  leftEye.write((int)curL);
  rightEye.write((int)curR);
  delay(15);
}

void handleLine(String s) {
  s.trim();
  if (s.length() == 0) return;

  int sp = s.indexOf(' ');
  if (sp < 0) return;
  String mode = s.substring(0, sp);
  String rest = s.substring(sp + 1);
  int sp2 = rest.indexOf(' ');
  if (sp2 < 0) return;
  float a = rest.substring(0, sp2).toFloat();
  float b = rest.substring(sp2 + 1).toFloat();

  if (mode == "DEAD") {
    deadUntil = millis() + 10000;
    frenzyEnd = 0;
    targetL = a;
    targetR = b;
    return;
  }
  if (mode == "FRENZY") {
    frenzyEnd = millis() + FRENZY_MS;
    targetL = a;
    targetR = b;
    return;
  }
  deadUntil = 0;
  frenzyEnd = 0;
  targetL = constrain(a, 0, 180);
  targetR = constrain(b, 0, 180);
}
