#include <Arduino.h>
#include <ESP32Servo.h>

Servo leftServo;
Servo rightServo;

constexpr int LEFT_PIN = 18;
constexpr int RIGHT_PIN = 19;
constexpr int SERIAL_BAUD = 115200;

float currentLeft = 85.0f;
float currentRight = 93.0f;
unsigned long lastCommandAt = 0;

void applyServo(float leftDeg, float rightDeg) {
  currentLeft = constrain(leftDeg, 45.0f, 135.0f);
  currentRight = constrain(rightDeg, 45.0f, 135.0f);
  leftServo.write(currentLeft);
  rightServo.write(currentRight);
}

void sendStatus(const char* type, const char* mode) {
  Serial.print("{\"type\":\"");
  Serial.print(type);
  Serial.print("\",\"mode\":\"");
  Serial.print(mode);
  Serial.print("\",\"left_deg\":");
  Serial.print(currentLeft, 2);
  Serial.print(",\"right_deg\":");
  Serial.print(currentRight, 2);
  Serial.println("}");
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  leftServo.attach(LEFT_PIN);
  rightServo.attach(RIGHT_PIN);
  applyServo(90.0f, 90.0f);
  sendStatus("status", "boot");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.indexOf("\"type\":\"servo\"") >= 0) {
      int leftIndex = line.indexOf("\"left_deg\":");
      int rightIndex = line.indexOf("\"right_deg\":");
      if (leftIndex >= 0 && rightIndex >= 0) {
        float left = line.substring(leftIndex + 11).toFloat();
        float right = line.substring(rightIndex + 12).toFloat();
        applyServo(left, right);
        lastCommandAt = millis();
        sendStatus("ack", "track");
      }
    }
  }

  if (millis() - lastCommandAt > 3000) {
    applyServo(85.0f, 93.0f);
  }
}

