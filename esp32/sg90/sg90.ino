#include <Arduino.h>
#include <ESP32Servo.h>

Servo leftServo;
Servo rightServo;

constexpr int LEFT_PIN = 18;
constexpr int RIGHT_PIN = 19;
constexpr int LED_LEFT_1_PIN = 25;
constexpr int LED_LEFT_2_PIN = 26;
constexpr int LED_RIGHT_1_PIN = 27;
constexpr int LED_RIGHT_2_PIN = 33;
constexpr int LED_LEFT_1_CHANNEL = 0;
constexpr int LED_LEFT_2_CHANNEL = 1;
constexpr int LED_RIGHT_1_CHANNEL = 2;
constexpr int LED_RIGHT_2_CHANNEL = 3;
constexpr int LED_PWM_FREQ = 5000;
constexpr int LED_PWM_RESOLUTION = 8;
constexpr int SERIAL_BAUD = 115200;

float currentLeft = 93.0f;
float currentRight = 85.0f;
float currentLedLeftPct = 0.0f;
float currentLedRightPct = 0.0f;
unsigned long lastCommandAt = 0;

void applyServo(float leftDeg, float rightDeg) {
  currentLeft = constrain(leftDeg, 45.0f, 135.0f);
  currentRight = constrain(rightDeg, 45.0f, 135.0f);
  leftServo.write(currentLeft);
  rightServo.write(currentRight);
}

int brightnessPctToDuty(float pct) {
  return static_cast<int>(roundf(constrain(pct, 0.0f, 100.0f) * 255.0f / 100.0f));
}

void applyLedBrightness(float leftPct, float rightPct) {
  currentLedLeftPct = constrain(leftPct, 0.0f, 100.0f);
  currentLedRightPct = constrain(rightPct, 0.0f, 100.0f);
  ledcWrite(LED_LEFT_1_CHANNEL, brightnessPctToDuty(currentLedLeftPct));
  ledcWrite(LED_LEFT_2_CHANNEL, brightnessPctToDuty(currentLedLeftPct));
  ledcWrite(LED_RIGHT_1_CHANNEL, brightnessPctToDuty(currentLedRightPct));
  ledcWrite(LED_RIGHT_2_CHANNEL, brightnessPctToDuty(currentLedRightPct));
}

float extractFloatField(const String& line, const char* key, float fallback) {
  String needle = String("\"") + key + "\":";
  int start = line.indexOf(needle);
  if (start < 0) {
    return fallback;
  }
  start += needle.length();
  int end = start;
  while (end < line.length()) {
    const char c = line.charAt(end);
    const bool isNumberChar = (c >= '0' && c <= '9') || c == '-' || c == '.';
    if (!isNumberChar) {
      break;
    }
    end += 1;
  }
  return line.substring(start, end).toFloat();
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
  Serial.print(",\"led_left_pct\":");
  Serial.print(currentLedLeftPct, 2);
  Serial.print(",\"led_right_pct\":");
  Serial.print(currentLedRightPct, 2);
  Serial.println("}");
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  leftServo.attach(LEFT_PIN);
  rightServo.attach(RIGHT_PIN);
  ledcSetup(LED_LEFT_1_CHANNEL, LED_PWM_FREQ, LED_PWM_RESOLUTION);
  ledcSetup(LED_LEFT_2_CHANNEL, LED_PWM_FREQ, LED_PWM_RESOLUTION);
  ledcSetup(LED_RIGHT_1_CHANNEL, LED_PWM_FREQ, LED_PWM_RESOLUTION);
  ledcSetup(LED_RIGHT_2_CHANNEL, LED_PWM_FREQ, LED_PWM_RESOLUTION);
  ledcAttachPin(LED_LEFT_1_PIN, LED_LEFT_1_CHANNEL);
  ledcAttachPin(LED_LEFT_2_PIN, LED_LEFT_2_CHANNEL);
  ledcAttachPin(LED_RIGHT_1_PIN, LED_RIGHT_1_CHANNEL);
  ledcAttachPin(LED_RIGHT_2_PIN, LED_RIGHT_2_CHANNEL);
  applyServo(90.0f, 90.0f);
  applyLedBrightness(0.0f, 0.0f);
  sendStatus("status", "boot");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.indexOf("\"type\":\"servo\"") >= 0) {
      if (line.indexOf("\"left_deg\":") >= 0 && line.indexOf("\"right_deg\":") >= 0) {
        float left = extractFloatField(line, "left_deg", currentLeft);
        float right = extractFloatField(line, "right_deg", currentRight);
        float ledLeftPct = extractFloatField(line, "led_left_pct", currentLedLeftPct);
        float ledRightPct = extractFloatField(line, "led_right_pct", currentLedRightPct);
        applyServo(left, right);
        applyLedBrightness(ledLeftPct, ledRightPct);
        lastCommandAt = millis();
        sendStatus("ack", "track");
      }
    }
  }

  if (millis() - lastCommandAt > 3000) {
    applyServo(93.0f, 85.0f);
    applyLedBrightness(0.0f, 0.0f);
  }
}
