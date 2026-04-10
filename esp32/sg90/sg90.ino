#include <Arduino.h>
#include <ESP32Servo.h>
#include <Adafruit_NeoPixel.h>

Servo leftServo;
Servo rightServo;

constexpr int LEFT_PIN = 18;
constexpr int RIGHT_PIN = 19;
constexpr int LED_LEFT_1_PIN = 25;
constexpr int LED_LEFT_2_PIN = 26;
constexpr int LED_RIGHT_1_PIN = 27;
constexpr int LED_RIGHT_2_PIN = 33;
constexpr int LED_ALWAYS_ON_1_PIN = 16;
constexpr int LED_ALWAYS_ON_2_PIN = 17;
constexpr int SERIAL_BAUD = 115200;
constexpr unsigned long DEFAULT_LED_SIGNAL_LOSS_FADE_OUT_MS = 3000;

// ================= NeoPixel 設定 =================
#define NUM_LEDS 30  // ★請改成你每一條燈條實際的燈珠數量★

Adafruit_NeoPixel ledsLeft1(NUM_LEDS, LED_LEFT_1_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel ledsLeft2(NUM_LEDS, LED_LEFT_2_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel ledsRight1(NUM_LEDS, LED_RIGHT_1_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel ledsRight2(NUM_LEDS, LED_RIGHT_2_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel ledsAlwaysOn1(NUM_LEDS, LED_ALWAYS_ON_1_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel ledsAlwaysOn2(NUM_LEDS, LED_ALWAYS_ON_2_PIN, NEO_GRB + NEO_KHZ800);

constexpr uint8_t STRIP_RED = 255;
constexpr uint8_t STRIP_GREEN = 255;
constexpr uint8_t STRIP_BLUE = 255;
// ================================================

float currentLeft = 87.0f;
float currentRight = 96.0f;
float currentLedLeftPct = 0.0f;
float currentLedRightPct = 0.0f;
float lastCommandLedLeftPct = 0.0f;
float lastCommandLedRightPct = 0.0f;
unsigned long lastCommandAt = 0;
unsigned long ledSignalLossFadeOutMs = DEFAULT_LED_SIGNAL_LOSS_FADE_OUT_MS;

void applyServo(float leftDeg, float rightDeg) {
  currentLeft = constrain(leftDeg, 45.0f, 135.0f);
  currentRight = constrain(rightDeg, 45.0f, 135.0f);
  leftServo.write(currentLeft);
  rightServo.write(currentRight);
}

int brightnessPctToDuty(float pct) {
  return static_cast<int>(roundf(constrain(pct, 0.0f, 100.0f) * 255.0f / 100.0f));
}

uint32_t scaledStripColor(Adafruit_NeoPixel& strip, uint8_t brightness) {
  return strip.Color(
      static_cast<uint8_t>((static_cast<uint16_t>(STRIP_RED) * brightness) / 255),
      static_cast<uint8_t>((static_cast<uint16_t>(STRIP_GREEN) * brightness) / 255),
      static_cast<uint8_t>((static_cast<uint16_t>(STRIP_BLUE) * brightness) / 255));
}

void fillStrip(Adafruit_NeoPixel& strip, uint8_t brightness) {
  strip.fill(scaledStripColor(strip, brightness));
  strip.show();
}

void renderLedBrightness(float leftPct, float rightPct) {
  currentLedLeftPct = constrain(leftPct, 0.0f, 100.0f);
  currentLedRightPct = constrain(rightPct, 0.0f, 100.0f);
  
  // 取得 0-255 的亮度數值
  uint8_t leftVal = brightnessPctToDuty(currentLedLeftPct);
  uint8_t rightVal = brightnessPctToDuty(currentLedRightPct);

  fillStrip(ledsLeft1, leftVal);
  fillStrip(ledsLeft2, leftVal);
  fillStrip(ledsRight1, rightVal);
  fillStrip(ledsRight2, rightVal);
  fillStrip(ledsAlwaysOn1, 255);
  fillStrip(ledsAlwaysOn2, 255);
}

void applyLedBrightness(float leftPct, float rightPct) {
  lastCommandLedLeftPct = constrain(leftPct, 0.0f, 100.0f);
  lastCommandLedRightPct = constrain(rightPct, 0.0f, 100.0f);
  renderLedBrightness(lastCommandLedLeftPct, lastCommandLedRightPct);
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
  
  ledsLeft1.begin();
  ledsLeft2.begin();
  ledsRight1.begin();
  ledsRight2.begin();
  ledsAlwaysOn1.begin();
  ledsAlwaysOn2.begin();
  
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
        ledSignalLossFadeOutMs = static_cast<unsigned long>(
            max(0.0f, extractFloatField(line, "led_signal_loss_fade_out_ms", static_cast<float>(ledSignalLossFadeOutMs))));
        
        applyServo(left, right);
        applyLedBrightness(ledLeftPct, ledRightPct);
        
        lastCommandAt = millis();
        sendStatus("ack", "track");
      }
    }
  }

  if (lastCommandAt > 0) {
    unsigned long silentMs = millis() - lastCommandAt;
    if (ledSignalLossFadeOutMs == 0 || silentMs >= ledSignalLossFadeOutMs) {
      applyServo(87.0f, 96.0f);
      renderLedBrightness(0.0f, 0.0f);
    } else {
      float fadeRatio = static_cast<float>(silentMs) / static_cast<float>(ledSignalLossFadeOutMs);
      renderLedBrightness(lastCommandLedLeftPct * (1.0f - fadeRatio), lastCommandLedRightPct * (1.0f - fadeRatio));
    }
  }
}
