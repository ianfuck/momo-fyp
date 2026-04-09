#include <Arduino.h>
#include <ESP32Servo.h>
#include <FastLED.h> // 引入 FastLED 函式庫

Servo leftServo;
Servo rightServo;

constexpr int LEFT_PIN = 18;
constexpr int RIGHT_PIN = 19;
constexpr int LED_LEFT_1_PIN = 25;
constexpr int LED_LEFT_2_PIN = 26;
constexpr int LED_RIGHT_1_PIN = 27;
constexpr int LED_RIGHT_2_PIN = 33;
constexpr int SERIAL_BAUD = 115200;

// ================= FastLED 設定 =================
#define NUM_LEDS 30       // ★請改成你每一條燈條實際的燈珠數量★
#define LED_TYPE WS2812B  // ★請確認你的燈條型號，常見為 WS2812B 或 WS2811★
#define COLOR_ORDER GRB   // 如果顏色顯示錯亂，請嘗試改為 RGB

CRGB ledsLeft1[NUM_LEDS];
CRGB ledsLeft2[NUM_LEDS];
CRGB ledsRight1[NUM_LEDS];
CRGB ledsRight2[NUM_LEDS];

// 你想要顯示的顏色，預設為白色 (如果想改顏色，例如改為紅色: CRGB::Red)
CRGB STRIP_COLOR = CRGB::White; 
// ===============================================

float currentLeft = 87.0f;
float currentRight = 96.0f;
float currentLedLeftPct = 0.0f;
float currentLedRightPct = 0.0f;
unsigned long lastCommandAt = 0;

void applyServo(float leftDeg, float rightDeg) {
  currentLeft = constrain(leftDeg, 45.0f, 135.0f);
  currentRight = constrain(rightDeg, 45.0f, 135.0f);
  leftServo.write(currentLeft);
  rightServo.write(currentRight);
}

// 將 0.0~100.0 的百分比轉換為 0~255 的 FastLED 亮度值
int brightnessPctToDuty(float pct) {
  return static_cast<int>(roundf(constrain(pct, 0.0f, 100.0f) * 255.0f / 100.0f));
}

void applyLedBrightness(float leftPct, float rightPct) {
  currentLedLeftPct = constrain(leftPct, 0.0f, 100.0f);
  currentLedRightPct = constrain(rightPct, 0.0f, 100.0f);
  
  // 取得 0-255 的亮度數值
  uint8_t leftVal = brightnessPctToDuty(currentLedLeftPct);
  uint8_t rightVal = brightnessPctToDuty(currentLedRightPct);

  // 填滿燈條顏色，利用 % 運算子等比例縮放 CRGB 亮度 (0 為全暗，255 為最亮)
  fill_solid(ledsLeft1, NUM_LEDS, STRIP_COLOR % leftVal);
  fill_solid(ledsLeft2, NUM_LEDS, STRIP_COLOR % leftVal);
  fill_solid(ledsRight1, NUM_LEDS, STRIP_COLOR % rightVal);
  fill_solid(ledsRight2, NUM_LEDS, STRIP_COLOR % rightVal);

  // 將資料送出到燈條
  FastLED.show();
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
  
  // 綁定 FastLED 腳位與陣列
  FastLED.addLeds<LED_TYPE, LED_LEFT_1_PIN, COLOR_ORDER>(ledsLeft1, NUM_LEDS);
  FastLED.addLeds<LED_TYPE, LED_LEFT_2_PIN, COLOR_ORDER>(ledsLeft2, NUM_LEDS);
  FastLED.addLeds<LED_TYPE, LED_RIGHT_1_PIN, COLOR_ORDER>(ledsRight1, NUM_LEDS);
  FastLED.addLeds<LED_TYPE, LED_RIGHT_2_PIN, COLOR_ORDER>(ledsRight2, NUM_LEDS);
  
  // 設定全域最高亮度 (0-255)，避免耗電過大可以調低
  FastLED.setBrightness(255); 
  
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

  // 若超過 3 秒沒有收到指令，回歸預設狀態並關閉燈光
  if (millis() - lastCommandAt > 3000) {
    applyServo(87.0f, 96.0f);
    applyLedBrightness(0.0f, 0.0f);
  }
}