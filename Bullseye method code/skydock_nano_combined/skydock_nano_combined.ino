// SkyDock combined bench-test firmware for Arduino Nano V3/R3 clone.
// Reads four microswitches and one HC-SR04, reports JSON to Python, receives
// camera guidance from Python, and displays guidance/landing status on an OLED.
// Default OLED driver: SH1106. Change the display class if the display is SSD1306.

#include <Arduino.h>
#include <U8x8lib.h>
#include <string.h>

U8X8_SH1106_128X64_NONAME_HW_I2C display(U8X8_PIN_NONE);  // A4 SDA, A5 SCL

const uint8_t SWITCH_PINS[] = {2, 3, 4, 5};
const uint8_t SWITCH_COUNT = sizeof(SWITCH_PINS) / sizeof(SWITCH_PINS[0]);
const uint8_t TRIG_PIN = 9;
const uint8_t ECHO_PIN = 10;
const uint32_t SAMPLE_PERIOD_MS = 100;
const float LANDING_DISTANCE_CM = 5.0f;
const uint8_t MIN_PRESSED_SWITCHES = 2;
const uint32_t SERIAL_BAUD = 115200;

char inputLine[64];
uint8_t inputLength = 0;
char guidanceX[17] = "HOLD POSITION";
char guidanceY[17] = "WAIT FOR VIEW";
float distanceCm = -1.0f;
uint8_t pressedCount = 0;
bool landed = false;

void printLine(uint8_t row, const char *text) {
  char clipped[17];
  uint8_t i = 0;
  while (i < 16 && text[i] != '\0') {
    clipped[i] = text[i];
    ++i;
  }
  clipped[i] = '\0';
  display.drawString(0, row, clipped);
}

void updateDisplay() {
  display.clearDisplay();
  printLine(0, "SKYDOCK GUIDE");
  printLine(1, "----------------");
  printLine(2, "MOVE DRONE:");
  printLine(4, guidanceX);
  printLine(5, guidanceY);
  printLine(7, landed ? "LANDING: YES" : "LANDING: NO");
}

void copyClipped(char *destination, size_t destinationSize, const char *source) {
  if (destinationSize == 0) return;
  strncpy(destination, source, destinationSize - 1);
  destination[destinationSize - 1] = '\0';
}

void handleGuidance(char *line) {
  if (strcmp(line, "HOLD") == 0) {
    copyClipped(guidanceX, sizeof(guidanceX), "HOLD POSITION");
    copyClipped(guidanceY, sizeof(guidanceY), "WAIT FOR VIEW");
    updateDisplay();
    return;
  }
  if (strcmp(line, "CENTERED") == 0) {
    copyClipped(guidanceX, sizeof(guidanceX), "CENTERED");
    copyClipped(guidanceY, sizeof(guidanceY), "HOLD POSITION");
    updateDisplay();
    return;
  }

  // Python versions may send GUIDE|LEFT:35:mm|UP:42:mm or GUIDE,GO LEFT.
  if (strncmp(line, "GUIDE|", 6) == 0) {
    char *x = line + 6;
    char *separator = strchr(x, '|');
    if (separator == NULL) {
      copyClipped(guidanceX, sizeof(guidanceX), "BAD CAMERA DATA");
      guidanceY[0] = '\0';
    } else {
      *separator = '\0';
      char *y = separator + 1;
      copyClipped(guidanceX, sizeof(guidanceX), strcmp(x, "-") == 0 ? "" : x);
      copyClipped(guidanceY, sizeof(guidanceY), strcmp(y, "-") == 0 ? "" : y);
    }
    updateDisplay();
    return;
  }
  if (strncmp(line, "GUIDE,", 6) == 0) {
    copyClipped(guidanceX, sizeof(guidanceX), line + 6);
    guidanceY[0] = '\0';
    updateDisplay();
    return;
  }

  copyClipped(guidanceX, sizeof(guidanceX), "BAD CAMERA DATA");
  guidanceY[0] = '\0';
  updateDisplay();
}

void readPythonCommands() {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\n') {
      inputLine[inputLength] = '\0';
      if (inputLength > 0) handleGuidance(inputLine);
      inputLength = 0;
    } else if (c != '\r') {
      if (inputLength < sizeof(inputLine) - 1) {
        inputLine[inputLength++] = c;
      } else {
        inputLength = 0;  // Discard an overlong command.
      }
    }
  }
}

float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  const unsigned long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);
  if (duration == 0) return -1.0f;
  return duration * 0.0343f / 2.0f;
}

void readSwitches() {
  pressedCount = 0;
  for (uint8_t i = 0; i < SWITCH_COUNT; ++i) {
    if (digitalRead(SWITCH_PINS[i]) == LOW) ++pressedCount;
  }
}

void sendDataLine() {
  // Keep this as one JSON object per line; Python can parse each line directly.
  Serial.print(F("{\"distance_cm\":"));
  if (distanceCm < 0.0f) Serial.print(F("null"));
  else Serial.print(distanceCm, 1);
  Serial.print(F(",\"pressed_count\":"));
  Serial.print(pressedCount);
  Serial.print(F(",\"landed\":"));
  Serial.print(landed ? F("true") : F("false"));
  for (uint8_t i = 0; i < SWITCH_COUNT; ++i) {
    Serial.print(F(",\"sw"));
    Serial.print(SWITCH_PINS[i]);
    Serial.print(F("\":"));
    Serial.print(digitalRead(SWITCH_PINS[i]) == LOW ? 1 : 0);
  }
  Serial.println(F("}"));
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  for (uint8_t i = 0; i < SWITCH_COUNT; ++i) {
    pinMode(SWITCH_PINS[i], INPUT_PULLUP);
  }
  display.begin();
  display.setFont(u8x8_font_chroma48medium8_r);
  updateDisplay();
}

void loop() {
  readPythonCommands();
  static uint32_t lastSample = 0;
  const uint32_t now = millis();
  if (now - lastSample >= SAMPLE_PERIOD_MS) {
    lastSample = now;
    distanceCm = readDistanceCm();
    readSwitches();
    landed = distanceCm >= 0.0f && distanceCm < LANDING_DISTANCE_CM &&
             pressedCount >= MIN_PRESSED_SWITCHES;
    updateDisplay();
    sendDataLine();
  }
}
