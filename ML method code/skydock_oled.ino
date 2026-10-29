// Arduino Nano V3 / R3 clone + 128x64 I2C OLED.
// Default OLED driver is SH1106, common on 1.3-inch modules. If the screen is
// SSD1306, replace the include/class below as described in START_HERE_OLED.txt.
#include <Arduino.h>
#include <U8x8lib.h>

// Nano hardware I2C pins: A4 = SDA, A5 = SCL.
U8X8_SH1106_128X64_NONAME_HW_I2C display(U8X8_PIN_NONE);

char lineBuffer[64];
uint8_t lineLength = 0;

void printLine(uint8_t row, const char *text) {
  char clipped[17];  // 16 characters plus the string terminator.
  uint8_t i = 0;
  while (i < 16 && text[i] != '\0') {
    clipped[i] = text[i];
    ++i;
  }
  clipped[i] = '\0';
  display.drawString(0, row, clipped);
}

void showHold(const char *reason) {
  display.clearDisplay();
  printLine(0, "SKYDOCK GUIDE");
  printLine(1, "----------------");
  printLine(3, "HOLD POSITION");
  printLine(5, reason);
}

void showMessage(char *line) {
  if (strcmp(line, "CENTERED") == 0) {
    display.clearDisplay();
    printLine(0, "SKYDOCK GUIDE");
    printLine(1, "----------------");
    printLine(3, "CENTERED");
    printLine(5, "HOLD POSITION");
    return;
  }
  if (strcmp(line, "HOLD") == 0) {
    showHold("Waiting for view");
    return;
  }
  if (strncmp(line, "GUIDE|", 6) != 0) {
    showHold("Bad camera data");
    return;
  }

  // Incoming example: GUIDE|LEFT:35:mm|UP:42:mm
  char *x = line + 6;
  char *separator = strchr(x, '|');
  if (separator == NULL) {
    showHold("Bad camera data");
    return;
  }
  *separator = '\0';
  char *y = separator + 1;

  display.clearDisplay();
  printLine(0, "SKYDOCK GUIDE");
  printLine(1, "----------------");
  printLine(2, "MOVE DRONE:");
  if (strcmp(x, "-") != 0) printLine(4, x);
  if (strcmp(y, "-") != 0) printLine(6, y);
}

void setup() {
  Serial.begin(115200);
  display.begin();
  display.setFont(u8x8_font_chroma48medium8_r);
  showHold("Waiting for view");
}

void loop() {
  while (Serial.available() > 0) {
    const char c = (char)Serial.read();
    if (c == '\n') {
      lineBuffer[lineLength] = '\0';
      if (lineLength > 0) showMessage(lineBuffer);
      lineLength = 0;
    } else if (c != '\r') {
      if (lineLength < sizeof(lineBuffer) - 1) {
        lineBuffer[lineLength++] = c;
      } else {
        lineLength = 0;  // Discard an overlong line and await the next one.
      }
    }
  }
}
