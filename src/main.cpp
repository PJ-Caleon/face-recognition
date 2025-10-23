#include <Arduino.h>

// Pin definitions
const int greenPin = D4;
const int redPin = D7;
const int ledPin = LED_BUILTIN;  // built-in LED for timer

unsigned long lastDetectTime = 0;
bool recognized = false;

void setup() {
  pinMode(greenPin, OUTPUT);
  pinMode(redPin, OUTPUT);
  pinMode(ledPin, OUTPUT);

  digitalWrite(greenPin, LOW);
  digitalWrite(redPin, LOW);
  digitalWrite(ledPin, LOW);

  Serial.begin(115200);
}

void loop() {
  // Check for serial input
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == "HELLO_PJ") {
      // Green LED on
      digitalWrite(greenPin, HIGH);
      digitalWrite(redPin, LOW);
      recognized = true;
      lastDetectTime = millis();
    } 
    else if (input == "UNKNOWN") {
      // Red LED on
      digitalWrite(greenPin, LOW);
      digitalWrite(redPin, HIGH);
      recognized = false;
      lastDetectTime = millis();
    }
  }

  // If no recognition after 5 seconds → blink red rapidly
  if (millis() - lastDetectTime > 5000 && !recognized) {
    for (int i = 0; i < 20; i++) {
      digitalWrite(redPin, HIGH);
      delay(100);
      digitalWrite(redPin, LOW);
      delay(100);
    }
    lastDetectTime = millis();
  }

  // Built-in LED timer pulse every second
  digitalWrite(ledPin, HIGH);
  delay(500);
  digitalWrite(ledPin, LOW);
  delay(500);
}
