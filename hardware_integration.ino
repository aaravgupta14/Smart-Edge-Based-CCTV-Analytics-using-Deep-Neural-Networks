#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

String msg="";

void clearLEDs() {
  digitalWrite(2,LOW);
  digitalWrite(3,LOW);
  digitalWrite(4,LOW);
  digitalWrite(5,LOW);
}

void showWelcome(String name) {
  lcd.clear();

  lcd.setCursor(0,0);
  lcd.print("WELCOME");

  lcd.setCursor(0,1);
  lcd.print(name);
}

void setup() {

  Serial.begin(9600);

  pinMode(2,OUTPUT);
  pinMode(3,OUTPUT);
  pinMode(4,OUTPUT);
  pinMode(5,OUTPUT);

  clearLEDs();

  lcd.init();
  lcd.backlight();

  lcd.setCursor(0,0);
  lcd.print("VisionGuard");

  lcd.setCursor(0,1);
  lcd.print("Ready");
}

void loop() {

  if(Serial.available()) {

    msg = Serial.readStringUntil('\n');
    msg.trim();

    clearLEDs();

    if(msg=="AARAV") {
      digitalWrite(2,HIGH);
      showWelcome("AARAV");
    }

    else if(msg=="ANSHU") {
      digitalWrite(3,HIGH);
      showWelcome("ANSHU");
    }

    else if(msg=="RISHU") {
      digitalWrite(4,HIGH);
      showWelcome("RISHU");
    }

    else if(msg=="NAVYA") {
      digitalWrite(5,HIGH);
      showWelcome("NAVYA");
    }

    else if(msg=="UNKNOWN") {

      lcd.clear();

      lcd.setCursor(0,0);
      lcd.print("UNKNOWN");

      lcd.setCursor(0,1);
      lcd.print("IMAGE SAVED");

      for(int i=0;i<3;i++) {

        digitalWrite(2,HIGH);
        digitalWrite(3,HIGH);
        digitalWrite(4,HIGH);
        digitalWrite(5,HIGH);

        delay(200);

        clearLEDs();

        delay(200);
      }
    }
  }
}
