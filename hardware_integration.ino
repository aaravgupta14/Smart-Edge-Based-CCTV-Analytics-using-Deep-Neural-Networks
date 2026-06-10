String msg="";

void clearLEDs() {
  digitalWrite(2,LOW);
  digitalWrite(3,LOW);
  digitalWrite(4,LOW);
  digitalWrite(5,LOW);
}

void setup() {

  Serial.begin(9600);

  pinMode(2,OUTPUT);
  pinMode(3,OUTPUT);
  pinMode(4,OUTPUT);
  pinMode(5,OUTPUT);

  clearLEDs();
}

void loop() {

  if(Serial.available()) {

    msg = Serial.readStringUntil('\n');
    msg.trim();

    clearLEDs();
  if(msg=="AARAV")
    digitalWrite(2,HIGH);
  else if(msg=="ANSHU")
    digitalWrite(3,HIGH);
  else if(msg=="RISHU")
    digitalWrite(4,HIGH);
  else if(msg=="NAVYA")
    digitalWrite(5,HIGH);
  else if(msg=="UNKNOWN") {
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