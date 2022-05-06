#include <Arduino.h>
#include <Servo.h>
#include <cvzone.h>

SerialData serialData(1, 1); // (numOfValReceived from python, digitPerValRec from python)
int valsRec[1]; //array of int with size numOfValRec

Servo servothumb;
Servo servoindex;
Servo servomiddle;
Servo servoring;
Servo servopinky;

void setup() {
  serialData.begin(9600);
  Serial.begin(9600);
  servothumb.attach(9);
  servoindex.attach(10);
  servomiddle.attach(11);
  servopinky.attach(12);
  servoring.attach(13);
}

void loop() {
  serialData.Get(valsRec);
  Serial.println(valsRec[0]);
  if (valsRec[0] == 1) {
    servothumb.write(180);
    Serial.println("Thumb Down");
  } else {
    servothumb.write(0);
    Serial.println("Thumb Up");
  }

  if (valsRec[0] == 2) {
    servoindex.write(180);
    Serial.println("Index Down");
  } else {
    servoindex.write(0);
    Serial.println("Index Up");
  }

  if (valsRec[0] == 3) {
    servomiddle.write(180);
    Serial.println("Middle Down");


  } else {
    servomiddle.write(0);
    Serial.println("Middle Up");

  }

  if (valsRec[0] == 4) {
    servoring.write(0);
    Serial.println("Ring Down");


  } else {
    servoring.write(180);
    Serial.println("Ring Up");

  }

  if (valsRec[0] == 5) {
    servopinky.write(0);
    Serial.println("Pinky Down");

  } else {
    servopinky.write(180);
    Serial.println("Pinky Up");

  }
  delay(1000);
}
