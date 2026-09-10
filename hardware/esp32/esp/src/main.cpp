#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

const char* ssid = "tanshuhotspot";
const char* password = "abhiamri";

WebServer server(80);

int mode = 0;
float torque = 0.0;

void receiveData() {

    StaticJsonDocument<256> doc;

    DeserializationError error =
        deserializeJson(doc, server.arg("plain"));

    if (!error) {

        mode = doc["mode_id"];
        torque = doc["target_torque"];

        Serial.print("Mode: ");
        Serial.println(mode);

        Serial.print("Torque: ");
        Serial.println(torque);

        // TODO: Drive BTS7960 here

        server.send(
            200,
            "application/json",
            "{\"status\":\"received\"}"
        );
    }
    else {

        Serial.println("Invalid JSON");

        server.send(
            400,
            "application/json",
            "{\"status\":\"invalid json\"}"
        );
    }
}


void setup() {
    Serial.begin(115200);

    Serial.println();
    Serial.println("Connecting to WiFi...");

    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
      Serial.print(".");
      Serial.print(" Status: ");
      Serial.println(WiFi.status());
      delay(1000);
    }

    Serial.println();
    Serial.println("Connected!");

    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    
    server.on("/exercise",HTTP_POST,receiveData);

    server.begin();
}

void loop() {
      server.handleClient();

}