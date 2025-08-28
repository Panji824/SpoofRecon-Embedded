#include <WiFi.h>
#include <PubSubClient.h>

// ======= Konfigurasi WiFi dan MQTT =======
const char* ssid = "Galaxy A52F721";
const char* password = "nonepass";
const char* mqtt_server = "192.168.67.57";

WiFiClient espClient;
PubSubClient client(espClient);

// ======= Definisi Pin =======
#define PIR_PIN     14   // Sensor PIR
#define LIMIT_PIN   25   // Limit Switch (aktif LOW)
#define RELAY_PIN   32   // Relay untuk kunci pintu

// ======= Setup Koneksi WiFi =======
void setup_wifi() {
  delay(10);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
}

// ======= Callback Saat Menerima Pesan MQTT =======
void callback(char* topic, byte* message, unsigned int length) {
  String msg = "";
  for (int i = 0; i < length; i++) {
    msg += (char)message[i];
  }

  // Logika kontrol relay
  if (String(topic) == "relay/control") {
    if (msg == "ON") {
      digitalWrite(RELAY_PIN, HIGH);  // 🔒 Kunci pintu
    } else if (msg == "OFF") {
      digitalWrite(RELAY_PIN, LOW);   // 🔓 Buka pintu
    }
  }
}

// ======= Reconnect MQTT jika terputus =======
void reconnect() {
  while (!client.connected()) {
    if (client.connect("ESP32Client")) {
      client.subscribe("relay/control");  // Subscribe ke topik kontrol
    } else {
      delay(5000);
    }
  }
}

// ======= Setup Awal =======
void setup() {
  pinMode(PIR_PIN, INPUT);
  pinMode(LIMIT_PIN, INPUT_PULLUP);  // Aktif LOW
  pinMode(RELAY_PIN, OUTPUT);

  digitalWrite(RELAY_PIN, HIGH); // 🔒 Kunci pintu saat boot

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

// ======= Loop Utama =======
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // ======= Kirim status PIR =======
  if (digitalRead(PIR_PIN) == HIGH) {
    client.publish("sensor/pir", "1");
  }

  // ======= Kirim status limit switch =======
  if (digitalRead(LIMIT_PIN) == LOW) {
    client.publish("sensor/limit_switch", "1");  // Tertekan (aktif LOW)
  } else {
    client.publish("sensor/limit_switch", "0");  // Tidak ditekan
  }

  delay(300);  // Hindari spam MQTT
}
