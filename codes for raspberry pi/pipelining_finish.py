import time
import threading
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import os
from dotenv import load_dotenv
import pygame
import paho.mqtt.client as mqtt

from modules.face_detection_webcam import detect_and_crop_face
from modules.cosine_face_recognition import recognize_face, compare_embeddings

# ===== Konfigurasi =====
MODEL_PATH = "modules/mobilenetv2-scripted.pt"
MQTT_BROKER = "192.168.67.57"
TOPIC_PIR = "sensor/pir"
TOPIC_LIMIT = "sensor/limit_switch"
TOPIC_RELAY = "relay/control"

AUDIO = {
    "alarm": "audio/audio_v1.mp3",
    "spoof": "audio/audio_v2.mp3",
    "unknown": "audio/audio_v3.mp3",
    "granted": "audio/audio_v4.mp3",
    "prompt_smile": "audio/audio_v5.mp3"
}

# Load .env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
    raise ValueError("⚠️ Token Telegram atau Chat ID tidak ditemukan dalam file .env")

# Load model spoof detection
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("⚠️ Model TorchScript tidak ditemukan!")
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# Transform untuk model spoof
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Init pygame
pygame.mixer.init()

# Telegram & audio
def play_audio(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def send_telegram_message(message, photo_path=None):
    try:
        data = {"chat_id": CHAT_ID, "text": message}
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data=data, timeout=5)

        if photo_path and os.path.exists(photo_path):
            with open(photo_path, 'rb') as photo:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                    data={"chat_id": CHAT_ID},
                    files={"photo": photo},
                    timeout=5
                )
    except Exception as e:
        print(f"⚠️ Gagal kirim ke Telegram: {e}")

# Flags
limit_triggered = threading.Event()
motion_detected = threading.Event()

# MQTT Callback
def on_connect(client, userdata, flags, rc):
    print("✅ MQTT terhubung.")
    client.subscribe(TOPIC_PIR)
    client.subscribe(TOPIC_LIMIT)

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode()

    if topic == TOPIC_LIMIT and payload == "1":
        if not limit_triggered.is_set():
            print("🚨 Limit switch terpicu - pembobolan!")
            play_audio(AUDIO["alarm"])
            send_telegram_message("🚨 Peringatan! Pintu dibuka paksa sebelum proses selesai.")
            limit_triggered.set()

    elif topic == TOPIC_PIR and payload == "1":
        if not limit_triggered.is_set():
            motion_detected.set()

# MQTT Init
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

# Main Loop
try:
    while True:
        if limit_triggered.is_set():
            continue

        if motion_detected.is_set():
            print("👁️ Gerakan terdeteksi. Menunggu 3 detik...")
            motion_detected.clear()
            time.sleep(4)

            print("📸 Ambil gambar pertama...")
            face1_path = detect_and_crop_face("gambar_ke1.jpg")
            if not face1_path or limit_triggered.is_set():
                continue

            print("🛡️ Deteksi spoofing dengan model...")
            image = Image.open(face1_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image_tensor)
                prob = torch.sigmoid(output).item()

            if prob < 0.5 or limit_triggered.is_set():
                play_audio(AUDIO["spoof"])
                send_telegram_message(f"⚠️ Deteksi wajah palsu (spoof)! Akses ditolak.\nConfidence: {prob:.4f}",face1_path)
                time.sleep(2)
                continue

            print("🙂 Wajah valid. Siapkan ekspresi baru (senyum)...")
            play_audio(AUDIO["prompt_smile"])
            time.sleep(4)

            print("📸 Mengambil wajah kedua...")
            face2_path = detect_and_crop_face("gambar_ke2.jpg")
            if not face2_path or limit_triggered.is_set():
                continue

            dist = compare_embeddings(face1_path, face2_path)
            print(f"📏 Cosine distance liveness: {dist:.4f}")
            if dist is None or dist < 0.07:
                play_audio(AUDIO["spoof"])
                send_telegram_message(f"⚠️ Liveness gagal! Wajah terlalu mirip.\nConfidence spoof model: {prob:.4f}\nCosine distance: {dist:.4f}",face1_path)
                continue
            elif dist > 0.4:
                play_audio(AUDIO["unknown"])
                send_telegram_message(f"⚠️ Liveness gagal! Wajah tidak konsisten.\nConfidence spoof model: {prob:.4f}\nCosine distance: {dist:.4f}",face1_path)
                continue

            print("✅ Liveness OK. Lanjut ke pengenalan wajah...")
            identity, max_score = recognize_face(face1_path)
            if identity and not limit_triggered.is_set():
                client.publish(TOPIC_RELAY, "ON")
                play_audio(AUDIO["granted"])
                send_telegram_message(
                    f"✅ Akses diberikan kepada {identity}.\nConfidence spoof model: {prob:.4f}\nCosine similarity: {max_score:.4f}",
                    face1_path
                )
                time.sleep(10)
                client.publish(TOPIC_RELAY, "ON")
            else:
                max_score = max_score if max_score else 0.0  # fallback jika None
                play_audio(AUDIO["unknown"])
                send_telegram_message(
                    f"⚠️ Wajah tidak dikenali! Akses ditolak.\nConfidence spoof model: {prob:.4f}\nCosine similarity: {max_score:.4f}",
                    face1_path
                )
        time.sleep(0.2)

except KeyboardInterrupt:
    print("❌ Program dihentikan oleh user.")
finally:
    pygame.mixer.quit()
    client.loop_stop()
    client.disconnect()

                                                                                                
