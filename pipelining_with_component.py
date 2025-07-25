# Kode Python: Integrasi Arduino dengan Pipeline ML

import serial
import time
from face_detection import detect_and_crop_face
from detect_spoof import predict_spoof
from cosine_face_recognition import recognize_face
import requests
import os
from dotenv import load_dotenv

# 🔹 Load file .env
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Inisialisasi Serial
ser = serial.Serial('COM4', 9600)  # Ganti COM3 sesuai port Arduino Anda

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=data)

while True:
    if ser.in_waiting > 0:
        data = ser.readline().decode().strip()
        print(f"Data dari Arduino: {data}")

        if data == "PIR_DETECTED":
            print("🔎 Mendeteksi wajah...")
            face_path = detect_and_crop_face()

            if not face_path:
                print("❌ Tidak ada wajah terdeteksi.")
                continue

            print("🛡️ Mengecek wajah Real atau Spoof...")
            spoof_label, prob = predict_spoof(face_path)
            if spoof_label == "Spoof":
                print(f"🔴 Wajah Palsu (Spoof) dengan probabilitas {prob:.4f}. Akses ditolak.")
                send_telegram_message(f"🔴 Wajah Palsu! Probabilitas {prob:.4f}. Akses ditolak.")
                continue

            print("✅ Wajah Asli (Real). Melakukan pengenalan wajah...")
            identity = recognize_face(face_path)
            if identity:
                print(f"✅ Wajah dikenali: {identity}. Membuka pintu.")
                ser.write(b'OPEN_SOLENOID\n')
                send_telegram_message(f"✅ Wajah dikenali: {identity}")
            else:
                print("⚠️ Wajah tidak dikenali! Akses ditolak.")
                send_telegram_message("⚠️ Wajah tidak dikenali!")

        elif data == "LIMIT_SWITCH_TRIGGERED":
            print("⚠️ Pintu dibuka paksa! Akses ditolak.")
            send_telegram_message("⚠️ Pintu dibuka paksa! Akses ditolak.")
