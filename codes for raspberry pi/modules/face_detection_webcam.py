import cv2
import os
import time

def detect_and_crop_face(filename, prompt_audio=None):
    save_dir = 'save_picture'
    os.makedirs(save_dir, exist_ok=True)

    if prompt_audio:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(prompt_audio)
        pygame.mixer.music.play()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Tidak dapat mengakses kamera.")
        return None

    face_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cropped_face = frame[y:y + h, x:x + w]
            face_path = os.path.join(save_dir, filename)
            cv2.imwrite(face_path, cropped_face)
            print(f"✅ Wajah disimpan: {face_path}")
            break

        cv2.imshow('Deteksi Wajah', frame)

        if face_path is not None:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return face_path
