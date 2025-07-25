import cv2
import os

# URL kamera eksternal (IP Webcam)
URL = "http://192.168.105.178:8080/video"  # Ganti dengan URL kamera HP

def detect_and_crop_face():
    # Buat direktori 'save_picture' jika belum ada
    save_dir = 'save_picture'
    os.makedirs(save_dir, exist_ok=True)

    # Muat classifier Haar Cascade untuk deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Buka kamera eksternal
    cap = cv2.VideoCapture(URL)
    if not cap.isOpened():
        print("❌ Tidak dapat mengakses kamera eksternal.")
        return

    face_path = None  # Untuk menyimpan lokasi gambar wajah

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame dari kamera eksternal.")
            break

        # Konversi frame ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Jika ada wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # Gambar kotak di sekitar wajah (opsional)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Lakukan cropping pada bounding box wajah
            cropped_face = frame[y:y + h, x:x + w]

            # Simpan gambar hasil cropping
            face_path = os.path.join(save_dir, 'cropped_face.jpg')
            cv2.imwrite(face_path, cropped_face)
            print(f"✅ Gambar wajah disimpan di {face_path}")

            # Keluar dari loop jika sudah menyimpan satu gambar wajah
            break

        # Tampilkan frame (opsional, bisa di-comment jika tidak diperlukan)
        cv2.imshow('Kamera Eksternal - Deteksi Wajah', frame)

        # Jika wajah sudah dideteksi, keluar dari loop
        if face_path is not None:
            break

        # Tekan 'q' untuk keluar manual jika tidak ingin melanjutkan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan kamera dan tutup semua jendela
    cap.release()
    cv2.destroyAllWindows()

    return face_path

# Contoh penggunaan
if __name__ == "__main__":
    detect_and_crop_face()
