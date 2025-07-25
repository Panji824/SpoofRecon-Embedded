import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# Set device: CUDA jika tersedia, jika tidak gunakan CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definisikan transformasi yang sama seperti saat training
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path, num_classes=2):
    """
    Muat model deteksi masker dari path yang diberikan.
    """
    # Inisialisasi model ResNet18 tanpa pretrained weights
    model = models.resnet18(weights=None)
    # Ubah layer fully-connected agar sesuai dengan jumlah kelas
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Muat state dict. Perhatikan, jika model disimpan dengan key tertentu atau parameter tambahan,
    # pastikan penyesuaian sesuai kebutuhan.
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model ke mode evaluasi
    return model


# Load model secara global agar tidak perlu load berulang kali tiap prediksi.
model_path = 'mask_detection_model.pth'  # Ganti sesuai path model Anda
model = load_model(model_path).to(device)


def predict_mask(image_path):
    """
    Fungsi predict_mask menerima path ke gambar wajah, lalu:
      - Membuka gambar menggunakan PIL.
      - Menerapkan transformasi.
      - Melakukan prediksi menggunakan model deteksi masker.
      - Mengembalikan string 'mask' atau 'unmask' berdasarkan output prediksi.
    """
    # Buka gambar dan pastikan format RGB
    image = Image.open(image_path).convert('RGB')
    # Terapkan transformasi dan tambahkan batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    # Mapping indeks ke label, sesuaikan dengan urutan kelas pada saat training
    class_names = ['mask', 'unmask']
    return class_names[predicted.item()]


# Contoh penggunaan secara mandiri untuk testing
if __name__ == '__main__':
    # Misalnya, kita ingin menguji prediksi dengan gambar lokal
    test_image_path = 'test_langsung/IMG-20250314-WA0007.jpg'  # Ganti dengan path gambar uji
    prediction = predict_mask(test_image_path)
    print(f'Prediksi: {prediction}')
