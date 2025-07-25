import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# 🔹 Load model TorchScript
model_path = "mobilenet/mobilenetv2-scripted.pt"

try:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    print("✅ Model berhasil diload.")
except Exception as e:
    print(f"❌ Gagal meload model: {e}")
    exit()

# 🔹 Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_spoof(image_path):
    """Memprediksi apakah wajah Real atau Spoof"""
    if not os.path.exists(image_path):
        print(f"❌ Gambar {image_path} tidak ditemukan!")
        return "Spoof", 0.0

    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Tambahkan batch dimension
    except Exception as e:
        print(f"❌ Gagal membuka gambar: {e}")
        return "Spoof", 0.0

    # 🔹 Prediksi
    with torch.no_grad():
        output = model(image)
        prob = output.item()

    # 🔹 Tentukan hasil
    if prob > 0.5:
        return "Real", prob
    else:
        return "Spoof", prob
