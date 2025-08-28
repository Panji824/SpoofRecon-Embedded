import os
import torch
import cv2
import numpy as np
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def get_embedding(face_path):
    img = cv2.imread(face_path)
    if img is None:
        print(f"❌ Gambar tidak ditemukan: {face_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = facenet(img_tensor).cpu().numpy()[0]
    return embedding

def recognize_face(face_path, db_path="database/face_embeddings.json"):
    target_embedding = get_embedding(face_path)
    if target_embedding is None:
        return None

    if not os.path.exists(db_path):
        print("⚠️ Database embedding tidak ditemukan.")
        return None

    with open(db_path, 'r') as f:
        db = json.load(f)

    best_match = None
    best_score = -1
    for name, emb in db.items():
        score = cosine_similarity([target_embedding], [np.array(emb)])[0][0]
        if score > best_score:
            best_score = score
            best_match = name

    if best_score > 0.6:
        return best_match
    return None

def compare_embeddings(path1, path2):
    emb1 = get_embedding(path1)
    emb2 = get_embedding(path2)
    if emb1 is None or emb2 is None:
        return None
    sim = cosine_similarity([emb1], [emb2])[0][0]
    dist = 1 - sim
    return dist
