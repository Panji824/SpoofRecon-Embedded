import os
import json
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = MTCNN(keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

data_json = "database.json"
penghuni_dir = "wajah_penghuni"
saved_picture_dir = "save_wajah_mtcnn"

import matplotlib as plt

def get_embedding(image_path):
    """Extract face embedding from an image file."""
    image = Image.open(image_path).convert("RGB")
    face = detector(image)  # Assuming you have a face detector function
    if face is not None and face.shape[0] > 0:
        return face[0]['embedding'].tolist()  # Convert NumPy array to list for JSON compatibility
    return None


def process_images(folder_path, output_json="face_embeddings.json"):
    embeddings = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.png', '.jpeg')):  # Filter image files
            embedding = get_embedding(file_path)
            if embedding:
                embeddings[filename] = embedding

    with open(output_json, "w") as f:
        json.dump(embeddings, f, indent=4)

    print(f"Embeddings saved to {output_json}")


process_images(penghuni_dir)