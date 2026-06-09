import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights  # <-- Güncel import yapısı
from PIL import Image

PRODUCTS_CSV = "../dataset/processed_products.csv"
OUTPUT_EMBEDDINGS = "../dataset/product_embeddings.npy"
OUTPUT_IDS = "../dataset/product_ids.npy"
LIMIT = 500

df = pd.read_csv(PRODUCTS_CSV)
df = df.head(LIMIT)

# backend/main.py ile birebir aynı güncel model yükleme mimarisi
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity() 
model.eval()

# backend/main.py ile birebir aynı transform adımları
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

embeddings = []
product_ids = []

print("Embedding çıkarma işlemi başlıyor...\n")

for index, row in df.iterrows():
    image_path = row["image_path"]

    if not os.path.exists(image_path):
        print(f"Görsel bulunamadı: {image_path}")
        continue

    try:
        # Görseli PIL ile RGB okuyoruz (Backend ile tam uyum)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = model(image_tensor)

        embedding = embedding.squeeze().numpy()

        embeddings.append(embedding)
        # ID'leri garanti olsun diye int tipine çevirip ekliyoruz
        product_ids.append(int(row["id"]))

        print(f"{index + 1}/{LIMIT} - Ürün işlendi: {row['id']}")

    except Exception as e:
        print(f"Hata oluştu: {row['id']} - {e}")

embeddings = np.array(embeddings)
product_ids = np.array(product_ids, dtype=int)  # Diziyi int array olarak zorunlu kılıyoruz

np.save(OUTPUT_EMBEDDINGS, embeddings)
np.save(OUTPUT_IDS, product_ids)

print("\nİşlem başarıyla tamamlandı.")
print("Embedding matrisi boyutu (Shape):", embeddings.shape)
print("ID listesi boyutu (Shape):", product_ids.shape)