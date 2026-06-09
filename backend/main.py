import os
import shutil
import numpy as np
import pandas as pd
import torch
import time

from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

PRODUCTS_CSV = "dataset/processed_products.csv"
EMBEDDINGS_FILE = "dataset/product_embeddings.npy"
IDS_FILE = "dataset/product_ids.npy"
UPLOAD_DIR = "backend/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# 1. Verileri Yükle
products_df = pd.read_csv(PRODUCTS_CSV)
embeddings = np.load(EMBEDDINGS_FILE)
product_ids = np.load(IDS_FILE).astype(int)

# 2. DataFrame'i hızlı ve güvenli arama için ID'ye göre indeksle ve Sözlüğe çevir
products_df["id"] = products_df["id"].astype(int)
# Bu işlem veri sırasının bozulmasını engeller ve eşleşmeyi garantiye alır
products_dict = products_df.set_index("id").to_dict(orient="index")

# Model Yükleme
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity() 
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.get("/")
def home():
    return {
        "message": "Fashion Visual Search API çalışıyor",
        "total_products": len(products_dict),
        "embedding_count": len(product_ids)
    }

def extract_embedding(image_path):
    # NOT: Eğer embedding'leri üretirken cv2 kullandıysan burayı da cv2 yapmalısın!
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image_tensor)

    return embedding.squeeze().numpy()

@app.post("/search")
async def search_similar_products(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    start_time = time.time()

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    uploaded_id = os.path.splitext(file.filename)[0]

    selected_product_category = None
    selected_product_gender = None
    selected_product_color = None
    reference_product_id = None

    query_embedding = extract_embedding(file_path).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_k_indices = similarities.argsort()[::-1]

    if uploaded_id.isdigit():
        uploaded_id_int = int(uploaded_id)

        if uploaded_id_int in products_dict:
            reference_product_id = uploaded_id_int
    else:
        for idx in top_k_indices:
            candidate_id = int(product_ids[idx])

            if candidate_id in products_dict:
                reference_product_id = candidate_id
                break

    if reference_product_id is not None:
        selected_product_category = products_dict[reference_product_id]["sub_category"]
        selected_product_gender = products_dict[reference_product_id]["gender"]
        selected_product_color = products_dict[reference_product_id]["color"]

    similar_color_groups = {
        "Black": ["Black", "Grey", "Navy Blue", "White"],
        "White": ["White", "Grey", "Black", "Blue", "Navy Blue"],
        "Navy Blue": ["Navy Blue", "Blue", "Black", "Grey"],
        "Blue": ["Blue", "Navy Blue", "Grey", "White"],
        "Grey": ["Grey", "Black", "White", "Navy Blue", "Blue"],
        "Red": ["Red", "Maroon", "Black", "White"],
        "Maroon": ["Maroon", "Red", "Black", "Grey"],
        "Green": ["Green", "Olive", "Black", "Beige"],
        "Olive": ["Olive", "Green", "Black", "Beige"],
        "Pink": ["Pink", "Purple", "White", "Grey"],
        "Purple": ["Purple", "Pink", "Black", "White"],
        "Yellow": ["Yellow", "Orange", "White", "Black"],
        "Orange": ["Orange", "Yellow", "Red", "Black"],
        "Brown": ["Brown", "Beige", "Black", "Navy Blue"],
        "Beige": ["Beige", "Brown", "White", "Navy Blue"],
        "Silver": ["Silver", "Grey", "White", "Black"],
        "Gold": ["Gold", "Yellow", "Brown", "Black"]
    }

    results = []

    for idx in top_k_indices:
        product_id = int(product_ids[idx])

        if reference_product_id is not None and product_id == reference_product_id:
            continue

        if product_id not in products_dict:
            continue

        product_info = products_dict[product_id]

        if selected_product_gender is not None:
            if product_info["gender"] not in [selected_product_gender, "Unisex"]:
                continue

        if selected_product_category is not None:
            if product_info["sub_category"] != selected_product_category:
                continue

        if selected_product_color in similar_color_groups:
            if product_info["color"] not in similar_color_groups[selected_product_color]:
                continue

        results.append({
            "id": product_id,
            "name": product_info["name"],
            "gender": product_info["gender"],
            "main_category": product_info["main_category"],
            "sub_category": product_info["sub_category"],
            "article_type": product_info["article_type"],
            "color": product_info["color"],
            "usage": product_info["usage"],
            "image_path": f"dataset/images/{product_id}.jpg",
            "similarity_score": float(similarities[idx])
        })

        if len(results) == 4:
            break

    end_time = time.time()
    latency = end_time - start_time

    return {
        "uploaded_file": file.filename,
        "reference_product_id": reference_product_id,
        "selected_category": selected_product_category,
        "selected_gender": selected_product_gender,
        "selected_color": selected_product_color,
        "latency_seconds": round(latency, 4),
        "results": results
    }
    
@app.post("/recommend-outfit")
async def recommend_outfit(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    start_time = time.time()
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    uploaded_id = os.path.splitext(file.filename)[0]

    query_embedding = extract_embedding(file_path).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    similar_indices = similarities.argsort()[::-1]

    selected_product_category = None
    selected_product_gender = None
    selected_product_usage = None
    selected_product_color = None
    reference_product_id = None

    if uploaded_id.isdigit():
        uploaded_id_int = int(uploaded_id)

        if uploaded_id_int in products_dict:
            reference_product_id = uploaded_id_int
    else:
        for idx in similar_indices:
            candidate_id = int(product_ids[idx])

            if candidate_id in products_dict:
                reference_product_id = candidate_id
                break

    if reference_product_id is not None:
        selected_product_category = products_dict[reference_product_id]["sub_category"]
        selected_product_gender = products_dict[reference_product_id]["gender"]
        selected_product_usage = products_dict[reference_product_id]["usage"]
        selected_product_color = products_dict[reference_product_id]["color"]

    outfit_rules = {
        "Topwear": ["Bottomwear", "Shoes", "Eyewear", "Bags"],
        "Bottomwear": ["Topwear", "Shoes", "Eyewear", "Bags"],
        "Shoes": ["Topwear", "Bottomwear", "Eyewear", "Bags"],
        "Bags": ["Topwear", "Bottomwear", "Shoes", "Eyewear"],
        "Eyewear": ["Topwear", "Bottomwear", "Shoes", "Bags"]
    }

    color_rules = {
        "Black": ["Black", "White", "Grey", "Red", "Blue", "Navy Blue", "Beige", "Brown"],
        "White": ["White", "Black", "Blue", "Navy Blue", "Grey", "Red", "Beige"],
        "Navy Blue": ["Navy Blue", "White", "Grey", "Blue", "Beige", "Black", "Brown"],
        "Blue": ["Blue", "White", "Grey", "Navy Blue", "Beige", "Black"],
        "Grey": ["Grey", "Black", "White", "Navy Blue", "Blue", "Red"],
        "Red": ["Red", "Black", "White", "Grey", "Navy Blue", "Beige"],
        "Green": ["Green", "Black", "White", "Beige", "Grey", "Brown"],
        "Beige": ["Beige", "White", "Navy Blue", "Brown", "Black", "Blue", "Green"],
        "Brown": ["Brown", "Beige", "White", "Black", "Navy Blue", "Green"],
        "Pink": ["Pink", "White", "Grey", "Black", "Blue", "Purple"],
        "Purple": ["Purple", "White", "Black", "Grey", "Pink"],
        "Yellow": ["Yellow", "Blue", "Navy Blue", "White", "Black", "Grey"],
        "Orange": ["Orange", "Blue", "White", "Black", "Grey", "Brown"],
        "Maroon": ["Maroon", "Black", "White", "Grey", "Beige", "Navy Blue"],
        "Silver": ["Silver", "Grey", "White", "Black", "Blue"],
        "Gold": ["Gold", "Black", "White", "Brown", "Beige"]
    }

    target_categories = outfit_rules.get(selected_product_category, [])

    candidate_results = []

    for idx in similar_indices:
        product_id = int(product_ids[idx])

        if reference_product_id is not None and product_id == reference_product_id:
            continue

        if product_id not in products_dict:
            continue

        product_info = products_dict[product_id]
        product_category = product_info["sub_category"]

        # 1. Cinsiyet filtresi
        if selected_product_gender is not None:
            if product_info["gender"] not in [selected_product_gender, "Unisex"]:
                continue

        # 2. Tamamlayıcı kategori filtresi
        if product_category not in target_categories:
            continue

        # 3. Geniş renk filtresi
        if selected_product_color in color_rules:
            if product_info["color"] not in color_rules[selected_product_color]:
                continue

        score = 0

        # Görsel benzerlik
        score += similarities[idx] * 0.50

        # Usage uyumu bonusu
        if selected_product_usage is not None and product_info["usage"] == selected_product_usage:
            score += 0.20

        # Renk uyumu bonusu
        if selected_product_color in color_rules:
            if product_info["color"] in color_rules[selected_product_color]:
                score += 0.20

        # Kategori uyumu bonusu
        score += 0.10

        candidate_results.append({
            "id": product_id,
            "name": product_info["name"],
            "gender": product_info["gender"],
            "main_category": product_info["main_category"],
            "sub_category": product_info["sub_category"],
            "article_type": product_info["article_type"],
            "color": product_info["color"],
            "usage": product_info["usage"],
            "image_path": f"dataset/images/{product_id}.jpg",
            "final_score": float(score)
        })

    candidate_results = sorted(
        candidate_results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    results = []
    selected_categories = set()

    for item in candidate_results:
        if item["sub_category"] in selected_categories:
            continue

        results.append(item)
        selected_categories.add(item["sub_category"])

        if len(results) == 4:
            break
    
    end_time = time.time()
    latency = end_time - start_time
    
    average_outfit_score = 0

    if len(results) > 0:
        average_outfit_score = sum(item["final_score"] for item in results) / len(results)
    
    return {
        "uploaded_file": file.filename,
        "reference_product_id": reference_product_id,
        "selected_category": selected_product_category,
        "selected_gender": selected_product_gender,
        "selected_usage": selected_product_usage,
        "selected_color": selected_product_color,
        "target_categories": target_categories,
        "average_outfit_score": round(average_outfit_score, 4),
        "latency_seconds": round(latency, 4),
        "recommendations": results
    }