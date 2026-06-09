import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PRODUCTS_CSV = "../dataset/processed_products.csv"
EMBEDDINGS_FILE = "../dataset/product_embeddings.npy"
IDS_FILE = "../dataset/product_ids.npy"

products_df = pd.read_csv(PRODUCTS_CSV)

embeddings = np.load(EMBEDDINGS_FILE)
product_ids = np.load(IDS_FILE)

print("Embedding shape:", embeddings.shape)
print("Product IDs shape:", product_ids.shape)

query_product_id = product_ids[0]

print("\nAranan ürün ID:")
print(query_product_id)

query_index = np.where(product_ids == query_product_id)[0][0]

query_embedding = embeddings[query_index].reshape(1, -1)

similarities = cosine_similarity(query_embedding, embeddings)[0]

similar_indices = similarities.argsort()[::-1]

print("\nEn benzer 5 ürün:")

count = 0

for idx in similar_indices:
    product_id = product_ids[idx]

    if product_id == query_product_id:
        continue

    product_info = products_df[products_df["id"] == product_id].iloc[0]

    print("--------------------------------")
    print("ID:", product_id)
    print("Ürün Adı:", product_info["name"])
    print("Kategori:", product_info["sub_category"])
    print("Ürün Tipi:", product_info["article_type"])
    print("Renk:", product_info["color"])
    print("Benzerlik Skoru:", similarities[idx])

    count += 1

    if count == 5:
        break