import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PRODUCTS_CSV = "../dataset/processed_products.csv"
EMBEDDINGS_FILE = "../dataset/product_embeddings.npy"
IDS_FILE = "../dataset/product_ids.npy"

TOP_K = 5

products_df = pd.read_csv(PRODUCTS_CSV)
products_df["id"] = products_df["id"].astype(int)

products_dict = products_df.set_index("id").to_dict(orient="index")

embeddings = np.load(EMBEDDINGS_FILE)
product_ids = np.load(IDS_FILE).astype(int)

total_queries = 0
successful_queries = 0
precision_scores = []

for query_index, query_id in enumerate(product_ids):
    if query_id not in products_dict:
        continue

    query_category = products_dict[query_id]["sub_category"]

    query_embedding = embeddings[query_index].reshape(1, -1)

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    similar_indices = similarities.argsort()[::-1]

    top_results = []

    for idx in similar_indices:
        candidate_id = int(product_ids[idx])

        if candidate_id == query_id:
            continue

        if candidate_id not in products_dict:
            continue

        top_results.append(candidate_id)

        if len(top_results) == TOP_K:
            break

    if len(top_results) == 0:
        continue

    correct_count = 0

    for candidate_id in top_results:
        candidate_category = products_dict[candidate_id]["sub_category"]

        if candidate_category == query_category:
            correct_count += 1

    precision_at_k = correct_count / len(top_results)

    precision_scores.append(precision_at_k)

    if correct_count > 0:
        successful_queries += 1

    total_queries += 1

average_precision_at_k = sum(precision_scores) / len(precision_scores)
top_k_hit_rate = successful_queries / total_queries

print("Top-K Evaluation Results")
print("------------------------")
print("Top K:", TOP_K)
print("Toplam sorgu sayısı:", total_queries)
print("Başarılı sorgu sayısı:", successful_queries)
print("Top-K Hit Rate:", round(top_k_hit_rate, 4))
print("Average Precision@K:", round(average_precision_at_k, 4))