import pandas as pd
import os

DATASET_CSV = "../dataset/styles.csv"
IMAGES_DIR = "../dataset/images"
OUTPUT_CSV = "../dataset/processed_products.csv"

df = pd.read_csv(DATASET_CSV, on_bad_lines="skip")

print("Başlangıç ürün sayısı:", len(df))

required_columns = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "usage",
    "productDisplayName"
]

df = df[required_columns]

df = df.dropna()

df["image_path"] = df["id"].apply(
    lambda product_id: f"{IMAGES_DIR}/{product_id}.jpg"
)

df = df[df["image_path"].apply(os.path.exists)]

df = df.rename(columns={
    "productDisplayName": "name",
    "masterCategory": "main_category",
    "subCategory": "sub_category",
    "articleType": "article_type",
    "baseColour": "color"
})

df = df.reset_index(drop=True)

df.to_csv(OUTPUT_CSV, index=False)

print("Temizlenmiş ürün sayısı:", len(df))
print("Dosya oluşturuldu:", OUTPUT_CSV)

print("\nİlk 5 temiz ürün:")
print(df.head())