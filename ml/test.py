import pandas as pd
import os

df = pd.read_csv("../dataset/styles.csv", on_bad_lines="skip")

print("Toplam ürün sayısı:", len(df))

print("\nKolonlar:")
print(df.columns.tolist())

print("\nİlk 5 ürün:")
print(df[[
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "usage",
    "productDisplayName"
]].head())

first_id = df.iloc[0]["id"]

image_path = f"../dataset/images/{first_id}.jpg"

print("\nİlk ürün görsel yolu:")
print(image_path)

if os.path.exists(image_path):
    print("Görsel bulundu.")
else:
    print("Görsel bulunamadı.")