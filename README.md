# Moda Ürünleri Görsel Arama ve Kombin Öneri Sistemi

## Proje Özeti

Bu proje, moda ürünleri için geliştirilmiş görsel tabanlı arama ve kombin öneri sistemidir. Kullanıcılar bir ürün görseli yükleyerek görsel olarak benzer ürünleri bulabilir ve aynı zamanda uyumlu kombin önerileri alabilir.

Sistem, ResNet50 derin öğrenme modeli ile görsel özellik çıkarımı yapmakta, benzerlik hesaplamasında Cosine Similarity yöntemini kullanmaktadır. Kombin önerileri ise görsel benzerliğin yanı sıra kategori uyumu, kullanım amacı (Usage), cinsiyet ve renk uyumu gibi kriterleri içeren sezgisel (Heuristic) bir puanlama sistemi ile oluşturulmaktadır.

---

# Kullanılan Teknolojiler

## Backend

- Python 3.12
- FastAPI
- Uvicorn

## Derin Öğrenme

- PyTorch
- Torchvision
- ResNet50
- NumPy
- Scikit-Learn

## Veri İşleme

- Pandas
- Pillow (PIL)

## Frontend

- HTML
- CSS
- JavaScript

---

# Veri Seti

Projede Kaggle platformunda bulunan **Fashion Product Images Dataset** kullanılmıştır.

Link: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

Veri seti;

- Ürün görselleri
- Cinsiyet bilgisi
- Ana kategori
- Alt kategori
- Ürün tipi
- Renk bilgisi
- Kullanım amacı (Casual, Sports vb.)

özelliklerini içermektedir.

---

# Proje Klasör Yapısı

```text
fashion/

│
├── backend/
│     ├── main.py
│     └── uploads/
│
├── dataset/
│     ├── images/
│     ├── processed_products.csv
│     ├── product_embeddings.npy
│     └── product_ids.npy
│
├── frontend/
│     ├── index.html
│     ├── style.css
│     └── script.js
│
├── ml/
│     ├── prepare_dataset.py
│     ├── extract_embeddings.py
│     └── evaluate_topk.py
│
└── README.md
```

---

# Gerekli Kütüphaneler

Aşağıdaki kütüphanelerin kurulu olması gerekmektedir:

- fastapi
- uvicorn
- torch
- torchvision
- pandas
- numpy
- pillow
- scikit-learn
- python-multipart

Kurulum:

```bash
pip install fastapi
pip install uvicorn
pip install torch torchvision
pip install pandas
pip install numpy
pip install pillow
pip install scikit-learn
pip install python-multipart
```

---

# Projenin Çalıştırılması

## Adım 1: Proje klasörüne geçiniz

```bash
cd fashion
```

---

## Adım 2: Backend'i başlatınız

```bash
python -m uvicorn backend.main:app --reload
```

Sunucu aşağıdaki adreste çalışacaktır:

```text
http://127.0.0.1:8000
```

Swagger arayüzü:

```text
http://127.0.0.1:8000/docs
```

---

## Adım 3: Frontend'i çalıştırınız

Visual Studio Code içerisinde Live Server eklentisi kullanılarak:

```text
frontend/index.html
```

dosyası açılır.

---

# Benzer Ürün Arama

Endpoint:

```http
POST /search
```

Çıktı olarak:

- Benzer ürünler
- Ürün kategorisi
- Renk bilgisi
- Benzerlik oranı

döndürülmektedir.

---

# Kombin Öneri Sistemi

Endpoint:

```http
POST /recommend-outfit
```

Sistem;

- Cinsiyet uyumu
- Kategori uyumu
- Kullanım amacı uyumu
- Renk uyumu

kriterlerini kullanarak kombin önerileri oluşturmaktadır.

---

# Kullanılan Değerlendirme Metrikleri

## Top-5 Hit Rate

Benzer ürün aramasında ilk 5 sonuç içerisinde doğru kategoriye ait en az bir ürünün bulunma oranını göstermektedir.

Sonuç:

````


## Sistem Tepki Süresi (Latency)

Bir sorgunun ortalama cevaplanma süresini ifade etmektedir.

Sonuç:

```text
0.175 saniye
````

---

## Heuristic Outfit Score

Kombin önerileri;

- Görsel benzerlik
- Kullanım amacı uyumu
- Renk uyumu
- Kategori uyumu

kriterleri dikkate alınarak oluşturulmaktadır.

---

# Geliştirici

**Talha Demirel**

İstanbul Topkapı Üniversitesi

Yazılım Mühendisliği Bölümü

Bitirme Projesi

2026

---

# Not

Bu proje akademik amaçlı olarak geliştirilmiş olup herhangi bir ticari amaç taşımamaktadır.
