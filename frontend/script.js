const API_URL = "http://127.0.0.1:8000";

const imageInput = document.getElementById("imageInput");
const previewArea = document.getElementById("previewArea");
const previewImage = document.getElementById("previewImage");

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    if (file) {
        previewImage.src = URL.createObjectURL(file);
        previewArea.style.display = "block";
    }
});

async function searchProducts() {
    await sendImageToApi("/search", "searchResults", "results");
}

async function recommendOutfit() {
    await sendImageToApi("/recommend-outfit", "outfitResults", "recommendations");
}

async function sendImageToApi(endpoint, containerId, responseKey) {
    const file = imageInput.files[0];

    if (!file) {
        alert("Lütfen önce bir görsel seçin.");
        return;
    }

    const container = document.getElementById(containerId);
    container.innerHTML = "<p>Yükleniyor...</p>";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(`${API_URL}${endpoint}`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        console.log(endpoint, data);

        const products = data[responseKey];

        if (!products || products.length === 0) {
            container.innerHTML = "<p>Sonuç bulunamadı.</p>";
            return;
        }

        displayResults(products, containerId);

    } catch (error) {
        console.error("Hata:", error);
        container.innerHTML = "<p>Bir hata oluştu.</p>";
    }
}

function displayResults(products, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";

    products.forEach(product => {
        const card = document.createElement("div");
        card.className = "product-card";

        const showScore = containerId === "searchResults";

        card.innerHTML = `
            <img src="${API_URL}/${product.image_path}" alt="${product.name}">
            <h3>${product.name}</h3>
            <p><strong>Kategori:</strong> ${product.sub_category}</p>
            <p><strong>Ürün Tipi:</strong> ${product.article_type}</p>
            <p><strong>Renk:</strong> ${product.color}</p>
            ${showScore ? `<p class="score">Benzerlik: ${(product.similarity_score * 100).toFixed(2)}%</p>` : ""}
        `;

        container.appendChild(card);
    });
}