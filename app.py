import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os

# -----------------------------
# 1️⃣ Modeli ve etiketleri yükle
# -----------------------------
SAVE_DIR = "saved_model_v1"

# Model dosyasını otomatik bul
model_path = None
for file in ["model.keras", "model.h5"]:
    full_path = os.path.join(SAVE_DIR, file)
    if os.path.exists(full_path):
        model_path = full_path
        break

if model_path is None:
    st.error("❌ Model dosyası bulunamadı! Lütfen 'saved_model_v1' klasörüne kaydettiğinden emin ol.")
    st.stop()

# Modeli yükle
model = keras.models.load_model(model_path)

# Etiket dosyasını oku
labels_path = os.path.join(SAVE_DIR, "class_names.json")
if not os.path.exists(labels_path):
    st.error("❌ class_names.json bulunamadı!")
    st.stop()

with open(labels_path, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# -----------------------------
# 2️⃣ Streamlit arayüzü
# -----------------------------
st.title("🐶🐱 Kedi–Köpek Görüntü Sınıflandırıcı")
st.write("Yapay zeka modeli ile bir görüntünün **kedi mi köpek mi** olduğunu tahmin eder.")

uploaded_file = st.file_uploader("Bir resim yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Görseli göster
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # -----------------------------
    # 3️⃣ Görseli modele uygun hale getir
    # -----------------------------
    IMAGE_SIZE = (224, 224)   # eğitim sırasında kullandığın boyut (örneğin 128x128 olabilir)
    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, height, width, 3)

    # -----------------------------
    # 4️⃣ Tahmin yap
    # -----------------------------
    predictions = model.predict(img_array)
    
    # Çıktı tipi: Binary veya Softmax olabilir
    if predictions.shape[1] == 1:
        prob = predictions[0][0]
        label = class_names[1] if prob > 0.5 else class_names[0]
        confidence = prob if prob > 0.5 else 1 - prob
    else:
        confidence = np.max(predictions)
        label = class_names[np.argmax(predictions)]

    # -----------------------------
    # 5️⃣ Sonucu göster
    # -----------------------------
    st.subheader("🎯 Tahmin Sonucu")
    st.write(f"**Sınıf:** {label}")
    st.write(f"**Olasılık:** {confidence*100:.2f}%")

    if confidence < 0.6:
        st.info("⚠️ Düşük güven düzeyi: Görüntü net değil veya model emin değil.")
