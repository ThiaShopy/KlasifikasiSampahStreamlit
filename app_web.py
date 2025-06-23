import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import io
import os
import datetime
from src.utils import download_file_if_not_exists

# Konfigurasi
MODEL_PATH = "waste_classifier_jepang.h5"
MODEL_FILE_ID = "1SK6bIDezZQvNGRZz1-NTi0Epg5slpG6Z"
IMG_SIZE = (64, 64)
CLASS_NAMES = ['moeru_gomi', 'moenai_gomi', 'shigen_gomi']
CATEGORY_INFO = {
    'moeru_gomi': {'jp': '燃えるゴミ', 'desc': 'Burnable – Sisa makanan, tisu, daun kering'},
    'moenai_gomi': {'jp': '燃えないゴミ', 'desc': 'Non-Burnable – Logam, kaca, baterai kecil'},
    'shigen_gomi': {'jp': '資源ゴミ', 'desc': 'Recyclable – Kardus, plastik, kertas, botol PET'}
}

# Pastikan model tersedia
with st.spinner("📦 Menyiapkan model klasifikasi..."):
    download_file_if_not_exists(MODEL_FILE_ID, MODEL_PATH)
    model = load_model(MODEL_PATH)

# Inisialisasi state
if 'log' not in st.session_state:
    st.session_state.log = []

if 'count' not in st.session_state:
    st.session_state.count = {'moeru_gomi': 0, 'moenai_gomi': 0, 'shigen_gomi': 0}

st.title("🗑️ Klasifikasi Sampah Jepang Otomatis")

# Upload gambar
uploaded_file = st.file_uploader("📷 Seret dan lepas gambar sampah", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar Anda", width=250)

    img_resized = img.resize(IMG_SIZE)
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Klasifikasikan"):
        pred = model.predict(img_array)
        pred_idx = np.argmax(pred)
        class_name = CLASS_NAMES[pred_idx]
        info = CATEGORY_INFO[class_name]

        # Tampilkan hasil
        st.success(f"{info['jp']} — {info['desc']}")

        # Simpan log
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.log.append(f"{now} - {uploaded_file.name} → {info['jp']} ({class_name})")
        st.session_state.count[class_name] += 1

# Statistik klasifikasi
st.header("📊 Statistik Klasifikasi")
st.bar_chart(st.session_state.count)

# Tombol simpan hasil klasifikasi ke .txt
if st.button("💾 Simpan hasil ke .txt"):
    txt_content = "\n".join(st.session_state.log)
    st.download_button("Unduh log klasifikasi", txt_content, file_name="klasifikasi_log.txt")

# Export dataset klasifikasi
if st.button("📦 Export dataset klasifikasi (.csv)"):
    import pandas as pd
    csv_data = pd.DataFrame([
        line.split(" - ") + [line.split("→")[-1].strip()]
        for line in st.session_state.log
    ], columns=["Waktu", "File", "Klasifikasi"])

    csv_bytes = csv_data.to_csv(index=False).encode('utf-8')
    st.download_button("Unduh CSV", csv_bytes, "klasifikasi_dataset.csv", "text/csv")
