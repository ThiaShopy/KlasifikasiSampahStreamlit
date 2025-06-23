import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from src.utils import download_file_if_not_exists

# Konfigurasi
IMG_SIZE = (64, 64)
MODEL_PATH = "waste_classifier_jepang.h5"
MODEL_FILE_ID = "1SK6bIDezZQvNGRZz1-NTi0Epg5slpG6Z"  # ID model terbaru dari Google Drive

# Label mapping
CLASS_NAMES = ['moeru_gomi', 'moenai_gomi', 'shigen_gomi']
CATEGORY_INFO = {
    'moeru_gomi': {
        'jp': '燃えるゴミ',
        'desc': 'Burnable Garbage – Sampah yang dapat dibakar (misalnya tisu, sisa makanan)'
    },
    'moenai_gomi': {
        'jp': '燃えないゴミ',
        'desc': 'Non-Burnable Garbage – Sampah logam/kaca seperti sendok atau botol kaca'
    },
    'shigen_gomi': {
        'jp': '資源ゴミ',
        'desc': 'Recyclable Garbage – Sampah seperti kardus, plastik, kertas, dan botol PET'
    }
}

# Fungsi prediksi gambar
def predict_image(img_path):
    # Pastikan model tersedia
    download_file_if_not_exists(MODEL_FILE_ID, MODEL_PATH)

    # Load dan proses gambar
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Load model dan prediksi
    model = load_model(MODEL_PATH)
    pred = model.predict(img_array)
    pred_idx = np.argmax(pred)
    class_name = CLASS_NAMES[pred_idx]

    # Tampilkan hasil
    print(f"\n✅ Gambar diklasifikasikan sebagai:")
    print(f"   {CATEGORY_INFO[class_name]['jp']} — {CATEGORY_INFO[class_name]['desc']}")

# Entry point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gunakan: python predict.py path/to/image.jpg")
    else:
        predict_image(sys.argv[1])
