import os
import numpy as np
import requests, zipfile
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from src.model import build_cnn_model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight

# Konfigurasi
GOOGLE_DRIVE_FILE_ID = "13V9MWX2YeuqUzduSsG5fb9CEL2MtneMr"
EXTRACTED_DIR = "extracted_data/jepang"
IMG_SIZE = (64, 64)
CLASS_NAMES = ['moeru_gomi', 'moenai_gomi', 'shigen_gomi']
LABEL_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def download_and_extract_dataset():
    if os.path.exists(EXTRACTED_DIR):
        print("✅ Dataset sudah ada, skip download.")
        return

    print("⬇️  Mengunduh dataset dari Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall("extracted_data")
    print("✅ Dataset berhasil diekstrak.")


def load_images():
    download_and_extract_dataset()
    images, labels = [], []
    for label_name in CLASS_NAMES:
        path = os.path.join(EXTRACTED_DIR, label_name)
        for fname in os.listdir(path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(path, fname)
                img = load_img(fpath, target_size=IMG_SIZE)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(LABEL_MAP[label_name])
    return np.array(images), np.array(labels)


def main():
    X, y = load_images()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hitung class weight
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    model = build_cnn_model(input_shape=(64, 64, 3), num_classes=3)
    model.summary()

    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=10, batch_size=32,
              class_weight=class_weights_dict)

    model.save("waste_classifier_jepang.h5")
    print("✅ Model saved as waste_classifier_jepang.h5")


if __name__ == "__main__":
    main()
