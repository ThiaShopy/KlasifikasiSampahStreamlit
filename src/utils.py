import os
import requests, zipfile
from io import BytesIO

def download_file_if_not_exists(file_id, target_path):
    if os.path.exists(target_path):
        print("✅ File sudah ada, skip download.")
        return

    print(f"⬇️  Mengunduh dari Google Drive: {target_path}")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    response.raise_for_status()

    with open(target_path, 'wb') as f:
        f.write(response.content)
    print("✅ Unduhan selesai.")
