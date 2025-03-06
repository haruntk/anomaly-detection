import cv2
import os
import numpy as np
from glob import glob

data_dir = "="
output_dir = "-"

os.makedirs(output_dir, exist_ok=True)

def process_image(img_path, save_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img[thresh == 0] = [0, 0, 0]  # Arka planı siyah yap
    
    resized = cv2.resize(img, (256, 256))
    
    normalized = resized.astype(np.float32) / 255.0
    
    cv2.imwrite(save_path, (normalized * 255).astype(np.uint8))

for split in ["train/good", "test/good", "test/defect", "ground_truth/defect"]:
    img_paths = glob(os.path.join(data_dir, split, "*.jpg"))  # JPG dosyalarını al
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, split, img_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        process_image(img_path, save_path)

print("Tüm görüntüler işlendi ve kaydedildi.")
