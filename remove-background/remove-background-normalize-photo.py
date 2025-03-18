import cv2
import os
import numpy as np

def threshold_otsu(img_gray):
    _, thresh_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)

    return thresh_img

def threshold_adaptive(img_gray):
    thresh_img = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 
        2    
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)

    return thresh_img

def preprocess_image(input_path, output_path, threshold_func):
    img = cv2.imread(input_path)
    if img is None:
        print(f"Hata: '{input_path}' okunamadı.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh_img = threshold_func(gray)

    resized_img = cv2.resize(thresh_img, (256, 256), interpolation=cv2.INTER_AREA)
    normalized_img = resized_img.astype(np.float32) / 255.0

    output_img = (normalized_img * 255).astype(np.uint8)
    cv2.imwrite(output_path, output_img)

def process_dataset(root_dir, output_root_dir, threshold_func=threshold_otsu):
    sub_dirs = ['train', 'test', 'ground_truth']

    for sub in sub_dirs:
        current_dir = os.path.join(root_dir, sub)
        if not os.path.exists(current_dir):
            continue  

        for sub_sub in os.listdir(current_dir):
            sub_sub_path = os.path.join(current_dir, sub_sub)
            if not os.path.isdir(sub_sub_path):
                continue

            output_dir = os.path.join(output_root_dir, sub, sub_sub)
            os.makedirs(output_dir, exist_ok=True)

            for img_name in os.listdir(sub_sub_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    input_img_path = os.path.join(sub_sub_path, img_name)

                    output_img_path = os.path.join(output_dir, img_name)

                    preprocess_image(input_img_path, output_img_path, threshold_func)

if __name__ == "__main__":
    root_dataset_dir = "C:/Users/harun_rvth/OneDrive/Desktop/courses/3/bahar/neural-networks/final_project/wood_dataset/wood"

    output_dataset_dir = "C:/Users/harun_rvth/OneDrive/Desktop/courses/3/bahar/neural-networks/final_project/wood_dataset/processed"

    process_dataset(root_dataset_dir, output_dataset_dir, threshold_func=threshold_otsu)
    
    print("Tüm görüntüler başarıyla işlenmiştir!")
