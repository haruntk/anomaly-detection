import cv2
import os
import numpy as np

def threshold_otsu(img_gray):
    # Otsu thresholding
    # threshold değeri otomatik bulunur
    _, thresh_img = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Olası gürültüyü azaltmak için morfolojik açma (isteğe bağlı)
    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)

    return thresh_img

def threshold_adaptive(img_gray):
    # Adaptive Thresholding (mean veya gaussian kullanılabilir)
    # Burada cv2.ADAPTIVE_THRESH_GAUSSIAN_C örneklenmiştir
    thresh_img = cv2.adaptiveThreshold(
        img_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # blok boyutu
        2    # sabit
    )

    # Olası gürültüyü azaltmak için morfolojik açma (isteğe bağlı)
    kernel = np.ones((3, 3), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)

    return thresh_img

def preprocess_image(input_path, output_path, threshold_func):
    # 1) Görüntüyü okuma (Renkli okuyoruz, sonra griye çeviriyoruz)
    img = cv2.imread(input_path)
    if img is None:
        print(f"Hata: '{input_path}' okunamadı.")
        return

    # 2) Griye dönüştürme
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3) Thresholding (arka plan temizleme)
    thresh_img = threshold_func(gray)

    # 4) 256x256 boyutlandırma
    resized_img = cv2.resize(thresh_img, (256, 256), interpolation=cv2.INTER_AREA)

    # 5) [0,1] aralığında normalize etme (float32 tipinde)
    #    - Model eğitiminde numpy array olarak kullanacaksanız .npy olarak kaydedebilirsiniz.
    #    - Ancak yine PNG/JPEG olarak kaydedecekseniz [0,255] aralığına geri ölçekleyip
    #      uint8 dönüştürerek yazmak gerekir.
    normalized_img = resized_img.astype(np.float32) / 255.0

    # Çıktıyı PNG olarak kaydedeceğiz (uint8'e geri dönüyoruz)
    # Görsel olarak incelenebilir olsun diye
    output_img = (normalized_img * 255).astype(np.uint8)

    # Çıktıyı yazma
    cv2.imwrite(output_path, output_img)

def process_dataset(root_dir, output_root_dir, threshold_func=threshold_otsu):
    # Klasör isimlerini tanımlıyoruz (train, test, ground_truth)
    sub_dirs = ['train', 'test', 'ground_truth']
    # train içindeki alt klasörler: good
    # test içindeki alt klasörler: good, defect
    # ground_truth içindeki alt klasör: defect

    for sub in sub_dirs:
        current_dir = os.path.join(root_dir, sub)
        if not os.path.exists(current_dir):
            continue  # ilgili klasör yoksa atla

        # Klasörün alt klasörlerini oku (örn. good, defect)
        for sub_sub in os.listdir(current_dir):
            sub_sub_path = os.path.join(current_dir, sub_sub)
            if not os.path.isdir(sub_sub_path):
                continue

            # Çıktı klasörünü oluşturma
            output_dir = os.path.join(output_root_dir, sub, sub_sub)
            os.makedirs(output_dir, exist_ok=True)

            # Bu alt klasördeki tüm görüntüleri işleyelim
            for img_name in os.listdir(sub_sub_path):
                # Görsel uzantısı kontrolü (jpg, png vs.)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    input_img_path = os.path.join(sub_sub_path, img_name)

                    # Çıktı dosya ismini belirleyelim
                    output_img_path = os.path.join(output_dir, img_name)

                    # Preprocess ve kaydet
                    preprocess_image(input_img_path, output_img_path, threshold_func)

if __name__ == "__main__":
    # Veri seti kök klasörü (Sizin verdiğiniz örnek: Wood_dataset/wood)
    root_dataset_dir = "C:/Users/harun_rvth/OneDrive/Desktop/courses/3/bahar/neural-networks/final_project/wood_dataset/wood"

    # Çıktı klasörü (örn: Wood_dataset_preprocessed/wood)
    output_dataset_dir = "C:/Users/harun_rvth/OneDrive/Desktop/courses/3/bahar/neural-networks/final_project/wood_dataset/processed"

    # Otsu thresholding kullanmak isterseniz:
    process_dataset(root_dataset_dir, output_dataset_dir, threshold_func=threshold_otsu)

    # Eğer adaptif thresholding denemek isterseniz, aşağıdaki satırı kullanabilirsiniz:
    # process_dataset(root_dataset_dir, output_dataset_dir, threshold_func=threshold_adaptive)

    print("Tüm görüntüler başarıyla işlenmiştir!")
