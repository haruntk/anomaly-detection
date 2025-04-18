# -*- coding: utf-8 -*-
"""PaDim.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17v9hXyRdtk3eEnzkY10aDTm9WdIh7s8T
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1️⃣ **Görüntü Veri Kümesi**
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, img_path

# 2️⃣ **Dönüşümler & Veri Yükleyiciler**
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dir = '/content/drive/MyDrive/processed/train/good'
test_good_dir = '/content/drive/MyDrive/processed/test/good'
test_defect_dir = '/content/drive/MyDrive/processed/test/defect'

train_dataset = ImageDataset(train_dir, transform=transform)
test_good_dataset = ImageDataset(test_good_dir, transform=transform)
test_defect_dataset = ImageDataset(test_defect_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_good_loader = DataLoader(test_good_dataset, batch_size=16, shuffle=False)
test_defect_loader = DataLoader(test_defect_dataset, batch_size=16, shuffle=False)

# 3️⃣ **Özellik Çıkartma (ResNet18)**
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-2])
feature_extractor.eval().cuda()

def extract_features(dataloader):
    features_list = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.cuda()
            features = feature_extractor(images)
            features = features.view(features.shape[0], features.shape[1], -1)
            features = features.permute(0, 2, 1)
            features_list.append(features.cpu().numpy())
    return np.concatenate(features_list, axis=0)

train_features = extract_features(train_loader)

# 4️⃣ **Ortalama ve Kovaryans Hesaplama**
mean = np.mean(train_features, axis=0)
cov = np.array([np.cov(train_features[:, i, :], rowvar=False) for i in range(train_features.shape[1])])
cov_inv = np.linalg.inv(cov + np.eye(cov.shape[1]) * 1e-6)

def anomaly_score(features):
    scores = []
    for i in range(features.shape[1]):
        delta = features[:, i, :] - mean[i]
        score = np.einsum('ij,jk,ik->i', delta, cov_inv[i], delta)  # Mahalanobis mesafesi
        scores.append(score)
    return np.max(np.array(scores), axis=0)  # En büyük skoru al

test_good_features = extract_features(test_good_loader)
test_defect_features = extract_features(test_defect_loader)

anomaly_scores_good = anomaly_score(test_good_features)
anomaly_scores_defect = anomaly_score(test_defect_features)

def plot_scores(good_scores, defect_scores):
    plt.figure(figsize=(10, 5))
    plt.hist(good_scores, bins=50, alpha=0.7, label='Good')
    plt.hist(defect_scores, bins=50, alpha=0.7, label='Defect')
    plt.axvline(np.percentile(good_scores, 86), color='red', linestyle='--', label='Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Anomaly Score Distribution')
    plt.show()

plot_scores(anomaly_scores_good, anomaly_scores_defect)

def extract_single_image_features(image_path, feature_extractor, transform):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).cuda()  # Model için uygun forma getir

    with torch.no_grad():
        features = feature_extractor(image)
        features = features.view(features.shape[0], features.shape[1], -1)
        features = features.permute(0, 2, 1)

    return features.cpu().numpy()

def predict_anomaly(image_path, mean, cov_inv, feature_extractor, transform):
    features = extract_single_image_features(image_path, feature_extractor, transform)
    score = anomaly_score(features)  # Anomali skoru hesapla

    return score

threshold = np.percentile(anomaly_scores_good, 85)

image_path = "/content/drive/MyDrive/processed/test/defect/100000006.jpg"
score = predict_anomaly(image_path, mean, cov_inv, feature_extractor, transform)

print(f"Anomaly Score: {score}")

if score > threshold:
    print("🚨 Görüntü KUSURLU!")
else:
    print("✅ Görüntü İYİ.")

"""Dogruluk"""

threshold = np.percentile(anomaly_scores_good, 86)

# Gerçek etiketler (1: Kusurlu, 0: İyi)
y_true = np.concatenate([
    np.zeros(len(anomaly_scores_good)),  # İyi örnekler (label = 0)
    np.ones(len(anomaly_scores_defect))  # Kusurlu örnekler (label = 1)
])

# Model tahminleri (Eşik değer üzerindekiler kusurlu sayılacak)
y_pred = np.concatenate([
    (anomaly_scores_good > threshold).astype(int),
    (anomaly_scores_defect > threshold).astype(int)
])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

def normalize_map(anomaly_map):
    return (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)

def predict_anomaly_map(image_path, mean, cov_inv, feature_extractor, transform):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb).unsqueeze(0).to(device)  # [1, C, H, W]
    with torch.no_grad():
        features = feature_extractor(img_tensor)  # [1, 512, 8, 8]
    B, C, H, W = features.shape
    features_reshaped = features.view(B, C, H * W).permute(0, 2, 1)   # [1, 64, 512]
    features_np = features_reshaped.cpu().numpy()[0]  # [64, 512]

    patch_scores = []
    for i in range(features_np.shape[0]):
        delta = features_np[i] - mean[i]  # [512]
        score = np.dot(np.dot(delta, cov_inv[i]), delta)  # skaler
        patch_scores.append(score)
    patch_scores = np.array(patch_scores)
    anomaly_map = patch_scores.reshape(H, W)
    anomaly_map = normalize_map(anomaly_map)
    anomaly_map_up = cv2.resize(anomaly_map, (256, 256))
    return anomaly_map_up, image_rgb

def compute_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(np.uint8)
    gt_mask = gt_mask.astype(np.uint8)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0

def evaluate_iou(image_path, gt_mask_path, mean, cov_inv, feature_extractor, transform, pred_threshold=0.5):
    anomaly_map, _ = predict_anomaly_map(image_path, mean, cov_inv, feature_extractor, transform)
    anomaly_map_norm = normalize_map(anomaly_map)
    pred_mask = (anomaly_map_norm > pred_threshold).astype(np.uint8)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"❌ Hata: Ground truth mask '{gt_mask_path}' bulunamadı!")
        return None, anomaly_map
    gt_mask = cv2.resize(gt_mask, (256, 256))
    gt_mask = (gt_mask > 127).astype(np.uint8)
    iou = compute_iou(pred_mask, gt_mask)
    return iou, anomaly_map

# Örnek kullanım:
image_path = "/content/drive/MyDrive/processed/test/defect/100000007.jpg"
gt_mask_path = "/content/drive/MyDrive/processed/ground_truth/defect/100000007_mask.jpg"

iou_score, anomaly_map = evaluate_iou(image_path, gt_mask_path, mean, cov_inv, feature_extractor, transform, pred_threshold=0.5)
if iou_score is not None:
    print(f"IoU Score: {iou_score:.4f}")

plt.figure(figsize=(10,5))
plt.imshow(anomaly_map, cmap="jet")
plt.title("Anomaly Map")
plt.colorbar()
plt.show()