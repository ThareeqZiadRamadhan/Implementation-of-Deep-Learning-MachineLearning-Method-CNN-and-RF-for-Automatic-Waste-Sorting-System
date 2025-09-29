import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import pandas as pd # Diperlukan untuk beberapa operasi

# Ukuran target yang konsisten
TARGET_SIZE = (128, 128)

def extract_features(image_file_or_path):
    """
    Satu fungsi final untuk mengubah gambar menjadi 39 fitur yang dibutuhkan model RF,
    dengan urutan yang sudah diperbaiki.
    """
    features = {}
    
    try:
        # 1. BACA & PREPROCESS GAMBAR
        if isinstance(image_file_or_path, str):
            image = cv2.imread(image_file_or_path)
        else:
            # Untuk file yang diupload dari Streamlit/Flask
            image_stream = np.asarray(bytearray(image_file_or_path.read()), dtype=np.uint8)
            image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)

        if image is None: raise ValueError("Gagal membaca gambar")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, TARGET_SIZE)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2HSV)

        # 2. EKSTRAK SEMUA FITUR
        
        # --- Fitur Warna (16 Fitur) ---
        features['mean_r'] = np.mean(image_resized[:, :, 0])
        features['mean_g'] = np.mean(image_resized[:, :, 1])
        features['mean_b'] = np.mean(image_resized[:, :, 2])
        features['std_r'] = np.std(image_resized[:, :, 0])
        features['std_g'] = np.std(image_resized[:, :, 1])
        features['std_b'] = np.std(image_resized[:, :, 2])
        features['mean_h'] = np.mean(hsv[:, :, 0])
        features['mean_s'] = np.mean(hsv[:, :, 1])
        features['mean_v'] = np.mean(hsv[:, :, 2])
        features['std_h'] = np.std(hsv[:, :, 0])
        features['std_s'] = np.std(hsv[:, :, 1])
        features['std_v'] = np.std(hsv[:, :, 2])
        pixels = np.float32(image_resized.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]
        features['dominant_r'] = dominant[0]
        features['dominant_g'] = dominant[1]
        features['dominant_b'] = dominant[2]
        features['dominant_ratio'] = np.max(counts) / len(labels)

        # --- Fitur Tekstur & Kecerahan (8 Fitur) ---
        radius = 3
        n_points = 24
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features['lbp_uniformity'] = hist[n_points+1] # Uniformity bin
        features['lbp_contrast'] = np.std(lbp)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        features['edge_density'] = np.mean(np.sqrt(sobelx**2 + sobely**2))
        features['edge_std'] = np.std(np.sqrt(sobelx**2 + sobely**2))
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['contrast'] = gray.std()
        features['entropy'] = -np.sum(hist * np.log2(hist + 1e-7))

        # --- Fitur Bentuk & Geometris (8 Fitur) ---
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            features['compactness'] = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            features['aspect_ratio'] = w / h if h > 0 else 0
            features['bbox_ratio'] = w * h / (TARGET_SIZE[0] * TARGET_SIZE[1])
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = area / hull_area if hull_area > 0 else 0
            features['extent'] = area / (w * h) if w * h > 0 else 0
            features['object_area_ratio'] = area / (TARGET_SIZE[0] * TARGET_SIZE[1])
        else:
            for key in ['compactness', 'aspect_ratio', 'bbox_ratio', 'solidity', 'extent', 'object_area_ratio']: features[key] = 0
        features['image_width'] = TARGET_SIZE[0]
        features['image_height'] = TARGET_SIZE[1]
        features['image_ratio'] = TARGET_SIZE[0] / TARGET_SIZE[1]

        # --- Fitur Turunan/Simulasi (7 Fitur) ---
        # Diisi dengan nilai default karena tidak bisa dihitung dari gambar baru.
        features['estimated_density'] = 2.0
        features['estimated_weight'] = 80.0
        features['estimated_volume'] = 0.0
        features['reflectivity'] = 0.4
        features['transparency'] = 0.0
        features['hardness'] = 4.0
        
        # 3. MENGURUTKAN FITUR SESUAI "BLUEPRINT"
        feature_order = [
            'mean_r','mean_g','mean_b','std_r','std_g','std_b','mean_h','mean_s',
            'mean_v','std_h','std_s','std_v','dominant_r','dominant_g','dominant_b',
            'dominant_ratio','lbp_uniformity','lbp_contrast','edge_density',
            'edge_std','brightness_mean','brightness_std','contrast','entropy',
            'compactness','aspect_ratio','bbox_ratio','extent','solidity',
            'image_width','image_height','image_ratio','object_area_ratio',
            'estimated_density','estimated_weight','estimated_volume',
            'reflectivity','transparency','hardness'
        ]
        
        # Buat array final sesuai urutan
        final_features = np.array([features[key] for key in feature_order])
        
        # Pastikan outputnya (1, 39)
        return final_features.reshape(1, -1)

    except Exception as e:
        print(f"Error di dalam extract_features: {e}")
        return None