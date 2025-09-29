import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# --- Konfigurasi ---
IMAGE_DATASET_PATH = "archive"
CNN_MODEL_PATH = "models/model_cnn.keras" # Path ke model CNN Anda yang sudah dilatih
OUTPUT_CSV_PATH = "data/dataset_cnn_features.csv" # Nama file output BARU
TARGET_SIZE = (128, 128)

def create_cnn_feature_dataset():
    """
    Mengekstrak fitur dari seluruh dataset gambar menggunakan model CNN
    dan menyimpannya ke dalam file CSV.
    """
    # 1. Muat model CNN dan ubah menjadi Feature Extractor
    print(f"🔄 Memuat model CNN dari '{CNN_MODEL_PATH}'...")
    try:
        full_cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
        
        # Buang lapisan output terakhir untuk mendapatkan fitur dari lapisan sebelumnya
        # Berdasarkan model Anda, lapisan sebelum output adalah Dense(128)
        last_layer_name = full_cnn_model.layers[-2].name
        feature_extractor = tf.keras.Model(
            inputs=full_cnn_model.input, 
            outputs=full_cnn_model.get_layer(last_layer_name).output
        )
        print(f"✅ Model feature extractor siap. Output shape: {feature_extractor.output_shape}")
    except Exception as e:
        print(f"❌ Gagal memuat model CNN: {e}")
        return

    # 2. Fungsi untuk memproses satu gambar
    def extract_features_from_image(image_path):
        try:
            # Muat dan preprocess gambar
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=TARGET_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Buat batch size 1
            
            # Ekstrak fitur menggunakan CNN
            features = feature_extractor.predict(img_array, verbose=0)
            return features.flatten() # Ratakan menjadi 1D array
        except Exception as e:
            print(f"\n⚠️ Gagal memproses gambar {image_path}: {e}")
            return None

    # 3. Iterasi melalui seluruh dataset untuk mengekstrak fitur
    print("\n🏭 Memulai proses ekstraksi fitur dari seluruh dataset...")
    all_features = []
    all_labels = []
    all_paths = []

    class_folders = sorted([d for d in os.listdir(IMAGE_DATASET_PATH) if os.path.isdir(os.path.join(IMAGE_DATASET_PATH, d))])

    for class_name in class_folders:
        class_path = os.path.join(IMAGE_DATASET_PATH, class_name)
        image_files = os.listdir(class_path)
        
        for image_file in tqdm(image_files, desc=f"-> {class_name}"):
            image_path = os.path.join(class_path, image_file)
            
            features = extract_features_from_image(image_path)
            
            if features is not None:
                all_features.append(features)
                all_labels.append(class_name)
                all_paths.append(image_path)

    # 4. Simpan hasil ke dalam file CSV
    print("\n💾 Menyimpan fitur ke dalam file CSV...")
    
    # Buat DataFrame dari fitur (setiap fitur menjadi satu kolom)
    df_features = pd.DataFrame(all_features)
    
    # Buat DataFrame untuk label dan path
    df_info = pd.DataFrame({
        'label': all_labels,
        'image_path': all_paths
    })
    
    # Gabungkan kedua DataFrame
    df_final = pd.concat([df_info, df_features], axis=1)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"\n🎉 Selesai! Dataset fitur CNN berhasil dibuat di: {OUTPUT_CSV_PATH}")
    print(f"   Total data: {len(df_final)} gambar")
    print(f"   Jumlah fitur per gambar: {df_features.shape[1]}")

if __name__ == "__main__":
    create_cnn_feature_dataset()
