import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Konfigurasi ---
FEATURES_CSV_PATH = 'data/dataset_cnn_features.csv'
OUTPUT_MODEL_PATH = 'models/model_rf_from_cnn.pkl'

def train_random_forest():
    """
    Melatih model Random Forest menggunakan dataset fitur yang sudah diekstrak.
    """
    print("Mulai proses training Random Forest...")
    
    # 1. Memuat dataset fitur
    try:
        df = pd.read_csv(FEATURES_CSV_PATH)
        print(f"✅ Dataset fitur '{FEATURES_CSV_PATH}' berhasil dimuat. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ File '{FEATURES_CSV_PATH}' tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan 'extract_features.py' terlebih dahulu.")
        return
        
    # 2. Memisahkan fitur (X) dan label (y)
    X = df.drop(['label', 'image_path'], axis=1) 
    y = df['label']
    print(f"DEBUG: Jumlah fitur yang digunakan untuk training adalah: {X.shape[1]}")
    # 3. Membagi data menjadi data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data dibagi: {len(X_train)} training, {len(X_test)} testing.")
    
    # 4. Melatih model Random Forest
    print("🏃 Melatih model RandomForestClassifier...")
    # Anda bisa menyesuaikan parameter ini (n_estimators, max_depth, dll.)
    rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=20)
    rf_classifier.fit(X_train, y_train)
    print("✅ Training selesai.")
    
    # 5. Mengevaluasi model
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Hasil Evaluasi Model ---")
    print(f"Akurasi pada data test: {accuracy * 100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))
    
    # 6. MENYIMPAN MODEL YANG SUDAH DILATIH
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    joblib.dump(rf_classifier, OUTPUT_MODEL_PATH)
    print(f"\n🎉 Model Random Forest berhasil disimpan di: {OUTPUT_MODEL_PATH}")
    print("Sekarang Anda bisa menjalankan aplikasi 'app.py'.")

if __name__ == '__main__':
    train_random_forest()
