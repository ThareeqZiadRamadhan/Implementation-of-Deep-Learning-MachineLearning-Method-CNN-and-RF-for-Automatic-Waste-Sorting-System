import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load dataset
df = pd.read_csv("bt_dataset_t3.csv")

# 2. Hapus kolom 'Image' jika ada
df = df.drop(columns=["Image"], errors='ignore')

# 3. Tangani missing values pada target dan fitur
df_clean = df.dropna(subset=['Target'])  # Pastikan 'Target' tidak ada nilai yang hilang

# 4. Pisahkan fitur dan target
X = df_clean.drop(columns=["Target"])
y = df_clean["Target"]

# 5. Tangani nilai inf atau NaN dalam fitur
X = X.replace([np.inf, -np.inf], np.nan)  # Ganti inf dengan NaN
X = X.fillna(0)  # Ganti NaN dengan 0 atau bisa pilih metode lain

# 6. Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Latih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Evaluasi model
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 10. Simpan model dan scaler
joblib.dump(model, "model_rf.pkl")
joblib.dump(scaler, "scaler_rf.pkl")  # scaler penting untuk digunakan kembali di Streamlit

print("✅ Model dan scaler berhasil disimpan.")
