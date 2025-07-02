import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 1. Muat model CNN yang sudah dilatih
cnn_model = tf.keras.models.load_model("model_cnn.h5")

# 2. Muat model Random Forest yang sudah dilatih
rf_model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler_rf.pkl")  # Memuat scaler yang digunakan selama pelatihan RF

# Nama kelas berdasarkan dataset
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]  # Sesuaikan dengan dataset kamu

# Fungsi untuk memproses gambar untuk CNN
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch
    img_array = img_array / 255.0  # Normalisasi
    return img_array

# Fungsi untuk memproses gambar untuk Random Forest (ekstraksi fitur)
def extract_features_for_rf(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Ekstraksi fitur menggunakan CNN
    features = cnn_model.predict(img_array)  # Output CNN, misalnya setelah Global Average Pooling (GAP)
    
    # Reshape atau flatten fitur jika perlu, agar sesuai dengan input Random Forest
    features = features.flatten()  # Mengubah fitur menjadi vektor 1D
    
    # Normalisasi fitur untuk sesuai dengan pelatihan model RF
    features = scaler.transform([features])  # Standarkan fitur dengan scaler yang sama
    
    return features

# Tampilan aplikasi Streamlit
st.title("Klasifikasi Gambar dengan CNN dan Random Forest")
st.markdown("Pilih gambar untuk diklasifikasikan")

# Membuat folder tempDir jika tidak ada
if not os.path.exists("tempDir"):
    os.makedirs("tempDir")

# Upload gambar
uploaded_image = st.file_uploader("Pilih gambar...", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Gambar yang Diupload", use_column_width=True)
    st.write("")
    st.write("Memproses gambar...")

    # Simpan gambar sementara
    image_path = os.path.join("tempDir", "uploaded_image.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Proses gambar untuk CNN
    img_array = preprocess_image(image_path)

    # Prediksi dengan CNN
    cnn_pred = cnn_model.predict(img_array)
    cnn_class_idx = np.argmax(cnn_pred)
    cnn_class = class_names[cnn_class_idx]
    cnn_confidence = cnn_pred[0][cnn_class_idx]

    st.write(f"**Prediksi dengan CNN:** {cnn_class} (Confidence: {cnn_confidence:.2f})")

    # Proses gambar untuk Random Forest
    rf_features = extract_features_for_rf(image_path)
    
    # Periksa dimensi fitur yang diekstraksi untuk memastikan kesesuaian
    if len(rf_features[0]) == 17:  # Pastikan jumlah fitur sesuai dengan yang diinginkan oleh Random Forest
        rf_pred = rf_model.predict(rf_features)
        rf_class = class_names[rf_pred[0]]
        st.write(f"**Prediksi dengan Random Forest:** {rf_class}")
    else:
        st.write("Jumlah fitur yang diekstraksi tidak sesuai dengan yang diharapkan oleh model Random Forest.")
