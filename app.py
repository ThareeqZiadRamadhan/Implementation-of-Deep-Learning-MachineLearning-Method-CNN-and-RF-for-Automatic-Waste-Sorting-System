import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import joblib

# --- Konfigurasi Global ---
CNN_EXTRACTOR_PATH = "models/model_cnn.keras" 
RF_CLASSIFIER_PATH = "models/model_rf_from_cnn.pkl" 
TARGET_SIZE = (128, 128)
CLASS_NAMES_RF = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_NAMES_DISPLAY = ["Kardus", "Kaca", "Logam", "Kertas", "Plastik", "Organik/Lainnya"]

# --- Fungsi CSS Kustom ---
def apply_custom_styling():
    """Menyuntikkan CSS kustom untuk tampilan yang unik."""
    styling = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
            
            html, body, [class*="st-"] { font-family: 'Poppins', sans-serif; }

            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 3rem;
                padding-right: 3rem;
            }

            [data-testid="stSidebar"] {
                background-color: #0E1117;
                border-right: 1px solid #262730;
                padding: 1rem;
            }
            
            [data-testid="stSidebar"] [data-testid="stAlert"] {
                background-color: rgba(40, 167, 69, 0.15);
                border-left: 5px solid #28a745;
            }

            .result-card {
                background-color: #0E1117;
                border-radius: 20px;
                padding: 25px;
                border: 1px solid #262730;
                box-shadow: 0 8px 16px rgba(0,0,0,0.05);
                transition: transform 0.2s;
            }
            .result-card:hover {
                transform: translateY(-5px);
                border-color: #007bff;
            }

            .prediction-header { font-size: 1.1rem; color: #a0a4ac; margin-bottom: 5px; }
            .prediction-value { font-size: 2.8rem; font-weight: 700; color: #fafafa; line-height: 1.1; }
            .confidence-text { font-size: 1rem; font-weight: 600; color: #28a745; }

            .advice-section {
                margin-top: 20px;
                padding: 15px;
                background-color: #161a22;
                border-radius: 10px;
                display: flex;
                align-items: center;
                gap: 15px;
                border-left: 5px solid #007bff;
            }
            .advice-icon { font-size: 1.8rem; }
            .advice-tip { font-size: 0.95rem; }

            .welcome-container {
                text-align: center;
                padding: 50px;
                border: 2px dashed #31333F;
                border-radius: 15px;
                margin-top: 2rem;
            }
            .welcome-icon { font-size: 4rem; }
            
            /* --- PERBAIKAN EXPANDER FINAL --- */
            [data-testid="stExpander"] {
                border: 1px solid #31333F !important;
                border-radius: 12px !important;
                background-color: #0E1117 !important;
                margin-top: 1.5rem;
            }
            /* Menargetkan header dari expander untuk memperbaiki layout dan bug visual */
            [data-testid="stExpander"] summary {
                display: flex;
                align-items: center;
                gap: 12px; /* Jarak antara panah dan teks */
            }
            [data-testid="stExpander"] summary p {
                font-size: 1.1rem !important;
                font-weight: 600;
                color: #fafafa !important;
            }
            /* ------------------------- */

        </style>
    """
    st.markdown(styling, unsafe_allow_html=True)

# --- Fungsi-fungsi Backend (Tidak berubah) ---
@st.cache_resource
def load_models():
    """Memuat kedua model untuk alur kerja hybrid."""
    cnn_extractor = None
    rf_model = None
    try:
        if os.path.exists(CNN_EXTRACTOR_PATH):
            full_cnn = tf.keras.models.load_model(CNN_EXTRACTOR_PATH, compile=False)
            last_layer_name = full_cnn.layers[-2].name 
            cnn_extractor = tf.keras.Model(inputs=full_cnn.input, outputs=full_cnn.get_layer(last_layer_name).output)
        else: st.sidebar.error("File CNN tidak ditemukan.")
    except Exception as e: st.sidebar.error(f"Gagal memuat CNN: {e}")
    try:
        if os.path.exists(RF_CLASSIFIER_PATH):
            rf_model = joblib.load(RF_CLASSIFIER_PATH)
        else: st.sidebar.error("File RF tidak ditemukan.")
    except Exception as e: st.sidebar.error(f"Gagal memuat RF: {e}")
    return cnn_extractor, rf_model

def predict_hybrid(cnn_extractor, rf_model, uploaded_image):
    try:
        image = Image.open(uploaded_image).convert("RGB")
        image_resized = image.resize(TARGET_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
        img_array = np.expand_dims(img_array, axis=0)
        features = cnn_extractor.predict(img_array, verbose=0)
        predicted_class_name = rf_model.predict(features)[0]
        class_index = CLASS_NAMES_RF.index(predicted_class_name)
        prediction_probs = rf_model.predict_proba(features)[0]
        confidence = 100 * np.max(prediction_probs)
        all_probs = sorted([(CLASS_NAMES_DISPLAY[i], 100 * prob) for i, prob in enumerate(prediction_probs)], key=lambda x: x[1], reverse=True)
        return class_index, confidence, all_probs
    except Exception as e:
        st.error(f"❌ Error saat memproses gambar: {e}")
        st.exception(e)
        return None, None, None

def get_disposal_advice(predicted_class):
    advice = {
        "Kardus": {"icon": "📦", "tip": "Dapat didaur ulang. Pastikan dalam keadaan kering & bersih."},
        "Kertas": {"icon": "📄", "tip": "Dapat didaur ulang. Pastikan tidak terkena minyak atau cairan."},
        "Kaca": {"icon": "🪟", "tip": "Dapat didaur ulang. Bersihkan dari sisa produk."},
        "Logam": {"icon": "🔩", "tip": "Dapat didaur ulang. Kumpulkan dan buang di tempat khusus."},
        "Plastik": {"icon": "🥤", "tip": "Dapat didaur ulang. Bersihkan dan buang ke tempat daur ulang."},
        "Organik/Lainnya": {"icon": "🗑️", "tip": "Tidak dapat didaur ulang. Buang ke TPA atau jadikan kompos."}
    }
    return advice.get(predicted_class, advice["Organik/Lainnya"])

# --- UI Aplikasi Streamlit ---
st.set_page_config(page_title="Klasifikasi Sampah Cerdas", page_icon="♻️", layout="wide")
apply_custom_styling()

# --- Sidebar ---
with st.sidebar:
    st.title("♻️ Klasifikasi Sampah")
    st.markdown("Unggah gambar sampah, dan biarkan model hybrid **CNN + Random Forest** kami menganalisisnya.")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown("---")
    st.subheader("Status Model")
    cnn_extractor, rf_model = load_models()
    if cnn_extractor and rf_model:
        st.success("Semua model siap!")
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #a0a4ac;'>Dibuat oleh<br><b>Thareeq Ziad R.</b></p>", unsafe_allow_html=True)

# --- Halaman Utama ---
st.header("Analisis & Klasifikasi Sampah Cerdas")
st.markdown("Selamat datang! Aplikasi ini menggunakan AI untuk mengidentifikasi jenis sampah dan memberikan saran penanganan yang tepat.")

if cnn_extractor and rf_model:
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)
        
        with col2:
            with st.spinner('Menganalisis dengan model hybrid...'):
                class_index, confidence, all_probs = predict_hybrid(cnn_extractor, rf_model, uploaded_file)
            
            if class_index is not None:
                predicted_class = CLASS_NAMES_DISPLAY[class_index]
                advice = get_disposal_advice(predicted_class)
                
                st.markdown(f"""
                <div class="result-card">
                    <p class="prediction-header">Prediksi Jenis Sampah</p>
                    <p class="prediction-value">{predicted_class}</p>
                    <p class="confidence-text">{confidence:.1f}% Keyakinan</p>
                    <div class="advice-section">
                        <span class="advice-icon">{advice['icon']}</span>
                        <span class="advice-tip"><b>Saran Penanganan:</b> {advice['tip']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Lihat Detail Probabilitas"):
                    for class_name, prob in all_probs:
                        st.text(f"{class_name}: {prob:.2f}%")
                        st.progress(int(prob))
    else:
        st.markdown("""
        <div class="welcome-container">
            <span class="welcome-icon">📤</span>
            <h2>Mulai Analisis Sampah</h2>
            <p>Unggah gambar sampah Anda melalui panel di sebelah kiri.<br>
            Model kami akan mengklasifikasikannya untuk Anda.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.error("⚠️ Model tidak dapat dimuat. Silakan periksa pesan error di sidebar dan pastikan file model sudah benar.")

