import streamlit as st
import tensorflow as tf
import os

MODEL_PATH = "model_cnn_transfer_learning.h5"

st.title("Tes Pemuatan Model")

if not os.path.exists(MODEL_PATH):
    st.error(f"File model '{MODEL_PATH}' tidak ditemukan!")
else:
    st.info(f"Mencoba memuat model dari '{MODEL_PATH}'...")
    try:
        # Ini adalah baris yang paling penting
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        st.success("✅ BERHASIL! Model sukses dimuat tanpa masalah.")
        st.balloons()
        st.code(model.summary()) # Menampilkan ringkasan arsitektur model
        
    except Exception as e:
        st.error(f"❌ GAGAL! Terjadi error saat memuat model:")
        st.exception(e)
