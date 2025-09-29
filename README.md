# Implementation-of-Deep-Learning-Machine-Learning-Method-CNN-RF-for-Automatic-Waste-Sorting-System
Sebuah sistem untuk mengklasifikasikan sampah secara otomatis dari gambar menggunakan model hybrid CNN dan Random Forest.

## Fitur

  * **Klasifikasi Berbasis Gambar**: Mengklasifikasikan sampah ke dalam berbagai kategori.
  * **Model Hybrid**: Menggabungkan Convolutional Neural Network (CNN) untuk ekstraksi fitur dan Random Forest (RF) untuk klasifikasi.
  * **Akurasi Tinggi**: Memanfaatkan CNN pre-trained untuk pengenalan fitur yang kuat.
  * **Antarmuka Web**: Dibuat dengan Streamlit untuk interaksi yang mudah.

## Kategori Sampah yang Diklasifikasi

  * **Organik**
  * **Kertas**
  * **Kardus**
  * **Plastik**
  * **Kaca**
  * **Logam**

## Instalasi

**Prasyarat**

  * Python 3.8 atau lebih tinggi
  * Manajer paket `pip`

**Setup**

1.  **Clone repository:**

    ```bash
    git clone https://github.com/ThareeqZiadRamadhan/Implementation-of-Deep-Learning-MachineLearning-Method-CNN-and-RF-for-Automatic-Waste-Sorting-System.git
    cd Implementation-of-Deep-Learning-MachineLearning-Method-CNN-and-RF-for-Automatic-Waste-Sorting-System
    ```

2.  **Buat virtual environment:**

    ```bash
    python -m venv .venv
    ```

3.  **Aktifkan virtual environment:**

      * **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
      * **Linux/Mac:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install dependensi:**

    ```bash
    pip install -r requirements.txt
    ```

## Penggunaan

1.  **Jalankan aplikasi:**

    ```bash
    streamlit run app.py
    ```

2.  Buka browser Anda dan pergi ke `http://localhost:8501`.

3.  Unggah gambar sampah (PNG, JPG, JPEG).

4.  Lihat hasil klasifikasi yang ditampilkan.

## Struktur Proyek

```
waste-sorting-system/
├── app.py              # Aplikasi utama Streamlit
├── train_model.py      # Script untuk melatih model
├── requirements.txt    # Dependensi Python
├── README.md           # Dokumentasi proyek
├── models/             # File model yang sudah dilatih
└── data/               # Dataset (contoh: TrashNet)
```

## Detail Algoritma

**Ekstraksi Fitur dengan CNN**

Sebuah CNN pre-trained (seperti MobileNetV2 atau VGG16) digunakan untuk memproses gambar input. Layer klasifikasi akhir dari model ini dihilangkan, dan output dari layer sebelumnya digunakan sebagai vektor fitur tingkat tinggi yang merepresentasikan gambar.

**Klasifikasi dengan Random Forest**

Vektor fitur ini kemudian dimasukkan ke dalam classifier Random Forest, yang dilatih untuk memetakan fitur ke kategori sampah akhir. Pendekatan ini efisien dan kuat.

## Dependensi

  * `streamlit`
  * `tensorflow`
  * `scikit-learn`
  * `opencv-python`
  * `pillow`
  * `numpy`

## Kontribusi

1.  Fork repository ini.
2.  Buat feature branch baru (`git checkout -b feature/FiturBaru`).
3.  Commit perubahan Anda (`git commit -am 'Menambahkan fitur baru'`).
4.  Push ke branch (`git push origin feature/FiturBaru`).
5.  Buat Pull Request baru.

## Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT - lihat file `LICENSE` untuk detailnya.

## Penulis

  * **Thareeq Ziad Ramadhan** - *Initial work*
