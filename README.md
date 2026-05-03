# Klasifikasi Kucing vs Anjing

Proyek ini merupakan implementasi model Convolutional Neural Network (CNN) untuk klasifikasi gambar kucing dan anjing menggunakan transfer learning dengan arsitektur MobileNetV2. Model diintegrasikan ke dalam aplikasi web berbasis Flask yang menyediakan antarmuka untuk mengunggah gambar dan mendapatkan hasil prediksi secara real-time.

## Tech Stack

- **Framework Web:** Flask 3.1.0
- **Machine Learning:** TensorFlow 2.18.0 + Keras Tuner 1.4.7
- **Pre-trained Model:** MobileNetV2 (ImageNet weights)
- **Frontend:** HTML + Tailwind CSS (CDN)
- **Python:** 3.11 atau 3.12

## Fitur Utama

- Transfer learning dengan MobileNetV2
- Hyperparameter tuning otomatis menggunakan Keras Tuner (Hyperband)
- Data augmentation untuk meningkatkan generalisasi model
- Fine-tuning dua fase pada layer MobileNetV2
- Aplikasi web minimalis dengan Flask
- Error handling menyeluruh pada backend

## Struktur Direktori

```
pet-prediction/
├── data/
│   └── PetImages/
│       ├── Cat/
│       └── Dog/
├── models/
│   └── best_model.keras
├── src/
│   ├── clean_dataset.py
│   ├── train_model.py
│   └── tune_hyperparameters.py
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
├── .gitignore
```

## Cara Menjalankan

### 1. Persiapan Environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Bersihkan Dataset

```bash
python src/clean_dataset.py
```

### 3. Latih Model

Pilih salah satu metode pelatihan:

**Opsi A: Pelatihan dengan Hyperparameter Tuning**

```bash
python src/tune_hyperparameters.py
```

**Opsi B: Pelatihan dengan Hyperparameter Manual**

```bash
python src/train_model.py
```

### 4. Jalankan Aplikasi Web

```bash
python app.py
```

Akses aplikasi melalui browser di `http://127.0.0.1:5000`.

### 5. Production Deployment

Untuk deployment production, gunakan Gunicorn sebagai WSGI server:

```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

## Informasi Penting

- TensorFlow belum mendukung Python 3.14. Gunakan Python 3.11 atau 3.12 untuk menghindari masalah kompatibilitas.
- File model `best_model.keras` memiliki ukuran sekitar 28 MB dan tidak perlu dilatih ulang jika sudah tersedia.
- Dataset tidak perlu diunggah ke repository karena sudah diatur pada `.gitignore`.
- Folder `data/` memerlukan ruang penyimpanan sekitar 1 GB karena berisi ribuan gambar.
