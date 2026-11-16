# Prediksi Penyakit Jantung (CRISP-DM)

Dataset: UCI Heart Disease (Cleveland). Hanya untuk edukasi, bukan alat diagnosis.

## Cara menjalankan

1. Pasang Python 3.10+ lalu pasang paket:
   ```
   pip install -r requirements.txt
   ```
2. Latih model (unduh data otomatis dari UCI):
   ```
   python train_model.py
   ```
   Jika tidak ada internet atau `ucimlrepo` bermasalah, taruh file `processed.cleveland.data` di folder `data/` lalu jalankan:
   ```
   python train_model.py --csv data/processed.cleveland.data
   ```
3. Jalankan aplikasi:
   ```
   streamlit run app.py
   ```
   Browser akan terbuka di http://localhost:8501.

## Alternatif tanpa instalasi
Buka `heart_risk_app.html` langsung di browser (klik dua kali). Versi ini sudah berisi model, jadi tidak perlu Python.

## Sitasi
Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
