"""Demo Streamlit: skrining risiko penyakit jantung (edukasi).

Jalankan:  streamlit run app.py   (setelah python train_model.py)
"""
import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cek Risiko Penyakit Jantung", page_icon="🫀")


@st.cache_resource
def load_model():
    model = joblib.load("models/heart_pipeline.joblib")
    meta = json.load(open("models/metadata.json"))
    return model, meta


try:
    model, meta = load_model()
except FileNotFoundError:
    st.error("Model belum ada. Jalankan dulu: python train_model.py")
    st.stop()

st.title("Cek risiko penyakit jantung")
st.warning("Hanya untuk edukasi. Aplikasi ini bukan alat diagnosis dan tidak "
           "menggantikan pemeriksaan dokter.")

UNKNOWN = "Tidak diketahui"
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Usia (tahun)", 20, 90, 55)
    gender = st.selectbox("Gender", [0, 1], index=1,
                          format_func=lambda v: "Perempuan" if v == 0 else "Laki-laki")
    trestbps = st.number_input("Tekanan darah istirahat (mm Hg)", 80, 220, 130)
    chol = st.number_input("Kolesterol serum (mg/dl)", 100, 600, 245)
    fbs = st.selectbox("Gula darah puasa > 120 mg/dl", [0, 1],
                       format_func=lambda v: "Ya" if v else "Tidak")
    cp = st.selectbox("Tipe nyeri dada", [1, 2, 3, 4], index=2,
                      format_func={1: "Angina tipikal", 2: "Angina atipikal",
                                   3: "Nyeri bukan angina", 4: "Tanpa gejala"}.get)
with col2:
    restecg = st.selectbox("EKG istirahat", [0, 1, 2],
                           format_func={0: "Normal", 1: "Kelainan gelombang ST-T",
                                        2: "Hipertrofi ventrikel kiri"}.get)
    thalach = st.number_input("Detak jantung maksimum", 60, 220, 150)
    exang = st.selectbox("Angina saat olahraga", [0, 1],
                         format_func=lambda v: "Ya" if v else "Tidak")
    oldpeak = st.number_input("Depresi ST (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox("Kemiringan segmen ST", [1, 2, 3], index=1,
                         format_func={1: "Naik", 2: "Datar", 3: "Turun"}.get)
    ca = st.selectbox("Pembuluh utama terlihat (0-3)", [0, 1, 2, 3, None],
                      format_func=lambda v: UNKNOWN if v is None else str(v))
    thal = st.selectbox("Hasil tes thallium", [3, 6, 7, None],
                        format_func=lambda v: {3: "Normal", 6: "Cacat tetap",
                                               7: "Cacat reversibel"}.get(v, UNKNOWN))

row = pd.DataFrame([{
    "age": age, "gender": gender, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope,
    "ca": float("nan") if ca is None else ca,
    "thal": float("nan") if thal is None else thal,
}])[meta["features"]]

if st.button("Prediksi", type="primary"):
    p = float(model.predict_proba(row)[0, 1])
    st.metric("Probabilitas penyakit jantung", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))
    if p >= meta["threshold"]:
        st.error("Risiko tinggi menurut model. Pemeriksaan lanjutan dengan tenaga medis dianjurkan.")
    else:
        st.success("Risiko rendah menurut model. Tetap konsultasikan ke tenaga medis bila ada keluhan.")
    st.caption(f"Batas keputusan model: {meta['threshold']:.1%} "
               "(dipilih agar recall minimal 90% pada data latih, sehingga banyak false positive).")
