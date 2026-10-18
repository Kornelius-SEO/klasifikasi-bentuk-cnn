"""Streamlit Community Cloud entrypoint for the trained shape CNN."""
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from project import CNN, LABELS, preprocess_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.npz"

st.set_page_config(page_title="Klasifikasi Bentuk CNN", page_icon="🔷", layout="centered")

@st.cache_resource
def load_model():
    return CNN.load(MODEL_PATH)

st.title("Klasifikasi Bentuk dengan CNN")
st.write("Unggah gambar berisi satu lingkaran, persegi, atau segitiga. Model dilatih pada gambar bentuk sintetis sederhana.")

uploaded = st.file_uploader("Pilih gambar PNG atau JPEG", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    try:
        data = uploaded.getvalue()
        if len(data) > 5_000_000:
            st.error("Ukuran gambar maksimal 5 MB.")
        else:
            image = Image.open(BytesIO(data))
            image.verify()
            image = Image.open(BytesIO(data))
            st.image(image, caption="Gambar yang diunggah", width=280)
            if st.button("Prediksi", type="primary"):
                probabilities = load_model().predict(preprocess_image(image))[0]
                winner = int(probabilities.argmax())
                st.success(f"Prediksi: {LABELS[winner]} ({probabilities[winner] * 100:.1f}%)")
                st.write("Probabilitas setiap kelas:")
                for label, probability in zip(LABELS, probabilities):
                    st.write(f"{label.capitalize()}: {probability * 100:.1f}%")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Gambar tidak dapat diproses: {exc}")

st.caption("Model ini untuk demonstrasi tugas. Hasil pada foto nyata atau latar rumit dapat keliru.")
