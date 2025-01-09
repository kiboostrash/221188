import streamlit as st
from app.utils.helpers import load_model, predict

# Judul aplikasi
st.title("Aplikasi Prediksi dengan Streamlit")

# Input dari pengguna
user_input = st.text_input("Masukkan teks:")

# Prediksi
if st.button("Prediksi"):
    model = load_model("models/model.pkl")
    result = predict(model, user_input)
    st.write(f"Hasil Prediksi: {result}")
