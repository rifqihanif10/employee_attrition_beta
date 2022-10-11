import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('modellogreg.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data  = load_model()
logreg = data["model"]


def halaman_prediksi():
    st.title("Employee Attrition Predictor Apps")
    st.write("""### Silahkan isi input data yang dibutuhkan""")

    lama_pd_kantor = st.number_input("Masukkan Lama Pada Kantor Saat Ini", 0)
    lama_pd_reg = st.number_input("Masukkan Lama Pada Regional Saat Ini", 0)
    gaji = st.number_input("Masukkan BSU Gaji Saat Ini", 0)
    usia = st.number_input("Masukkan Usia Saat Ini", 0)
    masa_bakti = st.number_input("Masukkan Masa Bekerja", 0)
    predict = st.button("Prediksi")

    if predict:
        X = np.array([[lama_pd_kantor, lama_pd_reg, gaji, usia, lama_pd_reg, masa_bakti]])
        hasil_prediksi  = logreg.predict(X)
        
        if hasil_prediksi==0:
            st.subheader("Karyawan dengan kriteria di atas diperkirakan Tidak Resign")
        else:
            st.subheader("Karyawan dengan kriteria di atas diperkirakan Resign")

halaman_prediksi()