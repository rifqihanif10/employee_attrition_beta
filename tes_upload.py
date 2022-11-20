import streamlit as st
import pickle
import numpy as np
from math import *
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

data_multi = []

def load_model():
    with open('clf_ori.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data  = load_model()
model_pred = data["model"]


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def load_data():
    encoder = preprocessing.LabelEncoder()
    df = pd.read_excel("dataset_1.4_2_MZ.xlsx")
    df["Jarak"] = df.apply(lambda row : haversine(row["Lng_Ktr"], row["Lat_Ktr"],row["Lng_Dms"], row["Lat_Dms"]), axis=1)
    df.Jenis_Kelamin = encoder.fit_transform(df.Jenis_Kelamin.fillna('0'))
    df = df[['Resign', 'Lama_Pada_Kantor_Terakhir', 'Lama_Pada_Regional_Terakhir', 'Jml_Pindah_Kantor', 
                 'Lama_Pada_Jabatan_Terakhir', 'Lama_Pada_Levjab_Terakhir', 'Usia_Saat_Resign', 'Masa_Bakti', 'Status_Menikah_2',
                 'Jenis_Kelamin', 'Jml_Anak', 'Pendidikan_2', 'Gaji_Bln_Terakhir', 'Jarak'
                ]]
    return df

df = load_data()

def halaman_explore():
    st.image("header_nyolong.jpg")
    st.write("""## Explore Company Employee Attrition Data""")
    st.write("""### Employee atrrition area chart based on age generation""")
    st.image("pensiun_by_generasi.png")
    st.write("""Berdasarkan chart diatas, kita menemukan bahwa Gen-X dan Boomers menyumbang angka cukup tinggi 
    terhadap total data karyawan yang memutuskan untuk resign. Meskipun jumlahnya banyak, tetapi masa bakti rata-rata 
    karyawan resign pada kedua generasi tersebut masihlah cukup tinggi, dibandingkan dengan generasi Millenials dan Gen-Z 
    yang memiliki rata-rata masa bakti yang lebih sebentar""")
    st.write(""" """)
    st.write("""## Faktor-faktor yang berkorelasi terhadap resign""")
    plt_1 = plt.figure(figsize=(15,15))
    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    dataplot=sb.heatmap(df.corr(method='pearson'),vmin=-0.4, vmax=0.4, cmap="BrBG", annot=True, mask=mask)
    st.pyplot(plt_1)
       
    plt_2 = plt.figure(figsize=(10, 14))
    heatmap = sb.heatmap(df.corr()[['Resign']].sort_values(by='Resign', ascending=False), vmin=-0.4, vmax=0.4, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with Resign', fontdict={'fontsize':18}, pad=16);
    st.pyplot(plt_2)

def halaman_prediksi():
    st.image("header_nyolong.jpg")
    st.write("""## Single Employee Attrition Predictor Apps""")
    st.write("""#### Silahkan isi input data yang dibutuhkan""")

    col1, col2 = st.columns(2)
    with col1:
        jns_kelamin_choices = {0: "Laki-laki", 1: "Perempuan"}
        def format_func_jns(option):
            return jns_kelamin_choices[option]

        jns_kelamin = st.selectbox("Masukkan Jenis Kelamin", options=list(jns_kelamin_choices.keys()), format_func=format_func_jns)
        usia = st.number_input("Masukkan Usia Saat Ini", 0)
    
        stat_menikah_choices = {1: "Blm Menikah", 2: "Cerai", 3: "Menikah"}
        def format_func_mrt(option):
            return stat_menikah_choices[option]
        stat_menikah = st.selectbox("Status Pernikahan", options=list(stat_menikah_choices.keys()), format_func=format_func_mrt)

        jml_anak = st.number_input("Masukkan Jml Anak (Jika Memiliki)", 0)
        pendidikan_choices = {0: "SD", 1: "SMP", 2: "SMA", 3: "D-1", 4: "D-2", 5: "D-3", 6: "D-4", 7: "S-1", 8: "S-2", 9: "S-3"}
        def format_func_edu(option):
            return pendidikan_choices[option]
        pendidikan = st.selectbox("Pendidikan", options=list(pendidikan_choices.keys()), format_func=format_func_edu)
    with col2:        
        masa_bakti = st.number_input("Masukkan Masa Bekerja", 0)
        jarak = st.number_input("Masukkan Jarak", 0)
        jml_pindah = st.number_input("Masukkan Jumlah Pindah Kantor", 0)
        lama_pada_reg = st.number_input("Masukkan Lama Pada Regional Terakhir", 0)
        lama_pada_jab = st.number_input("Masukkan Lama Pada Jabatan Terakhir", 0)
    predict = st.button("Prediksi")

    if predict:
        X = np.array([[jns_kelamin, usia, stat_menikah, jml_anak, pendidikan, masa_bakti, jarak, jml_pindah, lama_pada_reg, lama_pada_jab]])
        hasil_prediksi  = model_pred.predict(X)
        
        if hasil_prediksi==0:
            st.subheader("Karyawan dengan kriteria di atas diperkirakan Tidak Resign")
        else:
            st.subheader("Karyawan dengan kriteria di atas diperkirakan Resign")

def halaman_multi():
    st.image("header_nyolong.jpg")
    st.write("""## Multiple Employee Attrition Predictor Apps""")
    st.write("""#### Silahkan upload input data yang dibutuhkan""")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_files = st.file_uploader("Pilih file CSV atau XLSX", type=['csv', 'xlsx'])
    global data_multi
    if uploaded_files is not None:
        print(uploaded_files)
        try:
            data_multi = pd.read_csv(uploaded_files)
            data_multi = data_multi.astype(str)
            st.write(data_multi)
            predict2 = st.button("Prediksi")
        except Exception as e:
            print(e)
            data_multi = pd.read_excel(uploaded_files)
            data_multi = data_multi.astype(str)
            st.write(data_multi)
            predict2 = st.button("Prediksi")
    
        if predict2:
            mX = data_multi[["Jenis_Kelamin", "Usia", "Stat_Menikah", "Jml_Anak", "Pendidikan", "Masa_Bakti", "Jarak", "Jml_Pindah", "Lama_Pada_Reg", "Lama_Pada_Jab"]]
            mhasil_prediksi  = model_pred.predict(mX)
            # mhasil_prediksi.info()
            print(mhasil_prediksi)
            mX['Prediksi'] = mhasil_prediksi.tolist()
            # for i in mX:
            #     mhasil_prediksi  = model_pred.predict(i)
            #     print(i)
            #     print(mhasil_prediksi)
            #     # mX = mX.append({'Prediksi': mhasil_prediksi}, ignore_index=True)
            st.write("""##### Hasil Prediksi""")
            st.write(mX)

            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')

            csv = convert_df(mX)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Prediction.csv',
            mime='text/csv',
            )


def halaman_utama():
    global data_multi
    st.sidebar.image("images.png", width=150)
    st.sidebar.write("""##### Atrrition Prediction Apps""")
    page = st.sidebar.selectbox("Silahkan Pilih Satu", ("Explore", "Single Predictor", "Multi Predictor"))

    if page == "Explore":
        halaman_explore()
    if page == "Single Predictor":
        halaman_prediksi()
    if page == "Multi Predictor":
        global data_multi
        halaman_multi()

halaman_utama()