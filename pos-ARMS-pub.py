import streamlit as st
import streamlit.components.v1 as components
import pickle
import numpy as np
from math import *
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from streamlit_option_menu import option_menu
from sqlalchemy import create_engine
import sqlalchemy as sa
import pyodbc
import pymysql
from imblearn.over_sampling import SMOTE
import plotly.express as px

data_multi = []

def load_model():
    with open('clf_imb_adj.pkl', 'rb') as file:
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

# Membuat koneksi ke database
engine = sa.create_engine('postgresql+psycopg2://st.secrets.postgresql.username:st.secrets.postgresql.password@st.secrets['postgresql']['server']:5432/st.secrets.postgresql.database')

def load_data():
    encoder = preprocessing.LabelEncoder()
    df = pd.read_sql_query('SELECT * FROM "arms_data"', engine.connect())
    df = pd.read_excel("dataset_1.4_7_MZ.xlsx")
    df["Jarak"] = df.apply(lambda row : haversine(row["Lng_Ktr"], row["Lat_Ktr"],row["Lng_Dms"], row["Lat_Dms"]), axis=1)
    df.Jenis_Kelamin = encoder.fit_transform(df.Jenis_Kelamin.fillna('0'))
    df = df[['Resign', 'Lama_Pada_Kantor_Terakhir', 'Lama_Pada_Regional_Terakhir', 'Jml_Pindah_Kantor', 
                 'Lama_Pada_Jabatan_Terakhir', 'Lama_Pada_Levjab_Terakhir', 'Usia_Saat_Resign', 'Masa_Bakti', 'Status_Menikah_2',
                 'Jenis_Kelamin', 'Jml_Anak', 'Pendidikan_2', 'Gaji_Bln_Terakhir', 'Jarak', 'Kecocokan_JobFam'
                ]]
    return df

df = load_data()
SMOTE=SMOTE()

def halaman_explore():
    st.image("header_nyolong.jpg")
    st.write("""## Explore Company Employee Attrition Data""")
    st.write("""### 1. Company Current Condition""")
    html_temp = """
    <center><script type='text/javascript' src='https://tabel.posindonesia.co.id/javascripts/api/viz_v1.js'></script>
    <div class='tableauPlaceholder' style='width: 800px; height: 1027px;'>
    <object class='tableauViz' width='800' height='1027' style='display:none;'>
    <param name='host_url' value='https%3A%2F%2Ftabel.posindonesia.co.id%2F' /> 
    <param name='embed_code_version' value='3' /> <param name='site_root' value='' />
    <param name='name' value='HCStrategyDashboard&#47;Viz-PITA' /><param name='tabs' value='no' />
    <param name='toolbar' value='yes' /><param name='showAppBanner' value='false' /></object></div></center>"""
    components.html(html_temp, height=1100)
    st.write("""### 2. Employee atrrition area chart based on age generation""")
    col1, col2 = st.columns([3,1])
    with col1:
        st.image("pensiun_by_generasi_2.png")
    with col2:
        st.markdown("""Berdasarkan chart disamping, kita menemukan bahwa Gen-X dan Boomers menyumbang angka cukup tinggi 
        terhadap total data karyawan yang memutuskan untuk resign. Meskipun jumlahnya banyak, tetapi masa bakti rata-rata 
        karyawan resign pada kedua generasi tersebut masihlah cukup tinggi, dibandingkan dengan generasi Millenials dan Gen-Z 
        yang memiliki rata-rata masa bakti yang lebih sebentar""")
    
    st.write("""### 3. Faktor-faktor yang berkorelasi terhadap resign""")
    plt_1 = plt.figure(figsize=(15,15))
    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    dataplot=sb.heatmap(df.corr(method='pearson'),vmin=-0.4, vmax=0.4, cmap="BrBG", annot=True, mask=mask)
    st.pyplot(plt_1)
    col1, col2 = st.columns([1,3])
    with col1:
            st.write("""### Deskripsi""")
            st.write("""Berdasarkan hasil pengujian dengan menggunakan metode statistik terhadap data karyawan yang diperoleh dari berbagai 
            sumber database legacy, berikut ini adalah heatmap korelasi faktor-faktor yang berdasarkan statistik memiliki hubungan terhadap karyawan 
            yang memutuskan untuk resign.""")
            
            st.write("""Interpretasi dari warna pada heatmap adalah apabila suatu faktor berwarna biru maka faktor tersebut
            memiliki korelasi yang positif atau searah terhadap keputusan resign atau dapat diartikan semakin tinggi nilai faktor maka semakin
            tinggi pula kemungkinan resign.""")
    
            st.write("""Sedangkan apabila suatu faktor berwarna coklat maka faktor tersebut
            memiliki korelasi yang negatif atau berbanding terbalik terhadap keputusan resign atau dapat diartikan semakin tinggi nilai faktor maka semakin
            rendah kemungkinan resign atau berlaku sebaliknya. Semakin pekat kedua warna maka menandakan semakin besar nilai korelasinya terhadap keputusan resign""")

            st.write("""Semakin pekat kedua warna maka menandakan semakin besar nilai korelasinya terhadap keputusan resign.""")

    with col2:
        plt_2 = plt.figure(figsize=(10, 14))
        heatmap = sb.heatmap(df.corr()[['Resign']].sort_values(by='Resign', ascending=False), vmin=-0.4, vmax=0.4, annot=True, cmap='BrBG')
        heatmap.set_title('Features Correlating with Resign', fontdict={'fontsize':18}, pad=16);
        st.pyplot(plt_2)

    st.write("""### 4. Model Klasifikasi dan Feature Importance""")
    kolom_data = ['Jarak', 'Pendidikan_2', 'Jenis_Kelamin', 'Lama_Pada_Jabatan_Terakhir',
              'Usia_Saat_Resign', 'Lama_Pada_Regional_Terakhir', 'Masa_Bakti',
              'Jml_Anak', 'Status_Menikah_2', 'Jml_Pindah_Kantor', 'Kecocokan_JobFam'
             ]
    X = df[kolom_data]
    y = df['Resign']

    X_SMOTE, y_SMOTE = SMOTE.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    X_train_SMOTE, X_test_SMOTE, y_train_SMOTE, y_test_SMOTE = train_test_split(X_SMOTE, y_SMOTE, test_size=0.4, random_state=123)
    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X_train_SMOTE, y_train_SMOTE)
    y_pred_SMOTE = clf.predict(X_test_SMOTE)

    acc = accuracy_score(y_test_SMOTE,  y_pred_SMOTE)*100
    st.write("""#### Pemilihan dan akurasi model klasifikasi""")
    st.write("""Model yang digunakan untuk klasifikasi adalah Random Forest Classifier dengan akurasi model""", acc, """%""")
    st.write("""#### Feature Importance""")
    st.write("""Selanjutnya dilakukan analisa terhadap faktor-faktor yang dianggap penting terhadap keputusan resign berdasarkan model klasifikasi, 
    jika sebelumnya kita sudah menganalisa menggunakan analisis korelasi untuk melihat hubungan antar variabel secara linear, kali ini akan dikakukan
    Feature Importance untuk melihat seberapa berpengaruh suatu faktor terhadap hasil prediksi keputusan resign seseorang.""")

    permutasi = permutation_importance(clf, X_test_SMOTE, y_test_SMOTE, n_repeats=10, random_state=0)
    perm = pd.DataFrame(columns=['AVG_Importance', 'STD_Importance'], index=[i for i in X_train_SMOTE.columns])
    perm['AVG_Importance'] = permutasi.importances_mean

    perm_sorted = perm.sort_values('AVG_Importance')
    # st.write(perm_sorted) 
    # plt_6 = plt.barh(perm_sorted.AVG_Importance, perm_sorted.index)
    # st.pyplot(plt_6)
    perm_sorted2=px.data.tips()
    fig=px.bar(perm_sorted,x=perm_sorted.AVG_Importance, y=perm_sorted.index , orientation='h', text=perm_sorted.AVG_Importance)
    st.write(fig)

    


def halaman_prediksi():
    st.image("header_nyolong.jpg")
    st.write("""## Single Employee Attrition Predictor Apps""")
    st.write("""#### Silahkan isi input data yang dibutuhkan""")

    col1, col2 = st.columns(2)
    with col1:
        nippos = st.number_input("Masukkan Nippos", 0)
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
        jarak = st.number_input("Masukkan jarak domisili terhadap kantor (Kilometer)", 0)
        jml_pindah = st.number_input("Masukkan Jumlah Pindah Kantor", 0)
        lama_pada_reg = st.number_input("Masukkan Lama Pada Regional Terakhir", 0)
        lama_pada_jab = st.number_input("Masukkan Lama Pada Jabatan Terakhir", 0)
        jobfam_choices = {0: "Tidak Sesuai", 1: "Sesuai"}
        def format_func_job(option):
            return jobfam_choices[option]
        jobfam = st.selectbox("Kecocokan Jobfam dengan latar belakang pendidikan", options=list(jobfam_choices.keys()), format_func=format_func_job)
    predict = st.button("Prediksi")

    if predict:
        sX = np.array([[jns_kelamin, usia, stat_menikah, jml_anak, pendidikan, masa_bakti, jarak, jml_pindah, lama_pada_reg, lama_pada_jab, jobfam]])
        hasil_prediksi  = model_pred.predict(sX)
        probabilitas = model_pred.predict_proba(sX)*100
        if hasil_prediksi==0:
            st.subheader("Karyawan dengan kriteria di atas diperkirakan Tidak Resign")
            st.write("""###### Dengan Probabilitas Prediksi (%)""")
            st.write(probabilitas.astype(int))
            st.caption("""0 = Tidak Resign | 1 = Resign""")
        else:
            st.subheader("Karyawan dengan kriteria di atas diperkirakan Resign")
            st.write("""###### Dengan Probabilitas Prediksi (%)""")
            st.write(probabilitas.astype(int))
            st.caption("""0 = Tidak Resign | 1 = Resign""")

def halaman_multi():
    st.image("header_nyolong.jpg")
    st.write("""## Multiple Employee Attrition Predictor Apps""")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.write("""###### Untuk prediksi lebih dari satu pegawai silahkan isikan data sesuai pada template yang dapat diunduh dibawah.""")

    @st.cache
    def convert_csv(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    with open("pred_multi.xlsx", 'rb') as my_file:
        st.download_button(
            label = 'Download Template Pengisian', 
            data = my_file, 
            file_name = 'template_isian.xlsx', 
            mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')  

    st.write("""####""")
    st.write("""#### Silahkan upload input data yang dibutuhkan""")
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
            mX = data_multi[["Jenis_Kelamin", "Usia", "Stat_Menikah", "Jml_Anak", "Pendidikan", "Masa_Bakti", "Jarak", "Jml_Pindah", "Lama_Pada_Reg", "Lama_Pada_Jab", "Kecocokan_JobFam_dan_LatarPendidikan"]]
            mhasil_prediksi  = model_pred.predict(mX)
            mprobabilitas = model_pred.predict_proba(mX)*100
            
            print(mhasil_prediksi)
            mX['Prediksi'] = mhasil_prediksi.tolist()
            mX['Probabilitas[0,1]'] = mprobabilitas.tolist()
            st.write("""##### Hasil Prediksi""")
            st.write(mX)

            csv = convert_csv(mX)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='Prediction.csv',
            mime='text/csv',
            )

def halaman_utama():
    global data_multi
    st.set_page_config(page_title="Pos ARMS", page_icon="ui-checks-grid", layout="wide")
    st.sidebar.image("images.png", width=150)
    st.sidebar.write("""##### Atrrition and Retention Management System""")
    
    with st.sidebar:
        page = option_menu(
            menu_title = "Choose One",
            options = ["Explore", "Single Predictor", "Multi Predictor"],
            icons = ["zoom-in", "person-fill", "people-fill"],
            menu_icon = "ui-checks-grid",
            default_index = 0,
            )

    if page == "Explore":
        halaman_explore()
    if page == "Single Predictor":
        halaman_prediksi()
    if page == "Multi Predictor":
        global data_multi
        halaman_multi()

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets.password.password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password salah")
        return False
    else:
        # Password correct.
        return True

if check_password():
    st.write("Selamat datang di aplikasi Pos A.R.M.S...")
    gas = st.button("Click me")

if gas:
    halaman_utama()

