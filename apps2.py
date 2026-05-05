import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Mengabaikan warning scikit-learn agar tampilan terminal rapi
warnings.filterwarnings('ignore')

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Pelanggan Mall", layout="wide")

# --- STYLE CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Mall_Customers.csv")
        df.columns = ['ID', 'Gender', 'Usia', 'Pendapatan_Tahunan', 'Skor_Pengeluaran']
        return df
    except:
        st.error("File 'Mall_Customers.csv' tidak ditemukan. Pastikan sudah diupload ke GitHub!")
        return None

df = load_data()

if df is not None:
    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Pengaturan Model")
    k_value = st.sidebar.slider("Pilih Jumlah Kelompok (Cluster):", 2, 10, 5)
    
    st.title("🛍️ Aplikasi Segmentasi & Prediksi Pelanggan Mall")
    st.info("Aplikasi ini mengelompokkan pelanggan berdasarkan pendapatan dan perilaku belanja, serta dapat memprediksi segmen pelanggan baru.")

    # --- PROSES CLUSTERING (TRAINING MODEL) ---
    X = df[['Pendapatan_Tahunan', 'Skor_Pengeluaran']]
    
    model_kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42)
    df['Cluster'] = model_kmeans.fit_predict(X)
    
    score = silhouette_score(X, df['Cluster'])

    # --- BAGIAN 1: METRIK UTAMA ---
    st.subheader("📊 Ringkasan Model")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Data Pelanggan", f"{len(df)} orang")
    with col2:
        st.metric("Jumlah Cluster (K)", k_value)
    with col3:
        st.metric("Skor Silhouette (Akurasi)", f"{score:.3f}")
        st.caption("ℹ️ Skor mendekati 1.0 berarti pengelompokan sangat baik.")

    st.divider()

    # --- BAGIAN 2: VISUALISASI & SPREAD ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📍 Visualisasi Sebaran Cluster")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df, x='Pendapatan_Tahunan', y='Skor_Pengeluaran', 
            hue='Cluster', palette='bright', s=100, ax=ax, alpha=0.7
        )
        # Gambar titik tengah (Centroids)
        plt.scatter(
            model_kmeans.cluster_centers_[:, 0], 
            model_kmeans.cluster_centers_[:, 1], 
            s=300, c='black', marker='X', label='Pusat Kelompok'
        )
        plt.title("Grafik Pendapatan vs Skor Pengeluaran")
        plt.xlabel("Pendapatan Tahunan (k$)")
        plt.ylabel("Skor Pengeluaran (1-100)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

    with col_right:
        st.subheader("📑 Tabel Sebaran (Spread)")
        st.write("Rata-rata karakteristik tiap kelompok:")
        spread = df.groupby('Cluster').agg({
            'Pendapatan_Tahunan': 'mean',
            'Skor_Pengeluaran': 'mean',
            'Usia': 'mean',
            'ID': 'count'
        }).rename(columns={'ID': 'Jumlah_Orang'}).round(1)
        
        st.dataframe(spread, use_container_width=True)

    st.divider()

    # --- BAGIAN 3: PREDIKSI PELANGGAN BARU ---
    st.subheader("🔮 Prediksi Segmen Pelanggan Baru")
    st.write("Masukkan perkiraan pendapatan dan pengeluaran pelanggan baru untuk melihat ia masuk ke target pasar yang mana.")
    
    # Membuat form inputan
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        input_pendapatan = st.number_input("Pendapatan Tahunan (k$)", min_value=0, max_value=200, value=50, step=1)
    with col_input2:
        input_pengeluaran = st.number_input("Skor Pengeluaran (1-100)", min_value=1, max_value=100, value=50, step=1)

    # Tombol Prediksi
    if st.button("Lakukan Prediksi", type="primary"):
        # Siapkan data baru untuk diprediksi
        data_baru = pd.DataFrame({
            'Pendapatan_Tahunan': [input_pendapatan],
            'Skor_Pengeluaran': [input_pengeluaran]
        })
        
        # Eksekusi prediksi menggunakan model K-Means yang sudah dilatih
        prediksi_cluster = model_kmeans.predict(data_baru)[0]
        
        # Tampilkan Hasil
        st.success(f"🎉 **Hasil:** Pelanggan ini masuk ke dalam **Cluster {prediksi_cluster}**")
        
        # Tarik data rata-rata dari tabel spread untuk menjelaskan cluster tersebut
        rata_pend = spread.loc[prediksi_cluster, 'Pendapatan_Tahunan']
        rata_peng = spread.loc[prediksi_cluster, 'Skor_Pengeluaran']
        
        st.info(f"💡 **Info Cluster {prediksi_cluster}:** Kelompok ini rata-rata memiliki Pendapatan **{rata_pend}k $** dan Skor Pengeluaran **{rata_peng}**.")

    st.divider()

    # --- BAGIAN 4: DATA DETAIL ---
    with st.expander("Klik untuk melihat detail data pelanggan"):
        pilihan_cluster = st.selectbox("Pilih Cluster:", range(k_value))
        data_filtered = df[df['Cluster'] == pilihan_cluster]
        st.dataframe(data_filtered, use_container_width=True)