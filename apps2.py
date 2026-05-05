import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Analisis Pelanggan Mall", layout="wide")

# --- STYLE CSS (Agar tampilan lebih rapi) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);        color: #000000;
    }
    .stMetric * {
        color: #000000 !important;    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    # Pastikan file Mall_Customers.csv sudah kamu upload ke GitHub
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
    
    st.title("🛍️ Aplikasi Segmentasi Pelanggan Mall")
    st.info("Aplikasi ini mengelompokkan pelanggan berdasarkan pendapatan dan perilaku belanja mereka.")

    # --- PROSES CLUSTERING ---
    X = df[['Pendapatan_Tahunan', 'Skor_Pengeluaran']]
    
    model_kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42)
    df['Cluster'] = model_kmeans.fit_predict(X)
    
    # Perhitungan "Akurasi" (Silhouette Score)
    score = silhouette_score(X, df['Cluster'])

    # --- BAGIAN 1: METRIK UTAMA ---
    st.subheader("📊 Ringkasan Model")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Data Pelanggan", f"{len(df)} orang")
    with col2:
        st.metric("Jumlah Cluster (K)", k_value)
    with col3:
        # Silhouette score: Makin mendekati 1 makin bagus pemisahannya
        st.metric("Skor Silhouette (Akurasi)", f"{score:.3f}")
        st.caption("ℹ️ Skor mendekati 1.0 berarti pengelompokan sangat baik.")

    st.divider()

    # --- BAGIAN 2: VISUALISASI ---
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
        st.write("Rata-rata tiap kelompok:")
        # Menghitung profil tiap cluster
        spread = df.groupby('Cluster').agg({
            'Pendapatan_Tahunan': 'mean',
            'Skor_Pengeluaran': 'mean',
            'Usia': 'mean',
            'ID': 'count'
        }).rename(columns={'ID': 'Jumlah_Orang'}).round(1)
        
        st.dataframe(spread, use_container_width=True)

    # --- BAGIAN 3: ANALISIS DATA ---
    st.subheader("🔍 Data Detail per Cluster")
    with st.expander("Klik untuk melihat daftar pelanggan berdasarkan cluster"):
        pilihan_cluster = st.selectbox("Pilih Cluster untuk dilihat detailnya:", range(k_value))
        data_filtered = df[df['Cluster'] == pilihan_cluster]
        st.dataframe(data_filtered, use_container_width=True)

    st.success("✅ Aplikasi Berhasil Dijalankan. Silakan screenshot halaman ini untuk laporan Anda.")