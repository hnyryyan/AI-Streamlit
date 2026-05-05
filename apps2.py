import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# 1. Load Data
@st.cache_data
def load_data():
    # Menggunakan link raw dataset Mall Customer
    url = "https://raw.githubusercontent.com/stephi-ng/Mall-Customer-Segmentation-Analysis/master/Mall_Customers.csv"
    df = pd.read_csv(url)
    # Merubah nama kolom agar lebih mudah diakses
    df.columns = ['CustomerID', 'Gender', 'Age', 'Annual_Income', 'Spending_Score']
    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["Dataset Explorer", "Customer Segmentation"])

if page == "Dataset Explorer":
    st.title("🛍️ Mall Customer Dataset Explorer")
    st.write("Dataset ini digunakan untuk memahami segmentasi pelanggan di sebuah Mall.")
    
    st.dataframe(df, use_container_width=True)
    
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

    # Visualisasi Distribusi
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Distribusi Usia")
        fig, ax = plt.subplots()
        sns.histplot(df['Age'], kde=True, ax=ax, color='skyblue')
        st.pyplot(fig)
    with col2:
        st.write("### Gender")
        fig, ax = plt.subplots()
        df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

elif page == "Customer Segmentation":
    st.title("🎯 Customer Segmentation (K-Means Clustering)")
    
    # Memilih fitur untuk clustering
    X = df[['Annual_Income', 'Spending_Score']]
    
    # Input jumlah cluster
    st.sidebar.subheader("Pengaturan Model")
    clusters = st.sidebar.slider("Pilih Jumlah Cluster (K):", 2, 10, 5)
    
    # Jalankan K-Means
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    
    # Visualisasi Cluster
    st.write(f"### Visualisasi {clusters} Segmen Pelanggan")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='Cluster', 
                    palette='viridis', s=100, ax=ax)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=300, c='red', marker='X', label='Centroids')
    plt.title("Annual Income vs Spending Score")
    plt.legend()
    st.pyplot(fig)
    
    st.success("Analisis Selesai! Anda bisa melihat pengelompokan pelanggan di atas.")
    
    # Penjelasan Segmen (Jika K=5)
    if clusters == 5:
        st.info("""
        **Interpretasi Umum (K=5):**
        - **Cluster 0:** Pendapatan Rendah, Pengeluaran Rendah.
        - **Cluster 1:** Pendapatan Tinggi, Pengeluaran Rendah (Pelanggan Hemat).
        - **Cluster 2:** Pendapatan Menengah, Pengeluaran Menengah.
        - **Cluster 3:** Pendapatan Rendah, Pengeluaran Tinggi (Pelanggan Boros).
        - **Cluster 4:** Pendapatan Tinggi, Pengeluaran Tinggi (Target Utama/Sultan).
        """)