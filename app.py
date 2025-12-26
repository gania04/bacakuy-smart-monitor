import os
import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
from datetime import datetime

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Analisis Penjualan Buku",
    page_icon="📚",
    layout="wide"
)

# --- 2. KONEKSI SUPABASE (Environment Variables) ---
# Pastikan Anda sudah mengatur environment variable di sistem atau file .env
@st.cache_resource
def init_connection():
    try:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            st.error("Error: SUPABASE_URL dan SUPABASE_KEY tidak ditemukan di environment variables.")
            st.stop()
        return create_client(url, key)
    except Exception as e:
        st.error(f"Gagal menghubungkan ke Supabase: {e}")
        st.stop()

supabase = init_connection()

# --- 3. FUNGSI LOAD DATA ---
@st.cache_data(ttl=600)  # Cache data selama 10 menit
def fetch_data():
    try:
        # Mengambil data dari tabel 'penjualan_buku'
        response = supabase.table("penjualan_buku").select("*").execute()
        df = pd.DataFrame(response.data)
        
        # Preprocessing: Pastikan format tanggal benar
        if not df.empty:
            df['publish_date'] = pd.to_datetime(df['publish_date'])
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data dari database: {e}")
        return pd.DataFrame()

# Load data awal
raw_data = fetch_data()

if raw_data.empty:
    st.warning("Data tidak ditemukan atau tabel kosong.")
    st.stop()

# --- 4. SIDEBAR FILTER ---
st.sidebar.header("Filter Data")

# Filter Genre
all_genres = sorted(raw_data['genre'].unique())
selected_genre = st.sidebar.multiselect("Pilih Genre:", options=all_genres, default=all_genres)

# Filter Author Rating
min_rating = float(raw_data['author_rating'].min())
max_rating = float(raw_data['author_rating'].max())
selected_rating = st.sidebar.slider(
    "Rentang Author Rating:",
    min_value=min_rating,
    max_value=max_rating,
    value=(min_rating, max_rating)
)

# Apply Filter
df_filtered = raw_data[
    (raw_data['genre'].isin(selected_genre)) &
    (raw_data['author_rating'] >= selected_rating[0]) &
    (raw_data['author_rating'] <= selected_rating[1])
]

# --- 5. TAMPILAN UTAMA ---
st.title("📚 Dashboard Penjualan Buku")
st.markdown("Analisis performa penjualan berdasarkan data Supabase.")

if df_filtered.empty:
    st.info("Tidak ada data yang sesuai dengan filter yang dipilih.")
else:
    # --- 6. KPI SECTION ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_rev = df_filtered['publisher_revenue'].sum()
        st.metric("Total Publisher Revenue", f"${total_rev:,.2f}")
        
    with col2:
        total_units = df_filtered['units_sold'].sum()
        st.metric("Total Units Sold", f"{total_units:,}")
        
    with col3:
        avg_rating = df_filtered['book_average_rating'].mean()
        st.metric("Avg Book Rating", f"{avg_rating:.2f} / 5.0")

    st.divider()

    # --- 7. GRAFIK TREN BULANAN (Line Chart) ---
    st.subheader("Tren Gross Sales Bulanan")
    try:
        # Resampling data ke bulanan
        df_trend = df_filtered.copy()
        df_trend.set_index('publish_date', inplace=True)
        # 'gross_sales' diasumsikan ada, jika tidak, kita gunakan publisher_revenue sebagai proksi
        # atau pastikan kolom gross_sales tersedia di DB
        monthly_sales = df_trend.resample('M')['publisher_revenue'].sum().reset_index()
        
        fig_trend = px.line(
            monthly_sales, 
            x='publish_date', 
            y='publisher_revenue',
            labels={'publisher_revenue': 'Revenue', 'publish_date': 'Bulan'},
            template="plotly_white",
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal memproses grafik tren: {e}")

    # --- 8. BAR CHART HORIZONTAL (Units Sold by Genre) ---
    st.subheader("Komposisi Unit Terjual per Genre")
    try:
        genre_dist = df_filtered.groupby('genre')['units_sold'].sum().sort_values(ascending=True).reset_index()
        
        fig_genre = px.bar(
            genre_dist,
            x='units_sold',
            y='genre',
            orientation='h',
            color='units_sold',
            color_continuous_scale='Blues',
            labels={'units_sold': 'Total Unit Terjual', 'genre': 'Genre Buku'},
            template="plotly_dark"
        )
        st.plotly_chart(fig_genre, use_container_width=True)
    except Exception as e:
        st.error(f"Gagal memproses grafik genre: {e}")

    # --- 9. DATA TABLE (Opsional) ---
    with st.expander("Lihat Detail Data Mentah"):
        st.dataframe(df_filtered, use_container_width=True)

# Footer
st.caption(f"Terakhir diperbarui: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
