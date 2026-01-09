import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client
import os

# ==========================================
# 1. KONFIGURASI KONEKSI (PERBAIKAN URL)
# ==========================================
# Perbaikan: Menambahkan protokol https dan domain supabase
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9mdHB1bHNxeGpoaHRmdWttbXRyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU1NzAwNjksImV4cCI6MjA4MTE0NjA2OX0.aDLgRF2mzaJEW43h2hmZOBadEnDtUoRTZCueJHdfh04"
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE"

# Inisialisasi Service
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gagal Inisialisasi: {e}")

# ==========================================
# 2. FUNGSI LOAD DATA & PREDIKSI
# ==========================================
@st.cache_data
def load_data():
    try:
        # Mengambil data dari tabel 'sales_data'
        res = supabase.table("sales_data").select("*").execute()
        df = pd.DataFrame(res.data)
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data dari Supabase: {e}")
        return pd.DataFrame()

df_raw = load_data()

# ==========================================
# 3. INTERFACE DASHBOARD (UI)
# ==========================================
st.set_page_config(page_title="Bacakuy Strategic AI", layout="wide")
st.title("📚 Bacakuy Strategic Intelligence Hub")

if not df_raw.empty:
    # Sidebar Filter
    st.sidebar.header("🔍 Filter & Prediksi")
    
    # Filter Genre
    list_genre = df_raw['genre'].unique().tolist() if 'genre' in df_raw.columns else ["Umum"]
    selected_genre = st.sidebar.selectbox("Pilih Genre Buku", ["Semua Genre"] + list_genre)
    
    # Input Target Penjualan
    input_units = st.sidebar.number_input("Target Produk Terjual", min_value=1, value=100)
    input_rating = st.sidebar.slider("Target Rating Buku", 1.0, 5.0, 4.5)

    # Filter Data untuk Grafik & Model
    if selected_genre != "Semua Genre":
        df_filtered = df_raw[df_raw['genre'] == selected_genre]
    else:
        df_filtered = df_raw

    # Baris 1: Metrik Utama
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Data", f"{len(df_filtered)} Judul")
    m2.metric("Rata-rata Rating", f"{df_filtered['book_average_rating'].mean():.2f}")
    m3.metric("Genre Aktif", selected_genre)

    # Baris 2: Grafik & AI Insight
    col_graph, col_ai = st.columns([2, 1])

    with col_graph:
        st.subheader(f"Tren Gross Sales: {selected_genre}")
        st.area_chart(df_filtered[['gross_sales']])
        
    with col_ai:
        if st.button("Hitung Estimasi & Insight AI", use_container_width=True):
            try:
                # 1. Prediksi ML
                X = df_filtered[['units_sold', 'book_average_rating']]
                y = df_filtered['gross_sales']
                model = LinearRegression().fit(X, y)
                prediction = model.predict([[input_units, input_rating]])[0]

                st.success(f"**Estimasi Gross Profit:**\nRp {prediction:,.0f}")

                # 2. AI Insight
                with st.spinner("AI sedang menganalisis..."):
                    model_ai = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"""
                    Analisis Performa Bisnis Bacakuy:
                    - Genre: {selected_genre}
                    - Target Terjual: {input_units} buku
                    - Prediksi Profit: Rp {prediction:,.0f}
                    
                    Tugas: Berikan 2 langkah strategis untuk mencapai target ini dan 
                    berikan 1 nasihat muamalah (syariah) tentang kejujuran dalam berdagang.
                    """
                    response = model_ai.generate_content(prompt)
                    st.info(response.text)
            except Exception as e:
                st.error("Data tidak cukup untuk membuat prediksi pada genre ini.")

else:
    st.warning("Data kosong. Pastikan tabel 'sales_data' di Supabase sudah terisi.")
