import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client
import os

# ==========================================
# 1. KONFIGURASI KONEKSI
# ==========================================
# Gunakan URL lengkap (https://...) agar tidak terjadi Invalid URL
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." # Masukkan Key Lengkap Anda
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE"

# Inisialisasi Service
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gagal Inisialisasi Service: {e}")

# ==========================================
# 2. FUNGSI LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    try:
        # Mengambil dari tabel hasil pembersihan SQL
        res = supabase.table("bacakuy_sales_clean").select("*").execute()
        df = pd.DataFrame(res.data)
        return df
    except Exception as e:
        st.error(f"Gagal mengambil data dari Supabase: {e}")
        return pd.DataFrame()

# ==========================================
# 3. ANTARMUKA DASHBOARD (UI)
# ==========================================
st.set_page_config(page_title="Bacakuy Strategic Intelligence", layout="wide")

# Header Dashboard sesuai desain target
st.title("Strategic Intelligence Hub")
st.write("Menganalisis performa judul buku secara real-time.")

df_raw = load_data()

if not df_raw.empty:
    # --- BARIS 1: METRIK UTAMA (Metric Cards) ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("Market Valuation", f"Rp {df_raw['gross_sale'].sum():,.0f}", "+5.2%")
    with col_m2:
        st.metric("Circulation", f"{df_raw['units_sold'].sum():,.0f}", "Units Delivered")
    with col_m3:
        st.metric("Brand Loyalty", f"{df_raw['book_average_rating'].mean():.2f}/5", "Avg Sentiments")
    with col_m4:
        st.metric("Status", "Live Production", "Active")

    st.markdown("---")

    # --- BARIS 2: FILTER & PREDIKSI ---
    col_input, col_graph = st.columns([1, 2])

    with col_input:
        st.subheader("Segment Filter & Prediction")
        
        # Filter Genre Dinamis
        genres = df_raw['genre'].unique().tolist()
        selected_genre = st.selectbox("All Categories", ["Semua Genre"] + genres)
        
        # Data Filtered untuk Prediksi
        df_filtered = df_raw if selected_genre == "Semua Genre" else df_raw[df_raw['genre'] == selected_genre]
        
        st.write("---")
        # Input Fitur Prediksi
        in_units = st.number_input("Jumlah Unit Terjual (Units Sold)", value=100)
        in_rating = st.slider("Rating Rata-rata Buku", 0.0, 5.0, 4.5)
        
        if st.button("Generate Strategic Insight", use_container_width=True):
            # 1. Hitung Prediksi dengan Machine Learning
            try:
                X = df_filtered[['units_sold', 'book_average_rating']]
                y = df_filtered['gross_sale']
                model = LinearRegression().fit(X, y)
                prediction = model.predict([[in_units, in_rating]])[0]
                
                st.success(f"Estimasi Gross Sales: Rp {prediction:,.0f}")

                # 2. Panggil Gemini AI untuk Insight Syariah
                with st.spinner("AI sedang menganalisis strategi..."):
                    model_ai = genai.GenerativeModel('gemini-1.5-flash')
                    prompt = f"""
                    Anda adalah CDO Bacakuy. Berikan analisis untuk genre {selected_genre}:
                    - Target Penjualan: {in_units} unit
                    - Estimasi Profit: Rp {prediction:,.0f}
                    
                    Berikan 2 langkah strategis pemasaran dan 1 nasihat muamalah syariah tentang kejujuran.
                    """
                    response = model_ai.generate_content(prompt)
                    st.info(response.text)
            except:
                st.error("Data tidak cukup untuk melakukan prediksi pada segmen ini.")

    with col_graph:
        st.subheader("Sales Performance Trend")
        # Grafik Area sesuai desain target
        st.area_chart(df_filtered.set_index('genre')['gross_sale'] if 'genre' in df_filtered.columns else df_filtered['gross_sale'])
        
        st.subheader("Profitability Distribution")
        st.bar_chart(df_filtered['units_sold'])

else:
    st.warning("Data kosong atau tabel 'bacakuy_sales_clean' belum dibuat di Supabase.")
    st.info("Pastikan Anda sudah menjalankan query SQL untuk membersihkan data (koma ke titik).")
