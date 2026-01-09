import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
import os
from supabase import create_client, Client

# ==========================================
# 1. KONFIGURASI & SETUP (SUPABASE & GEMINI)
# ==========================================
st.set_page_config(page_title="Bacakuy Strategic AI", layout="wide")

# Credential (Gunakan st.secrets atau environment variable)
# Ganti dengan nilai asli Anda atau set di Environment Variables
URL_SUPABASE = os.environ.get("SUPABASE_URL") or "URL_SUPABASE_ANDA"
KEY_SUPABASE = os.environ.get("SUPABASE_KEY") or "KEY_SUPABASE_ANDA"
API_KEY_GEMINI = os.environ.get("GEMINI_API_KEY") or "API_KEY_GEMINI_ANDA"

# Inisialisasi Klien
try:
    supabase: Client = create_client(URL_SUPABASE, KEY_SUPABASE)
    genai.configure(api_key=API_KEY_GEMINI)
except Exception as e:
    st.error(f"Gagal Inisialisasi Service: {e}")

# ==========================================
# 2. FUNGSI DATA & AI
# ==========================================

@st.cache_data
def fetch_and_train():
    # 1. Ambil data dari Supabase (Tabel: sales_data)
    response = supabase.table("sales_data").select("*").execute()
    df = pd.DataFrame(response.data)
    
    # 2. Data Cleaning
    df = df.dropna(subset=['units_sold', 'book_average_rating', 'gross_sales'])
    
    # 3. Training Model ML
    X = df[['units_sold', 'book_average_rating']]
    y = df['gross_sales']
    model = LinearRegression().fit(X, y)
    
    return model, df

def get_gemini_insight(summary, prediction):
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Anda adalah Konsultan Bisnis Syariah untuk 'Bacakuy'.
    
    Konteks Data Perusahaan:
    - Total Koleksi: {summary['total_books']} judul
    - Rata-rata Rating: {summary['avg_rating']:.2f}
    - Total Pendapatan: Rp {summary['total_sales']:,.0f}
    
    Hasil Prediksi Input Baru:
    - Estimasi Penjualan: Rp {prediction:,.0f}
    
    Tugas:
    1. Berikan analisis singkat apakah hasil prediksi ini potensial dibandingkan rata-rata perusahaan.
    2. Berikan 3 strategi pemasaran berbasis nilai-nilai Islam (Kejujuran, Transparansi, Sedekah).
    3. Tambahkan satu kutipan singkat (Ayat/Hadits) yang relevan.
    """
    response = model_ai.generate_content(prompt)
    return response.text

# Load Data Awal
try:
    model, df_clean = fetch_and_train()
except:
    st.warning("Gagal memuat data dari Supabase. Menggunakan data dummy untuk demonstrasi.")
    # Fallback Data Dummy jika Supabase belum di-setup
    df_clean = pd.DataFrame({
        'units_sold': [100, 200, 150, 300, 50],
        'book_average_rating': [4.5, 4.0, 4.8, 3.9, 4.2],
        'gross_sales': [5000000, 9500000, 8000000, 14000000, 2000000]
    })
    X = df_clean[['units_sold', 'book_average_rating']]
    y = df_clean['gross_sales']
    model = LinearRegression().fit(X, y)

# ==========================================
# 3. ANTARMUKA DASHBOARD (UI)
# ==========================================

st.title("📚 Bacakuy Strategic Intelligence Hub")
st.write(f"Menganalisis performa {len(df_clean)} judul buku secara real-time.")

# --- BARIS 1: METRIK UTAMA ---
m1, m2, m3, m4 = st.columns(4)
total_sales = df_clean['gross_sales'].sum()
avg_rate = df_clean['book_average_rating'].mean()

m1.metric("Market Valuation", f"Rp {total_sales:,.0f}", "+5.2%")
m2.metric("Units Delivered", f"{df_clean['units_sold'].sum():,.0f}", "Books")
m3.metric("Avg Sentiments", f"{avg_rate:.2f}/5")
m4.metric("Status", "Live Production", "Active")

st.markdown("---")

# --- BARIS 2: ANALISIS & INPUT ---
col_input, col_graph = st.columns([1, 2])

with col_input:
    st.subheader("Segment Filter & Prediction")
    category = st.selectbox("Category", ["All Categories", "Religion", "Business", "Fiction"])
    
    st.write("---")
    # Input Prediksi
    in_units = st.number_input("Input Units Sold", min_value=0, value=100)
    in_rating = st.slider("Input Rating", 0.0, 5.0, 4.5)
    
    if st.button("Generate Strategic Insight", use_container_width=True):
        # Hitung Prediksi
        pred_result = model.predict([[in_units, in_rating]])[0]
        
        # Buat Ringkasan Data
        summary_data = {
            'total_books': len(df_clean),
            'avg_rating': avg_rate,
            'total_sales': total_sales
        }
        
        # Panggil AI
        with st.spinner("Gemini AI sedang berpikir..."):
            ai_insight = get_gemini_insight(summary_data, pred_result)
            
            st.markdown("### 🤖 AI Analysis")
            st.success(f"**Estimasi Pendapatan: Rp {pred_result:,.0f}**")
            st.info(ai_insight)

with col_graph:
    st.subheader("Sales Performance Trend")
    # Grafik Area untuk Visualisasi Penjualan
    st.area_chart(df_clean['gross_sales'])
    
    st.subheader("Distribution: Units vs Rating")
    # Grafik Bar untuk Unit
    st.bar_chart(df_clean['units_sold'])

# --- BARIS 3: DATA EXPLORER ---
with st.expander("Lihat Detail Data Supabase"):
    st.dataframe(df_clean, use_container_width=True)
