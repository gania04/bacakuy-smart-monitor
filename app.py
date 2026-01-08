import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
import os

# ==========================================
# 1. KONFIGURASI & SETUP AWAL
# ==========================================
st.set_page_config(page_title="Bacakuy Sales AI", layout="wide")

# --- KONFIGURASI GENAI (GEMINI) ---
# PENTING: Ganti string di bawah ini dengan API KEY asli Anda jika belum diset di Environment Variable
# Tips Senior Dev: Jangan pernah hardcode API Key di produksi. Gunakan st.secrets atau os.environ.
# Sesuai request, kita mencoba mengambil dari environment, jika tidak ada, harap masukkan manual.
api_key = os.environ.get("GEMINI_API_KEY") 

# Jika environment variable kosong, gunakan placeholder (Mahasiswa harus mengganti ini)
if not api_key:
    # Masukkan API Key Google Gemini Anda di sini
    api_key = "MASUKKAN_API_KEY_GEMINI_ANDA_DISINI" 

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Konfigurasi API Key Gagal: {e}")

# ==========================================
# 2. LOAD & CLEAN DATA (MODUL 5 CORE)
# ==========================================
@st.cache_data # Decorator agar data tidak di-load ulang setiap kali ada interaksi
def load_and_train_model():
    """
    Fungsi ini memuat data, membersihkan data, dan melatih model Linear Regression.
    """
    # --- A. LOAD DATA ---
    # Karena saya tidak memiliki file CSV fisik Anda, saya membuat Dummy Data 'bacakuy-sales'
    # agar aplikasi ini bisa langsung jalan (RUNNABLE).
    # Jika Anda punya file csv, uncomment baris: df = pd.read_csv('bacakuy-sales.csv')
    
    data_dummy = {
        'units_sold': [100, 150, 200, 120, 300, 50, 400, 250, 180, None], # Ada None untuk tes dropna
        'book_average_rating': [4.5, 4.2, 4.8, 3.9, 4.9, 3.5, 4.7, 4.1, 4.3, 4.0],
        'gross_sales': [5000000, 7500000, 12000000, 6000000, 18000000, 2000000, 25000000, 13000000, 9000000, 500000]
    }
    df = pd.DataFrame(data_dummy)
    
    # --- B. DATA CLEANING (WAJIB) ---
    # Menghapus baris yang memiliki nilai kosong (NaN)
    df.dropna(inplace=True)
    
    # --- C. TRAINING MODEL ---
    # Kita menggunakan units_sold dan book_average_rating untuk memprediksi gross_sales
    X = df[['units_sold', 'book_average_rating']]
    y = df['gross_sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, df

# Memanggil fungsi training saat aplikasi dimuat
try:
    model, df_clean = load_and_train_model()
except Exception as e:
    st.error(f"Terjadi kesalahan saat training model: {e}")
    st.stop()

# ==========================================
# 3. INTERFACE (UI) STREAMLIT
# ==========================================
st.title("📚 Bacakuy Sales Prediction & Islamic Strategy AI")
st.markdown("---")

# --- SIDEBAR INPUT ---
st.sidebar.header("🔍 Input Fitur Prediksi")

# Input Numerik untuk Units Sold
input_units = st.sidebar.number_input(
    "Jumlah Unit Terjual (Units Sold)", 
    min_value=0, 
    value=100,
    step=10,
    help="Masukkan perkiraan jumlah buku yang terjual."
)

# Input Numerik untuk Rating
input_rating = st.sidebar.number_input(
    "Rating Rata-rata Buku", 
    min_value=0.0, 
    max_value=5.0, 
    value=4.5,
    step=0.1,
    help="Skala 1.0 sampai 5.0"
)

# Tombol Prediksi
tombol_prediksi = st.sidebar.button("Prediksi Sekarang")

# ==========================================
# 4. LOGIKA PREDIKSI & GENAI
# ==========================================

if tombol_prediksi:
    # --- TAHAP 1: PREDIKSI ML ---
    # Membentuk array 2D karena sklearn membutuhkan format [[fitur1, fitur2]]
    input_data = [[input_units, input_rating]]
    
    # Melakukan prediksi
    hasil_prediksi = model.predict(input_data)[0]
    
    # Menampilkan Hasil Prediksi
    st.subheader("📊 Hasil Prediksi Penjualan")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Estimasi Gross Sales (IDR)", value=f"Rp {hasil_prediksi:,.0f}")
    
    with col2:
        st.info("Prediksi ini menggunakan algoritma Linear Regression berdasarkan data historis.")

    # --- TAHAP 2: ANALISIS GENAI (GEMINI) ---
    st.markdown("---")
    st.subheader("🤖 Analisis Strategi Bisnis Syariah (AI)")
    
    with st.spinner('Sedang meminta saran strategi syariah ke Gemini AI...'):
        try:
            # Membuat Prompt untuk AI
            prompt_text = f"""
            Anda adalah seorang konsultan bisnis Islami yang ahli.
            
            Data Prediksi:
            - Jumlah Unit Terjual: {input_units} buku
            - Rating Buku: {input_rating}/5.0
            - Estimasi Pendapatan Kotor: Rp {hasil_prediksi:,.0f}
            
            Tugas:
            1. Berikan analisis singkat mengenai performa penjualan ini.
            2. Berikan 3 strategi pemasaran yang sesuai dengan prinsip Syariah (Muamalah) untuk meningkatkan penjualan atau keberkahan pendapatan tersebut.
            3. Sertakan satu dalil (Ayat Quran atau Hadits) yang relevan tentang perdagangan yang jujur atau sedekah.
            """
            
            # Mengirim request ke Gemini
            # Menggunakan model 'gemini-pro' (pastikan API key valid)
            genai_model = genai.GenerativeModel('gemini-pro')
            response = genai_model.generate_content(prompt_text)
            
            # Menampilkan respon
            st.write(response.text)
            
        except Exception as e:
            st.warning("Gagal mendapatkan analisis AI. Pastikan API Key Gemini sudah benar.")
            st.error(f"Error details: {e}")

# ==========================================
# 5. DATA PREVIEW (OPSIONAL)
# ==========================================
with st.expander("Lihat Data Training (Cleaned)"):
    st.dataframe(df_clean)
```

---

### 🎓 Penjelasan Detail untuk Laporan Praktikum
