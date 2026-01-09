import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. DEFINISI KREDENSIAL (WAJIB DI ATAS) ---
# Menetapkan variabel agar tidak terjadi error 'not defined'
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE"
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9mdHB1bHNxeGpoaHRmdWttbXRyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU1NzAwNjksImV4cCI6MjA4MTE0NjA2OX0.aDLgRF2mzaJEW43h2hmZOBadEnDtUoRTZCueJHdfh04"

# --- 2. INISIALISASI SERVICE ---
try:
    # Inisialisasi Supabase
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Inisialisasi AI Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    
    # PERBAIKAN MODEL: Menggunakan 'gemini-1.5-flash' yang paling stabil
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal Inisialisasi Service: {e}")

@st.cache_data
def load_data():
    try:
        # Mengambil data dari tabel bacakuy_sales
        res = supabase.table("bacakuy_sales").select("*").execute()
        df = pd.DataFrame(res.data)
        
        # Membersihkan data angka (koma ke titik)
        for col in ['book_average_rating', 'gross_sale']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        return df
    except Exception as e:
        st.error(f"Gagal memuat data dari Supabase: {e}")
        return pd.DataFrame()

# --- 3. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Bacakuy Smart Monitor PRO", layout="wide")
df = load_data()

# =========================================================
# BAGIAN ATAS: PREDIKSI & INSIGHT AI
# =========================================================
st.title("📑 Bacakuy Sales Prediction & AI Analysis")
st.write("Dapatkan estimasi profit dan insight strategi bisnis secara real-time.")

col_input, col_insight = st.columns([1, 2])

with col_input:
    st.subheader("🔍 Input Fitur")
    in_units = st.number_input("Unit Terjual", value=100)
    in_rating = st.slider("Rating Buku", 0.0, 5.0, 4.0)
    # Tombol pemicu analisis AI
    predict_btn = st.button("Aktifkan Analisis AI", use_container_width=True)

with col_insight:
    if predict_btn and not df.empty:
        # Logika Prediksi (Linear Regression)
        X = df[['units_sold', 'book_average_rating']]
        y = df['gross_sale']
        model_lr = LinearRegression().fit(X, y)
        prediction = model_lr.predict([[in_units, in_rating]])[0]
        
        # Menampilkan metrik hasil prediksi
        st.metric("Estimasi Gross Sales (IDR)", f"Rp {prediction:,.0f}")
        
        # --- EKSEKUSI INSIGHT AI ---
        st.subheader("🤖 Strategi Bisnis & Insight (AI)")
        with st.spinner("AI sedang menganalisis data Anda..."):
            try:
                # Memberikan instruksi spesifik ke AI
                prompt = f"""
                Analisis data berikut:
                - Estimasi Gross Sales: Rp {prediction:,.0f}
                - Unit Terjual: {in_units}
                - Rating: {in_rating}
                
                Berikan 1 strategi bisnis syariah yang konkret dan 1 insight untuk meningkatkan profit.
                """
                response = model_ai.generate_content(prompt)
                st.info(response.text) # Menampilkan teks insight dari AI
            except Exception as e:
                st.error(f"AI Error: {e}. Pastikan kuota API Gemini Anda mencukupi.")
    else:
        st.info("Masukkan data di sebelah kiri dan klik tombol untuk mendapatkan insight AI.")

st.divider()

# =========================================================
# BAGIAN BAWAH: STRATEGIC INTELLIGENCE (METRIK & DROPDOWN)
# =========================================================
st.title("🚀 Strategic Intelligence Hub")

if not df.empty:
    # FILTER DROPDOWN (Pilihan ke bawah)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        genres = ["Semua Kategori"] + sorted(list(df['genre'].unique()))
        selected_genre = st.selectbox("Pilih Kategori Buku:", genres)
    with col_f2:
        ratings = [0.0, 3.0, 4.0, 4.5, 5.0]
        selected_min_rating = st.selectbox("Filter Minimal Rating:", ratings)

    # Filter Data Berdasarkan Dropdown
    df_f = df.copy()
    if selected_genre != "Semua Kategori":
        df_f = df_f[df_f['genre'] == selected_genre]
    df_f = df_f[df_f['book_average_rating'] >= selected_min_rating]

    # Menampilkan Metrik Utama
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Valuation", f"Rp {df_f['gross_sale'].sum():,.0f}", "+5.2%")
    m2.metric("Circulation", f"{df_f['units_sold'].sum():,.0f}", "Units Delivered")
    m3.metric("Brand Loyalty", f"{df_f['book_average_rating'].mean():.2f}/5", "Avg Sentiments")
    m4.metric("Status", "Live Production", "Active")

    # Visualisasi Tren
    st.subheader(f"Trend Penjualan: {selected_genre}")
    st.area_chart(df_f['gross_sale'])
else:
    st.warning("Data Supabase tidak ditemukan. Periksa koneksi tabel 'bacakuy_sales'.")
