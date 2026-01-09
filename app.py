import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. KREDENSIAL ---
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9mdHB1bHNxeGpoaHRmdWttbXRyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU1NzAwNjksImV4cCI6MjA4MTE0NjA2OX0.aDLgRF2mzaJEW43h2hmZOBadEnDtUoRTZCueJHdfh04"
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE"

# Inisialisasi Service
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    
    # PERBAIKAN UTAMA: Menggunakan nama model lengkap dengan prefix 'models/'
    # Ini solusi untuk error 404 pada screenshot Anda
    model_ai = genai.GenerativeModel('models/gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal Inisialisasi: {e}")

@st.cache_data
def load_data():
    try:
        # Mengambil data dari bacakuy_sales
        res = supabase.table("bacakuy_sales").select("*").execute()
        df = pd.DataFrame(res.data)
        for col in ['book_average_rating', 'gross_sale']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        return df
    except:
        return pd.DataFrame()

st.set_page_config(page_title="Bacakuy Smart Monitor PRO", layout="wide")
df = load_data()

# =========================================================
# BAGIAN ATAS: PREDIKSI & ANALISIS AI
# =========================================================
st.title("📑 Bacakuy Sales Prediction & AI Analysis")
col_in, col_res = st.columns([1, 2])

with col_in:
    st.subheader("🔍 Input Data")
    in_units = st.number_input("Unit Terjual", value=100)
    in_rating = st.slider("Rating Buku", 0.0, 5.0, 4.0)
    predict_btn = st.button("Aktifkan Analisis AI", use_container_width=True)

with col_res:
    if predict_btn and not df.empty:
        # Prediksi Logika
        X = df[['units_sold', 'book_average_rating']]
        y = df['gross_sale']
        regr = LinearRegression().fit(X, y)
        prediction = regr.predict([[in_units, in_rating]])[0]
        
        st.metric("Estimasi Gross Sales", f"Rp {prediction:,.0f}")
        
        st.subheader("🤖 Strategi Bisnis (AI Studio)")
        with st.spinner("Sedang memproses strategi..."):
            try:
                # Prompt untuk AI
                prompt = f"Berikan strategi marketing syariah singkat untuk profit Rp {prediction:,.0f}."
                # Menggunakan generate_content dengan penanganan error 404
                response = model_ai.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                # Jika masih gagal, coba gunakan model cadangan (gemini-pro)
                try:
                    alt_model = genai.GenerativeModel('models/gemini-pro')
                    response = alt_model.generate_content(prompt)
                    st.info(response.text)
                except:
                    st.error(f"AI masih tidak merespon. Mohon periksa API Key Anda di Google AI Studio. {e}")

st.divider()

# =========================================================
# BAGIAN BAWAH: STRATEGIC HUB DENGAN FILTER DROPDOWN
# =========================================================
st.title("🚀 Strategic Intelligence Hub")

if not df.empty:
    # FILTER DROPDOWN (Pilihan ke bawah)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        genres = ["Semua Genre"] + sorted(list(df['genre'].unique()))
        sel_genre = st.selectbox("Pilih Genre (Dropdown):", genres)
    with col_f2:
        ratings = [0.0, 3.0, 4.0, 4.5, 5.0]
        sel_rating = st.selectbox("Pilih Minimal Rating:", ratings)

    # Filter Data
    df_f = df.copy()
    if sel_genre != "Semua Genre":
        df_f = df_f[df_f['genre'] == sel_genre]
    df_f = df_f[df_f['book_average_rating'] >= sel_rating]

    # Metrik Utama
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Valuation", f"Rp {df_f['gross_sale'].sum():,.0f}")
    m2.metric("Circulation", f"{df_f['units_sold'].sum():,.0f}")
    m3.metric("Brand Loyalty", f"{df_f['book_average_rating'].mean():.2f}/5")
    m4.metric("Status", "Live Production")

    # Grafik Tren
    st.subheader(f"Statistik Penjualan: {sel_genre}")
    st.area_chart(df_f['gross_sale'])
