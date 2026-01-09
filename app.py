import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. KREDENSIAL ---
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9mdHB1bHNxeGpoaHRmdWttbXRyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU1NzAwNjksImV4cCI6MjA4MTE0NjA2OX0.aDLgRF2mzaJEW43h2hmZOBadEnDtUoRTZCueJHdfh04"
# API Key Google AI Studio Anda
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE"

# Inisialisasi Service dengan penanganan error 404
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    # Menggunakan model 'gemini-1.5-flash' yang mendukung generateContent
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal Inisialisasi: {e}")

@st.cache_data
def load_data():
    try:
        # Nama tabel diperbaiki menjadi bacakuy_sales tanpa spasi
        res = supabase.table("bacakuy_sales").select("*").execute()
        df = pd.DataFrame(res.data)
        # Normalisasi angka koma ke titik
        for col in ['book_average_rating', 'gross_sale']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        return df
    except Exception as e:
        st.error(f"Gagal koneksi database: {e}")
        return pd.DataFrame()

# --- SETUP TAMPILAN ---
st.set_page_config(page_title="Bacakuy Smart Monitor PRO", layout="wide")
df = load_data()

# =========================================================
# BAGIAN ATAS: KALKULATOR PREDIKSI & AI STRATEGY
# =========================================================
st.title("📑 Bacakuy Sales Prediction & Islamic Strategy AI")
st.write("Prediksi profit masa depan dan dapatkan insight strategi bisnis syariah.")

col_in, col_res = st.columns([1, 2])

with col_in:
    st.subheader("🔍 Input Fitur Prediksi")
    in_units = st.number_input("Jumlah Unit Terjual (Units)", value=100)
    in_rating = st.slider("Rating Rata-rata Buku", 0.0, 5.0, 4.0)
    predict_btn = st.button("Prediksi Sekarang", use_container_width=True)

with col_res:
    if predict_btn and not df.empty:
        # Perhitungan Prediksi (Linear Regression)
        X = df[['units_sold', 'book_average_rating']]
        y = df['gross_sale']
        model_lr = LinearRegression().fit(X, y)
        prediction = model_lr.predict([[in_units, in_rating]])[0]
        
        st.subheader("📊 Hasil Prediksi Penjualan")
        st.metric("Estimasi Gross Sales (IDR)", f"Rp {prediction:,.0f}")
        
        # Eksekusi AI Gemini
        st.subheader("☪️ Analisis Strategi Bisnis Syariah (AI)")
        with st.spinner("AI sedang merancang strategi..."):
            try:
                prompt = f"Berikan strategi pemasaran syariah yang jujur untuk buku dengan potensi profit Rp {prediction:,.0f}."
                response = model_ai.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"Koneksi AI Gagal: {e}. Pastikan API Key di Google AI Studio aktif.")
    else:
        st.info("Klik 'Prediksi Sekarang' untuk melihat hasil.")

st.divider()

# =========================================================
# BAGIAN TENGAH: STRATEGIC INTELLIGENCE HUB (METRIK UTAMA)
# =========================================================
st.title("🚀 Strategic Intelligence Hub")
if not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    # Metrik real-time sesuai desain
    m1.metric("Market Valuation", f"Rp {df['gross_sale'].sum():,.0f}", "+5.2%")
    m2.metric("Circulation", f"{df['units_sold'].sum():,.0f}", "Units Delivered")
    m3.metric("Brand Loyalty", f"{df['book_average_rating'].mean():.2f}/5", "Avg Sentiments")
    m4.metric("Status", "Live Production", "Active")

st.divider()

# =========================================================
# BAGIAN BAWAH: ANALYTICS DENGAN FILTER DROPDOWN
# =========================================================
st.title("🤖 Performance Analytics & Filter Hub")
col_side, col_main = st.columns([1, 3])

with col_side:
    st.write("⚙️ **Dropdown Filter**")
    # Filter pilihan ke bawah (Dropdown)
    genre_list = ["Semua Kategori"] + sorted(list(df['genre'].unique()))
    selected_genre = st.selectbox("Pilih Genre Buku:", genre_list)
    
    rating_list = [0.0, 1.0, 2.0, 3.0, 4.0, 4.5, 5.0]
    selected_min_rating = st.selectbox("Minimal Rating Buku:", rating_list, index=0)
    
    st.divider()
    st.button("Analyze Author Performance", use_container_width=True)
    st.button("Track Profitability", use_container_width=True)

# Logika Filter
df_final = df.copy()
if selected_genre != "Semua Kategori":
    df_final = df_final[df_final['genre'] == selected_genre]
df_final = df_final[df_final['book_average_rating'] >= selected_min_rating]

with col_main:
    tab1, tab2, tab3 = st.tabs(["Monthly Trend", "Units Distribution", "Correlation Analysis"])
    
    with tab1:
        st.subheader(f"Trend Penjualan: {selected_genre}")
        # Area chart sesuai Operational Trends
        st.area_chart(df_final['gross_sale'])
        
    with tab2:
        st.subheader("Units Sold per Genre")
        # Bar chart horizontal
        genre_summary = df_final.groupby('genre')['units_sold'].sum()
        st.bar_chart(genre_summary, horizontal=True)
        
    with tab3:
        st.subheader("Korelasi Rating vs Popularitas")
        # Scatter chart Market Popularity
        st.scatter_chart(df_final[['book_average_rating', 'units_sold']])
