import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- KONFIGURASI ---
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9mdHB1bHNxeGpoaHRmdWttbXRyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU1NzAwNjksImV4cCI6MjA4MTE0NjA2OX0.aDLgRF2mzaJEW43h2hmZOBadEnDtUoRTZCueJHdfh04"
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE"

# Inisialisasi Service
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    # PERBAIKAN: Menggunakan nama model yang lebih spesifik untuk menghindari 404
    model_ai = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error(f"Gagal Inisialisasi: {e}")

@st.cache_data
def load_data():
    try:
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
# BAGIAN 1: PREDIKSI & ANALISIS AI (BAGIAN ATAS)
# =========================================================
st.title("📑 Bacakuy Sales Prediction & AI Analysis")
col_in, col_res = st.columns([1, 2])

with col_in:
    st.subheader("🔍 Input Fitur")
    in_units = st.number_input("Unit Terjual", value=100)
    in_rating = st.slider("Rating Buku", 0.0, 5.0, 4.0)
    predict_btn = st.button("Aktifkan Analisis AI", use_container_width=True)

with col_res:
    if predict_btn and not df.empty:
        # Prediksi Linear Regression
        X = df[['units_sold', 'book_average_rating']]
        y = df['gross_sale']
        regr = LinearRegression().fit(X, y)
        prediction = regr.predict([[in_units, in_rating]])[0]
        
        st.metric("Estimasi Gross Sales", f"Rp {prediction:,.0f}")
        
        # Output Analisis AI
        st.subheader("🤖 Strategi Bisnis (AI Studio)")
        with st.spinner("Menghubungkan ke Gemini..."):
            try:
                # Memanggil AI untuk analisis otomatis
                prompt = f"Sebagai konsultan bisnis, berikan 1 strategi marketing syariah untuk buku dengan estimasi profit Rp {prediction:,.0f}."
                response = model_ai.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"AI masih gagal terhubung: {e}")

st.divider()

# =========================================================
# BAGIAN 2: STRATEGIC INTELLIGENCE HUB (METRIK & FILTER)
# =========================================================
st.title("🚀 Strategic Intelligence Hub")

if not df.empty:
    # FILTER DROPDOWN (Pilihan ke bawah)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        genre_list = ["Semua Kategori"] + sorted(list(df['genre'].unique()))
        selected_genre = st.selectbox("Filter Genre (Dropdown):", genre_list)
    with col_f2:
        rating_list = [0.0, 3.0, 4.0, 4.5, 5.0]
        selected_min_rating = st.selectbox("Minimal Rating (Dropdown):", rating_list)

    # Data Terfilter
    df_filtered = df.copy()
    if selected_genre != "Semua Kategori":
        df_filtered = df_filtered[df_filtered['genre'] == selected_genre]
    df_filtered = df_filtered[df_filtered['book_average_rating'] >= selected_min_rating]

    # Metrik Utama
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Valuation", f"Rp {df_filtered['gross_sale'].sum():,.0f}")
    m2.metric("Circulation", f"{df_filtered['units_sold'].sum():,.0f}")
    m3.metric("Brand Loyalty", f"{df_filtered['book_average_rating'].mean():.2f}/5")
    m4.metric("Status", "Live Production")

    st.subheader(f"Trend Penjualan: {selected_genre}")
    st.area_chart(df_filtered['gross_sale'])
