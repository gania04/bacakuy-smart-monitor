import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. KONFIGURASI KREDENSIAL ---
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9mdHB1bHNxeGpoaHRmdWttbXRyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU1NzAwNjksImV4cCI6MjA4MTE0NjA2OX0.aDLgRF2mzaJEW43h2hmZOBadEnDtUoRTZCueJHdfh04"
# Masukkan API Key dari Google AI Studio (berawalan AIzaSy...)
GEMINI_API_KEY = "MASUKKAN_API_KEY_GOOGLE_AI_STUDIO_ANDA"

# Inisialisasi Service
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal Inisialisasi: {e}")

@st.cache_data
def load_data():
    try:
        # Terhubung ke tabel bacakuy_sales sesuai permintaan Anda
        res = supabase.table("bacakuy_sales").select("*").execute()
        df = pd.DataFrame(res.data)
        # Pembersihan data otomatis (koma ke titik)
        for col in ['book_average_rating', 'gross_sale']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        return df
    except:
        return pd.DataFrame()

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Bacakuy Smart Monitor PRO", layout="wide")
df = load_data()

# =========================================================
# BAGIAN ATAS: KALKULATOR PREDIKSI & AI INSIGHT
# =========================================================
st.title("📑 Bacakuy Sales Prediction & Islamic Strategy AI")
col_in, col_res = st.columns([1, 2])

with col_in:
    st.subheader("🔍 Input Fitur Prediksi")
    u_input = st.number_input("Jumlah Unit Terjual", value=100)
    r_input = st.slider("Rating Buku", 0.0, 5.0, 4.0)
    btn_predict = st.button("Prediksi Sekarang", use_container_width=True)

with col_res:
    if btn_predict and not df.empty:
        # Prediksi Linear Regression
        X = df[['units_sold', 'book_average_rating']]
        y = df['gross_sale']
        model = LinearRegression().fit(X, y)
        pred = model.predict([[u_input, r_input]])[0]
        
        st.metric("Estimasi Gross Sales (IDR)", f"Rp {pred:,.0f}")
        
        # AI Insight
        st.subheader("☪️ Analisis Strategi Bisnis Syariah")
        try:
            prompt = f"Berikan 1 strategi marketing syariah untuk target profit Rp {pred:,.0f}."
            response = model_ai.generate_content(prompt)
            st.info(response.text)
        except:
            st.warning("Gagal terhubung ke AI. Pastikan API Key Google AI Studio benar.")

st.divider()

# =========================================================
# BAGIAN TENGAH: STRATEGIC INTELLIGENCE HUB (Supabase Data)
# =========================================================
st.title("🚀 Strategic Intelligence Hub")
if not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Valuation", f"Rp {df['gross_sale'].sum():,.0f}")
    m2.metric("Circulation", f"{df['units_sold'].sum():,.0f}")
    m3.metric("Brand Loyalty", f"{df['book_average_rating'].mean():.2f}/5")
    m4.metric("Status", "Live Production")

st.divider()

# =========================================================
# BAGIAN BAWAH: GOOGLE AI STUDIO INTERFACE & PERFORMANCE ANALYTICS
# =========================================================
st.title("🤖 Google AI Studio & Performance Analytics")
col_side, col_main = st.columns([1, 3])

with col_side:
    st.write("⚙️ **Code Assistant**")
    st.caption("kualitas berkorelasi dengan volume?")
    st.write("**Suggestions:**")
    st.button("Analyze Author Performance")
    st.button("Track Profitability")
    st.text_area("Ask AI Studio...", placeholder="Make changes, add new features...")

with col_main:
    tab1, tab2, tab3 = st.tabs(["Monthly Trend", "Units by Genre", "Performance Intelligence"])
    
    with tab1:
        st.subheader("Monthly Sales Trend")
        # Meniru grafik garis Operational Trends
        st.line_chart(df['gross_sale'])
        
    with tab2:
        st.subheader("Units Sold by Genre")
        # Meniru grafik batang horizontal
        genre_data = df.groupby('genre')['units_sold'].sum().sort_values()
        st.bar_chart(genre_data, horizontal=True)
        
    with tab3:
        st.subheader("Rating vs Market Popularity")
        # Meniru scatter plot korelasi
        st.scatter_chart(df[['book_average_rating', 'units_sold']])
