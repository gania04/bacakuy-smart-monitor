import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. KREDENSIAL (PASTIKAN KEY BENAR) ---
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9mdHB1bHNxeGpoaHRmdWttbXRyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU1NzAwNjksImV4cCI6MjA4MTE0NjA2OX0.aDLgRF2mzaJEW43h2hmZOBadEnDtUoRTZCueJHdfh04"

# Ganti dengan API Key dari Google AI Studio Anda
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE" 

# --- 2. INISIALISASI ---
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    # Gunakan model terbaru yang stabil
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Error Inisialisasi: {e}")

@st.cache_data
def load_data():
    try:
        res = supabase.table("bacakuy_sales_clean").select("*").execute()
        return pd.DataFrame(res.data)
    except:
        return pd.DataFrame()

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Bacakuy Smart Monitor Pro", layout="wide")
df = load_data()

# =========================================================
# BAGIAN 1: KALKULATOR PREDIKSI & AI (POSISI ATAS)
# =========================================================
st.title("📑 Bacakuy Sales Prediction & Islamic Strategy AI")
st.write("Masukkan data untuk mendapatkan estimasi profit dan strategi bisnis.")

col_calc, col_result = st.columns([1, 2])

with col_calc:
    st.subheader("🔍 Input Fitur Prediksi")
    in_units = st.number_input("Jumlah Unit Terjual (Units Sold)", value=100)
    in_rating = st.slider("Rating Rata-rata Buku", 0.0, 5.0, 4.08) # Default rating dari data Anda
    predict_btn = st.button("Prediksi Sekarang", use_container_width=True)

with col_result:
    if predict_btn and not df.empty:
        # Perhitungan Linear Regression
        X = df[['units_sold', 'book_average_rating']]
        y = df['gross_sale']
        regr = LinearRegression().fit(X, y)
        prediction = regr.predict([[in_units, in_rating]])[0]
        
        st.subheader("📊 Hasil Prediksi Penjualan")
        st.metric("Estimasi Gross Sales (IDR)", f"Rp {prediction:,.0f}")
        
        # --- KONEKSI AI GEMINI ---
        st.subheader("☪️ Analisis Strategi Bisnis Syariah (AI)")
        with st.spinner("AI sedang merancang strategi..."):
            try:
                # Prompt yang dikirim ke AI Studio
                prompt = f"""
                Anda adalah konsultan bisnis syariah untuk Bacakuy. 
                Data Penjualan: {in_units} unit buku dengan rating {in_rating}.
                Estimasi Profit: Rp {prediction:,.0f}.
                
                Berikan 1 strategi pemasaran kreatif dan 1 nasihat muamalah syariah tentang kejujuran.
                """
                response = model_ai.generate_content(prompt)
                st.markdown(response.text) # Menampilkan teks dari AI
            except Exception as e:
                st.error(f"AI gagal merespons: {e}. Pastikan API Key Gemini di Google AI Studio sudah aktif.")
    else:
        st.info("Silakan klik 'Prediksi Sekarang' untuk melihat analisis.")

st.markdown("---")

# =========================================================
# BAGIAN 2: DASHBOARD MONITORING (POSISI BAWAH)
# =========================================================
st.title("📊 Strategic Intelligence Hub")
if not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Valuation", f"Rp {df['gross_sale'].sum():,.0f}", "+5.2%")
    m2.metric("Circulation", f"{df['units_sold'].sum():,.0f}", "Units Delivered")
    m3.metric("Brand Loyalty", f"{df['book_average_rating'].mean():.2f}/5", "Avg Sentiments")
    m4.metric("Status", "Live Production", "Active")
    
    st.subheader("Sales Performance Trend")
    st.area_chart(df['gross_sale'])
