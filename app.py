import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. KONFIGURASI ---
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9mdHB1bHNxeGpoaHRmdWttbXRyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU1NzAwNjksImV4cCI6MjA4MTE0NjA2OX0.aDLgRF2mzaJEW43h2hmZOBadEnDtUoRTZCueJHdfh04"
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE"

# Inisialisasi Service
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    # Gunakan model ini untuk stabilitas koneksi
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal Inisialisasi: {e}")

@st.cache_data
def load_data():
    try:
        # Menghubungkan ke tabel bacakuy_sales
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
# BAGIAN ATAS: KALKULATOR PREDIKSI & ANALISIS AI
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
        # Prediksi Angka
        X = df[['units_sold', 'book_average_rating']]
        y = df['gross_sale']
        model_lr = LinearRegression().fit(X, y)
        prediction = model_lr.predict([[in_units, in_rating]])[0]
        
        st.metric("Estimasi Gross Sales", f"Rp {prediction:,.0f}")
        
        # Analisis AI
        st.subheader("🤖 Hasil Analisis Strategi AI")
        with st.spinner("Menghubungkan ke Google AI Studio..."):
            try:
                prompt = f"Berikan strategi bisnis singkat untuk penjualan buku dengan potensi Rp {prediction:,.0f}."
                response = model_ai.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"AI masih gagal terhubung. Detail: {e}")
    else:
        st.info("Silakan masukkan data dan klik tombol untuk analisis.")

st.divider()

# =========================================================
# BAGIAN TENGAH: STRATEGIC INTELLIGENCE (METRIK)
# =========================================================
st.title("🚀 Strategic Intelligence Hub")
if not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Sales", f"Rp {df['gross_sale'].sum():,.0f}")
    m2.metric("Total Units", f"{df['units_sold'].sum():,.0f}")
    m3.metric("Avg Rating", f"{df['book_average_rating'].mean():.2f}")
    m4.metric("Status", "Live Connected")

st.divider()

# =========================================================
# BAGIAN BAWAH: GOOGLE AI STUDIO INTERFACE & DROPDOWN FILTER
# =========================================================
st.title("📟 Google AI Studio & Analytics")
col_side, col_main = st.columns([1, 3])

with col_side:
    st.write("⚙️ **Filters**")
    # Filter Genre Dropdown (Pilihan ke bawah)
    genre_opt = ["Semua Genre"] + sorted(list(df['genre'].unique()))
    sel_genre = st.selectbox("Pilih Genre:", genre_opt)
    
    # Filter Rating Dropdown
    sel_rating = st.selectbox("Minimal Rating:", [0.0, 3.0, 4.0, 4.5, 5.0])
    
    st.write("---")
    st.write("**AI Studio Mode:**")
    st.caption("Active: Gemini 1.5 Flash")

# Logika Filter Data
df_f = df.copy()
if sel_genre != "Semua Genre":
    df_f = df_f[df_f['genre'] == sel_genre]
df_f = df_f[df_f['book_average_rating'] >= sel_rating]

with col_main:
    # Grafik yang berubah sesuai filter dropdown
    st.subheader(f"Statistik Penjualan: {sel_genre}")
    st.area_chart(df_f['gross_sale'])
    
    # Bagian Chat AI Studio (Asisten Kode)
    user_ask = st.text_input("Tanya AI Studio (Contoh: Bagaimana cara optimasi tabel?)")
    if user_ask:
        try:
            res_studio = model_ai.generate_content(user_ask)
            st.write(f"**AI Studio:** {res_studio.text}")
        except:
            st.warning("Chat AI Studio sedang sibuk.")
