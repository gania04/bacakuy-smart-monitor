import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Bacakuy Smart Monitor PRO", layout="wide")

# --- 2. KREDENSIAL DARI SECRETS ---
try:
    # Memastikan Secrets terbaca dengan benar dari menu Settings
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal memuat konfigurasi: {e}")

@st.cache_data
def load_data():
    try:
        res = supabase.table("bacakuy_sales").select("*").execute()
        df = pd.DataFrame(res.data)
        # PERBAIKAN GRAFIK: Memaksa kolom menjadi angka murni agar tidak "Unrecognized"
        for col in ['units_sold', 'book_average_rating', 'gross_sale']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        return df.dropna(subset=['gross_sale']).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

df = load_data()

# =========================================================
# BAGIAN 1: KALKULATOR PREDIKSI & AI
# =========================================================
st.title("📑 Bacakuy Sales Prediction & AI Analysis")
st.info("Gunakan kalkulator di bawah untuk simulasi pendapatan.")

col_in, col_res = st.columns([1, 2])

with col_in:
    st.subheader("🔍 Kalkulator Prediksi")
    in_units = st.number_input("Target Unit Terjual", min_value=1, value=100)
    in_rating = st.slider("Target Rating Buku", 0.0, 5.0, 4.20)
    predict_btn = st.button("Hitung Prediksi & Insight AI", use_container_width=True)

with col_res:
    if predict_btn and not df.empty:
        # Kalkulasi Prediksi (Linear Regression)
        X = df[['units_sold', 'book_average_rating']]
        y = df['gross_sale']
        regr = LinearRegression().fit(X, y)
        prediction = regr.predict([[in_units, in_rating]])[0]
        
        st.metric("Estimasi Gross Sales (IDR)", f"Rp {prediction:,.0f}")
        
        # Penanganan Insight AI
        st.subheader("🤖 Strategi Bisnis Syariah (AI Insight)")
        with st.spinner("Menghubungkan ke AI Studio..."):
            try:
                response = model_ai.generate_content(f"Berikan 1 strategi bisnis syariah untuk profit Rp {prediction:,.0f}")
                st.success(response.text)
            except Exception:
                st.warning("Insight AI sementara tidak tersedia (Error 404). Silakan lakukan Reboot App.")
    else:
        st.info("Klik tombol untuk melihat hasil prediksi.")

st.divider()

# =========================================================
# BAGIAN 2: STRATEGIC HUB (METRIK & GRAFIK)
# =========================================================
st.title("🚀 Strategic Intelligence Hub")

if not df.empty:
    # Filter Dropdown yang menarik ke bawah
    c1, c2 = st.columns(2)
    with c1:
        genres = ["Semua Kategori"] + sorted(list(df['genre'].unique()))
        sel_genre = st.selectbox("Pilih Kategori Buku:", genres)
    with c2:
        sel_rating = st.selectbox("Minimal Rating:", [0.0, 3.0, 4.0, 4.5, 5.0])

    # Filter Data
    df_f = df.copy()
    if sel_genre != "Semua Kategori":
        df_f = df_f[df_f['genre'] == sel_genre]
    df_f = df_f[df_f['book_average_rating'] >= sel_rating]

    # Metrik Dashboard
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Valuation", f"Rp {df_f['gross_sale'].sum():,.0f}")
    m2.metric("Total Units", f"{df_f['units_sold'].sum():,.0f}")
    m3.metric("Avg Rating", f"{df_f['book_average_rating'].mean():.2f}")
    m4.metric("Status", "Live Sync")

    # PERBAIKAN GRAFIK: Menghilangkan error "Unrecognized data set"
    st.subheader(f"Tren Penjualan: {sel_genre}")
    if not df_f.empty:
        st.area_chart(df_f.set_index(df_f.index)['gross_sale'])
    else:
        st.warning("Tidak ada data untuk filter ini.")
