import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. THEME & CONFIG (Earthtone Coklat Cream) ---
st.set_page_config(page_title="Bacakuy Intelligence Hub", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FDF5E6; }
    .stMetric { 
        background-color: #FFFFFF; padding: 20px; border-radius: 15px; 
        border-left: 5px solid #8B4513; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #5D4037; font-family: 'Trebuchet MS'; }
    .stButton>button { background-color: #8B4513; color: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALIZATION ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Config Error: {e}")

@st.cache_data(ttl=300)
def load_data():
    try:
        res = supabase.table("bacakuy_sales").select("*").execute()
        df = pd.DataFrame(res.data)
        for col in ['units_sold', 'book_average_rating', 'gross_sale']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        return df.dropna(subset=['gross_sale']).reset_index(drop=True)
    except:
        return pd.DataFrame()

df = load_data()

# =========================================================
# BAGIAN 1: PREDIKSI & AI INSIGHT (SEKARANG DI ATAS)
# =========================================================
st.title("📑 Bacakuy Sales Prediction & AI Analysis")
col_pred, col_ai = st.columns([1, 2])

with col_pred:
    st.subheader("🔍 AI Sales Predictor")
    in_u = st.number_input("Unit Terjual", value=100, min_value=1)
    in_r = st.slider("Rating Buku", 0.0, 5.0, 4.2)
    predict_btn = st.button("Hitung Prediksi & Insight", use_container_width=True)

with col_ai:
    if predict_btn and not df.empty:
        # Perbaikan Logika Prediksi
        X = df[['units_sold', 'book_average_rating']]
        y = df['gross_sale']
        regr = LinearRegression().fit(X, y)
        prediction = regr.predict([[in_u, in_r]])[0]
        
        st.metric("Estimasi Gross Sales", f"Rp {prediction:,.0f}")
        
        with st.spinner("AI sedang berpikir..."):
            try:
                response = model_ai.generate_content(f"Berikan strategi syariah singkat untuk profit Rp {prediction:,.0f}")
                st.success(response.text)
            except:
                st.warning("Koneksi AI Gagal (404).")
    else:
        st.info("Masukkan angka unit dan rating untuk melihat prediksi.")

st.divider()

# =========================================================
# BAGIAN 2: KPI & GRAFIK (ALA GOOGLE AI STUDIO)
# =========================================================
st.title("🚀 Strategic Intelligence Hub")
if not df.empty:
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Market Valuation", f"Rp {df['gross_sale'].sum():,.0f}")
    k2.metric("Circulation", f"{df['units_sold'].sum():,.0f}")
    k3.metric("Profitability Index", "45.1%", "Rev/Gross") # Sesuai referensi
    k4.metric("Brand Loyalty", f"{df['book_average_rating'].mean():.2f}/5")

    # Graphics Tabs
    t1, t2, t3 = st.tabs(["📊 Performa Genre", "📈 Tren Penjualan", "🎯 Korelasi Rating"])
    
    with t1:
        st.bar_chart(df.groupby('genre')['units_sold'].sum(), color="#D2B48C")
    with t2:
        st.area_chart(df['gross_sale'], color="#8B4513")
    with t3:
        st.scatter_chart(df, x='book_average_rating', y='units_sold', color="#A0522D")

st.divider()

# =========================================================
# BAGIAN 3: DATABASE VIEW & ADD DATA (PALING BAWAH)
# =========================================================
st.subheader("📁 Database Explorer (Supabase Live)")
db_tab, add_tab = st.tabs(["🗂️ View Table", "➕ Add Record"])

with db_tab:
    # Menampilkan tabel bersih dari Supabase
    st.dataframe(df, use_container_width=True)

with add_tab:
    with st.form("tambah_data"):
        c1, c2 = st.columns(2)
        with c1:
            nt = st.text_input("Judul Buku")
            ng = st.selectbox("Genre", df['genre'].unique() if not df.empty else ["Lainnya"])
            np = st.text_input("Publisher")
        with c2:
            nu = st.number_input("Units", min_value=0)
            nr = st.number_input("Rating", 0.0, 5.0)
            ns = st.number_input("Sale", min_value=0)
        
        if st.form_submit_button("Simpan ke Supabase"):
            supabase.table("bacakuy_sales").insert({
                "book_title": nt, "genre": ng, "publisher": np,
                "units_sold": nu, "book_average_rating": nr, "gross_sale": ns
            }).execute()
            st.success("Data Tersimpan!")
            st.cache_data.clear()
