import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. CONFIG & EARTH TONE THEME ---
st.set_page_config(page_title="Bacakuy Strategic Monitor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #FDF5E6; }
    .stMetric { 
        background-color: #FFFFFF; padding: 20px; border-radius: 15px; 
        border-left: 5px solid #8B4513; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #5D4037; font-family: 'Trebuchet MS'; }
    .stButton>button { background-color: #8B4513; color: white; border-radius: 10px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INITIALIZATION ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Koneksi Gagal: {e}")

@st.cache_data(ttl=60)
def load_data():
    try:
        res = supabase.table("bacakuy_sales").select("*").execute()
        df = pd.DataFrame(res.data)
        # Pastikan kolom angka bersih dan akurat
        for col in ['units_sold', 'book_average_rating', 'gross_sale']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        return df.dropna(subset=['gross_sale']).reset_index(drop=True)
    except:
        return pd.DataFrame()

df = load_data()

# =========================================================
# BAGIAN 1: PREDIKSI & AI INSIGHT (DI ATAS)
# =========================================================
st.title("📑 Bacakuy Sales Prediction & AI Analysis")
col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("🔍 AI Sales Predictor")
    target_u = st.number_input("Unit Terjual", value=100, min_value=1)
    target_r = st.slider("Rating Buku", 0.0, 5.0, 4.20)
    calc_btn = st.button("Hitung Prediksi & Strategi")

with col_result:
    if calc_btn and not df.empty:
        # Logika Prediksi Linear Regression
        X_reg = df[['units_sold', 'book_average_rating']]
        y_reg = df['gross_sale']
        model_reg = LinearRegression().fit(X_reg, y_reg)
        pred_val = model_reg.predict([[target_u, target_r]])[0]
        
        st.metric("Estimasi Gross Sales", f"Rp {pred_val:,.0f}")
        
        with st.spinner("AI menyusun strategi..."):
            try:
                prompt = f"Berikan 1 strategi marketing syariah untuk target sales Rp {pred_val:,.0f}"
                response = model_ai.generate_content(prompt)
                st.success(response.text)
            except:
                st.warning("Insight AI Gagal (404). Silakan Reboot aplikasi.")
    else:
        st.info("Gunakan panel kiri untuk memulai simulasi.")

st.divider()

# =========================================================
# BAGIAN 2: KPI & GRAFIK (ALA GOOGLE AI STUDIO)
# =========================================================
st.title("🚀 Strategic Intelligence Hub")

if not df.empty:
    # KPI Row - Referensi Google AI Studio
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Market Valuation", f"Rp {df['gross_sale'].sum():,.0f}")
    k2.metric("Circulation", f"{df['units_sold'].sum():,.0f}")
    k3.metric("Profitability Index", "45.1%", "Rev/Gross") #
    k4.metric("Brand Loyalty", f"{df['book_average_rating'].mean():.2f}/5")

    # Grafik Multivariat
    g_tab1, g_tab2 = st.tabs(["📊 Performance Intelligence", "📈 Operational Trends"])
    
    with g_tab1:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Units Sold by Genre") #
            st.bar_chart(df.groupby('genre')['units_sold'].sum(), color="#D2B48C")
        with col_g2:
            st.subheader("Top Publishers Revenue") #
            top_pub = df.groupby('publisher')['gross_sale'].sum().nlargest(5)
            st.bar_chart(top_pub, color="#BC8F8F")
            
    with g_tab2:
        st.subheader("Monthly Sales Trend") #
        st.area_chart(df['gross_sale'], color="#8B4513")

st.divider()

# =========================================================
# BAGIAN 3: DATABASE EXPLORER (PALING BAWAH)
# =========================================================
st.title("📁 Database Management")
tab_view, tab_add = st.tabs(["🗂️ View Supabase Data", "➕ Add New Record"])

with tab_view:
    # Menampilkan tabel data clean secara transparan
    st.dataframe(df, use_container_width=True)

with tab_add:
    with st.form("input_baru"):
        c1, c2 = st.columns(2)
        with c1:
            in_t = st.text_input("Judul Buku")
            in_g = st.selectbox("Genre", df['genre'].unique() if not df.empty else ["General"])
            in_p = st.text_input("Publisher")
        with c2:
            in_u = st.number_input("Units Sold", min_value=0)
            in_r = st.number_input("Avg Rating", 0.0, 5.0)
            in_s = st.number_input("Gross Sale", min_value=0)
        
        if st.form_submit_button("Submit ke Supabase"):
            supabase.table("bacakuy_sales").insert({
                "book_title": in_t, "genre": in_g, "publisher": in_p,
                "units_sold": in_u, "book_average_rating": in_r, "gross_sale": in_s
            }).execute()
            st.success("Berhasil! Refresh halaman untuk update.")
            st.cache_data.clear()
