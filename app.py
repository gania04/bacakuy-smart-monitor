import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- 1. CONFIG & THEME ---
st.set_page_config(page_title="Bacakuy Intelligence Hub", layout="wide")

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
    st.error(f"Konfigurasi Error: {e}")

@st.cache_data(ttl=10)
def load_data():
    try:
        res = supabase.table("bacakuy_sales").select("*").execute()
        df = pd.DataFrame(res.data)
        if df.empty: return df
        # Konversi numerik yang aman
        for col in ['units_sold', 'book_average_rating', 'gross_sale']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        
        target_date_col = 'tanggal_transaksi' if 'tanggal_transaksi' in df.columns else 'created_at'
        if target_date_col in df.columns:
            df['dt_temp'] = pd.to_datetime(df[target_date_col])
            df['bulan_tahun'] = df['dt_temp'].dt.strftime('%B %Y')
            df = df.sort_values('dt_temp')
        else:
            df['bulan_tahun'] = "No Date"
        return df.dropna(subset=['gross_sale']).reset_index(drop=True)
    except Exception as e:
        return pd.DataFrame()

df_raw = load_data()

# =========================================================
# BAGIAN 1: PREDIKSI & AI INSIGHT (RATING BOOSTER)
# =========================================================
st.title("📑 Bacakuy Sales Prediction & AI Analysis")
col_p1, col_p2 = st.columns([1, 2])

with col_p1:
    st.subheader("🔍 AI Sales Predictor")
    in_u = st.number_input("Unit Terjual", value=100, min_value=1)
    in_r = st.slider("Rating Buku", 0.0, 5.0, 4.2)
    btn_predict = st.button("Hitung & Dapatkan Insight")

with col_p2:
    if btn_predict and not df_raw.empty:
        X = df_raw[['units_sold', 'book_average_rating']]
        y = df_raw['gross_sale']
        model = LinearRegression().fit(X, y)
        base_prediction = model.predict([[in_u, in_r]])[0]
        
        # Logika Rating Booster
        rating_impact = (in_r - 3.5) * (base_prediction * 0.1) 
        final_prediction = max(0, base_prediction + rating_impact)
        
        st.metric("Estimasi Gross Sales", f"Rp {final_prediction:,.0f}")
        try:
            resp = model_ai.generate_content(f"Strategi syariah untuk buku rating {in_r} agar profit Rp {final_prediction:,.0f} meningkat.")
            st.success(resp.text)
        except:
            st.warning("Insight AI Gagal.")

st.divider()

# =========================================================
# BAGIAN 2: STRATEGIC HUB
# =========================================================
st.title("🚀 Strategic Intelligence Hub")

if not df_raw.empty:
    f1, f2 = st.columns(2)
    with f1:
        sel_genre = st.selectbox("Pilih Genre:", ["Semua Genre"] + sorted(list(df_raw['genre'].unique())))
    with f2:
        list_bulan = df_raw['bulan_tahun'].unique().tolist()
        sel_month = st.selectbox("Pilih Bulan Transaksi:", ["Semua Bulan"] + list_bulan)

    df = df_raw.copy()
    if sel_genre != "Semua Genre": df = df[df['genre'] == sel_genre]
    if sel_month != "Semua Bulan": df = df[df['bulan_tahun'] == sel_month]

    # KPI Row (Statis 45.1% sesuai instruksi)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Market Valuation", f"Rp {df['gross_sale'].sum():,.0f}")
    k2.metric("Circulation", f"{df['units_sold'].sum():,.0f}")
    k3.metric("Profitability Index", "45.1%", "Rev/Gross")
    k4.metric("Brand Loyalty", f"{df['book_average_rating'].mean():.2f}/5")

    # TABS GRAFIK
    t1, t2, t3, t4 = st.tabs(["📊 Performance", "📈 Monthly Trend", "🎯 Popularity", "✍️ Author Performance"])
    
    with t1:
        st.subheader("Publisher Performance")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("**Revenue by Publisher**")
            # Perbaikan filter kolom agar tidak error
            pub_rev = df.groupby('publisher')['gross_sale'].sum().nlargest(5).reset_index()
            st.bar_chart(data=pub_rev, x='publisher', y='gross_sale', color="#D2B48C")
        with col_g2:
            st.write("**Units Sold by Publisher**")
            # Perbaikan: Mengambil kolom 'units_sold' dengan benar
            pub_uni = df.groupby('publisher')['units_sold'].sum().nlargest(5).reset_index()
            st.bar_chart(data=pub_uni, x='publisher', y='units_sold', color="#8B4513")
    
    with t2:
        st.subheader("Monthly Sales Trend")
        monthly_trend = df.groupby('bulan_tahun')['gross_sale'].sum().reset_index()
        st.area_chart(data=monthly_trend.set_index('bulan_tahun'), color="#A0522D")
    
    with t3:
        st.subheader("Rating vs Market Popularity")
        pop_data = df.groupby('genre').agg({'book_average_rating': 'mean', 'units_sold': 'sum'}).reset_index()
        st.area_chart(data=pop_data.set_index('genre'), color=["#5D4037", "#D2B48C"])
        st.caption("Coklat Tua: Rating | Coklat Muda: Units Sold")

    with t4:
        st.subheader("Top Performing Authors")
        if 'author' in df.columns:
            author_data = df.groupby('author')['units_sold'].sum().nlargest(10).reset_index()
            st.bar_chart(data=author_data, x='author', y='units_sold', color="#5D4037")

st.divider()

# =========================================================
# BAGIAN 3: DATABASE & TAMBAH DATA
# =========================================================
st.title("📁 Database Management")
tab_view, tab_add = st.tabs(["🗂️ View Table", "➕ Add Record"])

with tab_view:
    show_data = st.checkbox("Show Database Table", value=False)
    if show_data:
        st.dataframe(df_raw, use_container_width=True)

with tab_add:
    with st.form("add_form"):
        c1, c2 = st.columns(2)
        with c1:
            nt = st.text_input("Judul Buku")
            na = st.text_input("Penulis (Author)")
            ng = st.selectbox("Genre", sorted(df_raw['genre'].unique()) if not df_raw.empty else ["Umum"])
            np = st.text_input("Publisher")
        with c2:
            nu = st.number_input("Units Sold", min_value=0)
            nr = st.number_input("Rating", 0.0, 5.0)
            ns = st.number_input("Gross Sale", min_value=0)
            ntgl = st.date_input("Tanggal Transaksi")
        
        if st.form_submit_button("Simpan Data"):
            supabase.table("bacakuy_sales").insert({
                "book_title": nt, "author": na, "genre": ng, "publisher": np,
                "units_sold": nu, "book_average_rating": nr, "gross_sale": ns,
                "tanggal_transaksi": str(ntgl)
            }).execute()
            st.success("Data Tersimpan!")
            st.cache_data.clear()
