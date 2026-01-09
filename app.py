import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
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
    .confidence-box {
        padding: 15px; border-radius: 12px; text-align: center; font-weight: bold;
        border: 2px solid; margin-top: 10px; font-size: 18px;
    }
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
# BAGIAN 1: PREDIKSI & MARKET CONFIDENCE
# =========================================================
st.title("📑 Bacakuy Intelligence: Sales & Confidence")
col_p1, col_p2 = st.columns([1, 2])

with col_p1:
    st.subheader("🔍 Predictor Input")
    available_genres = sorted(df_raw['genre'].unique()) if not df_raw.empty else ["Fiction"]
    in_g = st.selectbox("Pilih Genre:", available_genres)
    in_u = st.number_input("Target Unit Terjual", value=100, min_value=1)
    in_r = st.slider("Asumsi Rating Pelanggan", 0.0, 5.0, 4.2)
    btn_predict = st.button("Jalankan Analisa")

with col_p2:
    if btn_predict and not df_raw.empty:
        # PREDIKSI: Murni Genre & Unit (Tanpa Rating di Rumus)
        le = LabelEncoder()
        df_ml = df_raw.copy()
        df_ml['genre_enc'] = le.fit_transform(df_ml['genre'])
        
        X = df_ml[['genre_enc', 'units_sold']] 
        y = df_ml['gross_sale']
        model = LinearRegression().fit(X, y)
        
        try:
            g_val = le.transform([in_g])[0]
            prediction = model.predict([[g_val, in_u]])[0]
            prediction = max(0, prediction)
        except:
            prediction = 0
        
        # LOGIKA MARKET CONFIDENCE (Berdasarkan Rating)
        if in_r >= 4.5:
            c_lab, c_col, c_txt = "EXCELLENT", "#2E7D32", "High trust: Pasar sangat loyal."
        elif in_r >= 3.5:
            c_lab, c_col, c_txt = "GOOD", "#F9A825", "Stable: Kualitas memenuhi standar."
        else:
            c_lab, c_col, c_txt = "AT RISK", "#C62828", "Critical: Resiko churn tinggi."

        cp1, cp2 = st.columns(2)
        cp1.metric(f"Estimasi Gross Sales (${in_g})", f"$ {prediction:,.2f}")
        with cp2:
            st.markdown(f"""
                <div class="confidence-box" style="background-color: {c_col}22; border-color: {c_col}; color: {c_col};">
                    {c_lab} CONFIDENCE
                </div>
            """, unsafe_allow_html=True)
            st.caption(f"_{c_txt}_")

        try:
            resp = model_ai.generate_content(f"Berikan strategi marketing syariah untuk genre {in_g} dengan target ${prediction:,.2f} dan status pasar {c_lab}.")
            st.success(resp.text)
        except:
            st.warning("Gagal memuat AI Insight.")

st.divider()

# =========================================================
# BAGIAN 2: STRATEGIC HUB (SEMUA FITUR TETAP)
# =========================================================
st.title("🚀 Strategic Intelligence Hub")

if not df_raw.empty:
    f1, f2 = st.columns(2)
    with f1:
        sel_genre = st.selectbox("Filter Genre:", ["Semua Genre"] + sorted(list(df_raw['genre'].unique())))
    with f2:
        list_bulan = df_raw['bulan_tahun'].unique().tolist()
        sel_month = st.selectbox("Filter Bulan:", ["Semua Bulan"] + list_bulan)

    df = df_raw.copy()
    if sel_genre != "Semua Genre": df = df[df['genre'] == sel_genre]
    if sel_month != "Semua Bulan": df = df[df['bulan_tahun'] == sel_month]

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Market Valuation", f"$ {df['gross_sale'].sum():,.2f}")
    k2.metric("Circulation", f"{df['units_sold'].sum():,.0f} Units")
    k3.metric("Profitability Index", "45.1%", "Rev/Gross")
    k4.metric("Brand Loyalty", f"{df['book_average_rating'].mean():.2f}/5")

    # TABS GRAFIK
    t1, t2, t3, t4 = st.tabs(["📊 Performance", "📈 Monthly Trend", "🎯 Popularity", "✍️ Author"])
    
    with t1:
        st.subheader("Publisher Performance")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.write("**Revenue by Publisher ($)**")
            pub_rev = df.groupby('publisher')['gross_sale'].sum().nlargest(5).reset_index()
            st.bar_chart(data=pub_rev, x='publisher', y='gross_sale', color="#D2B48C")
        with col_g2:
            st.write("**Units Sold by Publisher**")
            pub_uni = df.groupby('publisher')['units_sold'].sum().nlargest(5).reset_index()
            st.bar_chart(data=pub_uni, x='publisher', y='units_sold', color="#8B4513")
    
    with t2:
        st.subheader("Monthly Sales Trend ($)")
        monthly_trend = df.groupby('bulan_tahun')['gross_sale'].sum().reset_index()
        st.area_chart(data=monthly_trend.set_index('bulan_tahun'), color="#A0522D")
    
    with t3:
        st.subheader("Rating vs Market Popularity")
        pop_data = df.groupby('genre').agg({'book_average_rating': 'mean', 'units_sold': 'sum'}).reset_index()
        st.area_chart(data=pop_data.set_index('genre'), color=["#5D4037", "#D2B48C"])

    with t4:
        st.subheader("Top Performing Authors")
        if 'author' in df.columns:
            author_data = df.groupby('author')['units_sold'].sum().nlargest(10).reset_index()
            st.bar_chart(data=author_data, x='author', y='units_sold', color="#5D4037")

st.divider()

# =========================================================
# BAGIAN 3: DATABASE
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
            na = st.text_input("Author")
            ng = st.selectbox("Genre", sorted(df_raw['genre'].unique()) if not df_raw.empty else ["Fiction"])
            np = st.text_input("Publisher")
        with c2:
            nu = st.number_input("Units Sold", min_value=0)
            nr = st.number_input("Rating", 0.0, 5.0)
            ns = st.number_input("Gross Sale (USD)", min_value=0.0)
            ntgl = st.date_input("Tanggal Transaksi")
        
        if st.form_submit_button("Simpan Data"):
            supabase.table("bacakuy_sales").insert({
                "book_title": nt, "author": na, "genre": ng, "publisher": np,
                "units_sold": nu, "book_average_rating": nr, "gross_sale": ns,
                "tanggal_transaksi": str(ntgl)
            }).execute()
            st.success("Data Tersimpan!")
            st.cache_data.clear()
