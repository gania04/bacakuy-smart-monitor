import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# --- KONFIGURASI ---
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." # Gunakan Key Anda
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_data
def load_data_from_supabase():
    # Menggunakan nama tabel sesuai skema Anda
    res = supabase.table("bacakuy - sales").select("*").execute()
    df = pd.DataFrame(res.data)
    
    if not df.empty:
        # Konversi kolom TEXT ke NUMERIC (Penting karena skema Anda TEXT)
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
        df['book_average_rating'] = pd.to_numeric(df['book_average_rating'], errors='coerce')
        df['gross_sale'] = pd.to_numeric(df['gross_sale'], errors='coerce')
        df = df.dropna(subset=['units_sold', 'book_average_rating', 'gross_sale'])
    return df

# --- UI DASHBOARD ---
st.set_page_config(page_title="Bacakuy Intelligence", layout="wide")
st.title("🚀 Strategic Intelligence Hub")

df_clean = load_data_from_supabase()

if not df_clean.empty:
    # FILTER GENRE
    genres = df_clean['genre'].unique().tolist()
    sel_genre = st.sidebar.selectbox("Pilih Genre", ["Semua"] + genres)
    
    # FILTER DATA
    df_final = df_clean if sel_genre == "Semua" else df_clean[df_clean['genre'] == sel_genre]

    # METRIC CARDS (Seperti Gambar Google Studio)
    c1, c2, c3 = st.columns(3)
    c1.metric("Market Valuation", f"Rp {df_final['gross_sale'].sum():,.0f}")
    c2.metric("Circulation", f"{df_final['units_sold'].sum():,.0f} Units")
    c3.metric("Brand Loyalty", f"{df_final['book_average_rating'].mean():.2f}/5")

    # PREDIKSI & AI
    st.divider()
    col_in, col_res = st.columns([1, 2])
    
    with col_in:
        st.subheader("Prediction Input")
        in_u = st.number_input("Target Terjual", value=100)
        in_r = st.slider("Target Rating", 1.0, 5.0, 4.5)
        
        if st.button("Generate AI Insight", use_container_width=True):
            # ML
            model = LinearRegression().fit(df_final[['units_sold', 'book_average_rating']], df_final['gross_sale'])
            prediction = model.predict([[in_u, in_r]])[0]
            
            with col_res:
                st.subheader("AI Analysis Result")
                st.success(f"Estimasi Gross Profit: Rp {prediction:,.0f}")
                
                # GEMINI
                model_ai = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"Beri saran bisnis Islami untuk menjual {in_u} buku {sel_genre} dengan profit Rp {prediction}."
                st.info(model_ai.generate_content(prompt).text)

    # GRAFIK TREN
    st.subheader("Sales Performance Trend")
    st.area_chart(df_final['gross_sale'])
else:
    st.error("Data di Supabase masih kosong atau tipe data tidak sesuai.")
