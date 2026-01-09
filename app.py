import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
from supabase import create_client

# Konfigurasi (Gunakan URL lengkap)
SUPABASE_URL = "https://oftpulsqxjhhtfukmmtr.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." # Key Anda
GEMINI_API_KEY = "AIzaSyApzYuBJ0QWbw6QXd75X9CYjo_E6_fZHoE"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_data
def load_data():
    # MEMANGGIL NAMA TABEL: bacakuy sales
    res = supabase.table("bacakuy sales").select("*").execute()
    return pd.DataFrame(res.data)

df = load_data()

# --- Bagian Filter Genre ---
if not df.empty:
    st.sidebar.header("Filter Genre & Prediksi")
    genres = df['genre'].unique().tolist()
    selected_genre = st.sidebar.selectbox("Pilih Genre", ["Semua"] + genres)
    
    # Logika Filter
    df_filtered = df if selected_genre == "Semua" else df[df['genre'] == selected_genre]
    
    # Input Prediksi
    target_unit = st.sidebar.number_input("Target Unit Terjual", value=100)
    target_rating = st.sidebar.slider("Target Rating", 1.0, 5.0, 4.5)
    
    if st.sidebar.button("Hitung & Beri Insight"):
        # Model ML
        model = LinearRegression().fit(df_filtered[['units_sold', 'book_average_rating']], df_filtered['gross_sales'])
        pred = model.predict([[target_unit, target_rating]])[0]
        
        st.metric(f"Estimasi Profit ({selected_genre})", f"Rp {pred:,.0f}")
        
        # Insight AI
        model_ai = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Berikan 2 strategi bisnis untuk menjual {target_unit} buku genre {selected_genre} dengan potensi profit Rp {pred}."
        st.info(model_ai.generate_content(prompt).text)
