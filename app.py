import streamlit as st
import google.generativeai as genai
from supabase import create_client

# --- 1. AMBIL KREDENSIAL DARI SECRETS ---
# Pastikan Anda sudah mengisi 'Secrets' di Settings Streamlit dengan benar
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    
    # Inisialisasi Service
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    genai.configure(api_key=GEMINI_API_KEY)
    
    # PERBAIKAN UTAMA: Menggunakan model dasar untuk menghindari error v1beta
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Gagal Inisialisasi: {e}")

# ... (Kode load_data tetap sama seperti sebelumnya) ...

# =========================================================
# BAGIAN ANALISIS AI (UPDATE PROSESNYA)
# =========================================================
# Gunakan blok ini di dalam bagian col_res Anda
with st.spinner("Sedang menghubungkan ke server AI..."):
    try:
        # Menambahkan konfigurasi keamanan agar tidak diblokir
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        }
        
        prompt = f"Berikan 1 strategi bisnis syariah singkat untuk profit Rp {prediction:,.0f}."
        
        # Eksekusi AI
        response = model_ai.generate_content(prompt, generation_config=generation_config)
        
        if response.text:
            st.success(response.text)
    except Exception as e:
        # Jika masih 404, coba gunakan model alternatif 'gemini-pro'
        try:
            alt_model = genai.GenerativeModel('gemini-pro')
            response = alt_model.generate_content(prompt)
            st.info(response.text)
        except:
            st.error(f"Akses AI Terblokir (404).")
            st.warning("Solusi Terakhir: Hapus aplikasi ini dari Dashboard Streamlit dan Deploy ulang dari awal.")
