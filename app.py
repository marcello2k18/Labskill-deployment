import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# ============================
# Load Models with Error Handling
# ============================
@st.cache_resource
def load_models():
    try:
        artifact_p = joblib.load("858a0792-922b-48a6-ae6c-6a193e16959f.joblib")
        artifact_r = joblib.load("a6d64db9-3902-4f1c-b5ec-57fbc545fc89.joblib")
        return artifact_p, artifact_r
    except FileNotFoundError as e:
        st.error(f"Model file tidak ditemukan: {e}")
        st.stop()

artifact_p, artifact_r = load_models()
model_p = artifact_p["model"]
scaler_p = artifact_p["scaler"]
features_p = artifact_p["selected_features"]

model_r = artifact_r["model"]
scaler_r = artifact_r["scaler"]
features_r = artifact_r["selected_features"]

# ============================
# Load Historical Dataset
# ============================
try:
    df_hist = pd.read_excel("2ac25940-83c4-497c-8d68-6678dbf6c89f.xlsx")
except FileNotFoundError:
    st.error("File historis tidak ditemukan")
    st.stop()

def generate_future_features(df, n_months, diskon, rating, referrals, harga):
    """
    HARUS DIISI: Implement feature engineering yang SAMA dengan training
    Contoh features yang mungkin dibutuhkan:
    - Lag_Peserta_1, Lag_Peserta_3
    - Rolling_Mean_3, Rolling_Std_3
    - Quarter, Is_Holiday
    - Interaction features
    """
    future = []
    
    # Get last known values untuk lag features
    last_date = pd.to_datetime(df['Tanggal'].iloc[-1]) if 'Tanggal' in df.columns else None
    
    for i in range(n_months):
        new_row = {}
        
        # Basic inputs
        new_row["Diskon"] = diskon
        new_row["Rating_Diberikan"] = rating
        new_row["Total_Referrals"] = referrals
        new_row["Harga_Program"] = harga
        
        # Time features
        if last_date:
            current_date = last_date + timedelta(days=30*(i+1))
            new_row["Month"] = current_date.month
            new_row["Year"] = current_date.year
            new_row["Quarter"] = (current_date.month - 1) // 3 + 1
        
        # TODO: Tambahkan lag features dari df + future sebelumnya
        # TODO: Tambahkan rolling features
        # TODO: Tambahkan seasonal features
        
        future.append(new_row)
    
    future_df = pd.DataFrame(future)
    
    # Validasi bahwa semua required features ada
    missing_p = set(features_p) - set(future_df.columns)
    missing_r = set(features_r) - set(future_df.columns)
    
    if missing_p or missing_r:
        st.error(f"Missing features untuk peserta: {missing_p}")
        st.error(f"Missing features untuk revenue: {missing_r}")
        st.stop()
    
    return future_df

# ============================
# STREAMLIT UI
# ============================
st.title("🎓 LabSkill Multi-Scenario Forecast Dashboard")

uploaded = st.file_uploader("Upload file skenario (forecast_input.xlsx)", type=["xlsx"])

if uploaded:
    try:
        sk_df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        st.stop()
    
    # Validasi kolom required
    required_cols = ["Diskon", "Rating", "Total_Referrals", "Harga_Program", "Forecast_Months"]
    missing_cols = set(required_cols) - set(sk_df.columns)
    if missing_cols:
        st.error(f"Kolom yang hilang: {missing_cols}")
        st.stop()
    
    # Deteksi single atau multi scenario
    multi = "Skenario" in sk_df.columns
    st.success(f"{'Multi' if multi else 'Single'}-skenario terdeteksi ({len(sk_df)} baris)")
    
    results = []
    
    for idx, row in sk_df.iterrows():
        with st.spinner(f"Processing scenario {idx+1}/{len(sk_df)}..."):
            diskon = row["Diskon"]
            rating = row["Rating"]
            referrals = row["Total_Referrals"]
            harga = row["Harga_Program"]
            n_months = int(row["Forecast_Months"])
            
            future_df = generate_future_features(
                df_hist,
                n_months=n_months,
                diskon=diskon,
                rating=rating,
                referrals=referrals,
                harga=harga
            )
            
            # Predict peserta
            Xp = future_df[features_p]
            Xp_scaled = scaler_p.transform(Xp)
            y_p = model_p.predict(Xp_scaled)
            
            # Predict revenue
            Xr = future_df[features_r]
            Xr_scaled = scaler_r.transform(Xr)
            y_r = model_r.predict(Xr_scaled)
            
            res = future_df.copy()
            res["Pred_Peserta"] = np.maximum(0, y_p)  # Tidak boleh negatif
            res["Pred_Revenue"] = np.maximum(0, y_r)
            
            if multi:
                res["Skenario"] = row["Skenario"]
            
            results.append(res)
    
    final_df = pd.concat(results, ignore_index=True)
    
    st.subheader("📊 Hasil Forecast")
    st.dataframe(final_df)
    
    # Download button
    st.download_button(
        "💾 Download CSV",
        final_df.to_csv(index=False).encode('utf-8'),
        "forecast_results.csv",
        "text/csv"
    )
    
    # Grafik
    if not multi:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediksi Peserta")
            st.line_chart(final_df["Pred_Peserta"])
        with col2:
            st.subheader("Prediksi Revenue")
            st.line_chart(final_df["Pred_Revenue"])
    else:
        st.subheader("Perbandingan Skenario")
        st.line_chart(final_df.pivot(columns="Skenario", values="Pred_Peserta"))
