import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================
# Load Models
# ============================

@st.cache_resource
def load_models():
    artifact_p = joblib.load("858a0792-922b-48a6-ae6c-6a193e16959f.joblib")
    artifact_r = joblib.load("a6d64db9-3902-4f1c-b5ec-57fbc545fc89.joblib")

    return artifact_p, artifact_r

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

df_hist = pd.read_excel("2ac25940-83c4-497c-8d68-6678dbf6c89f.xlsx")

# --- penting ---
# lakukan preprocessing yang sama seperti di notebook training kamu
# misalnya: pembentukan Month, Quarter, Lag, Rolling, Seasonal, dll.

def generate_future_features(df, n_months, diskon, rating, referrals, harga):
    # TODO: isi dengan pipeline feature engineering kamu
    # contoh placeholder:
    future = []

    last_row = df.iloc[-1].copy()

    for i in range(n_months):
        new = last_row.copy()

        # update manual input
        new["Diskon"] = diskon
        new["Rating_Diberikan"] = rating
        new["Total_Referrals"] = referrals
        new["Harga_Program"] = harga

        # update tanggal
        new["Month"] += 1
        if new["Month"] > 12:
            new["Month"] = 1
            new["Year"] += 1

        future.append(new)
        last_row = new

    future_df = pd.DataFrame(future)
    return future_df


# ============================
# STREAMLIT UI
# ============================

st.title("LabSkill Multi-Scenario Forecast Dashboard")

uploaded = st.file_uploader("Upload file skenario (forecast_input.xlsx)", type=["xlsx"])

if uploaded:
    sk_df = pd.read_excel(uploaded)

    # deteksi single atau multi scenario
    if "Skenario" in sk_df.columns:
        st.success("Multi-skenario terdeteksi")
        multi = True
    else:
        st.success("Single-skenario terdeteksi")
        multi = False

    results = []

    for idx, row in sk_df.iterrows():

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

        # peserta
        Xp = future_df[features_p]
        Xp_scaled = scaler_p.transform(Xp)
        y_p = model_p.predict(Xp_scaled)

        # revenue
        Xr = future_df[features_r]
        Xr_scaled = scaler_r.transform(Xr)
        y_r = model_r.predict(Xr_scaled)

        res = future_df.copy()
        res["Pred_Peserta"] = y_p
        res["Pred_Revenue"] = y_r

        if multi:
            res["Skenario"] = row["Skenario"]

        results.append(res)

    final_df = pd.concat(results, ignore_index=True)

    st.subheader("Hasil Forecast")
    st.dataframe(final_df)

    # grafik jika single skenario
    if not multi:
        st.line_chart(final_df[["Pred_Peserta", "Pred_Revenue"]])
