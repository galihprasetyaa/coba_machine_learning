import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Deteksi Anomali Aktivasi Lengkap", layout="wide")

st.title("📡 Deteksi Anomali Aktivasi & IMEI dengan Visualisasi")

uploaded_file = st.file_uploader("📁 Upload file CSV data aktivasi", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Aktivasi")
    st.dataframe(df.head())

    # --- Anomali IMEI ---
# Hitung jumlah kemunculan IMEI
imei_counts = df["IMEI"].dropna().value_counts()

# Tandai IMEI duplikat
def mark_duplicate(imei):
    if pd.isna(imei):
        return "Unknown"
    return "Duplicate" if imei_counts.get(imei, 0) > 1 else "Unique"

    df["IMEI_Duplicate"] = df["IMEI"].apply(mark_duplicate)


    # --- Anomali waktu ---
    df["AssignHour"] = pd.to_datetime(df["AssignTime"], errors='coerce').dt.hour.fillna(-1)
    df["ActivationHour"] = pd.to_datetime(df["Activation Time"], errors='coerce').dt.hour.fillna(-1)

    features = ["AssignHour", "ActivationHour", "Qty", "Duration", "Provider", "SKU"]
    df_features = df[features].copy()

    le = LabelEncoder()
    for col in ["Provider", "SKU"]:
        df_features[col] = le.fit_transform(df_features[col].astype(str))

    model = IsolationForest(contamination=0.1, random_state=42)
    df["Time_Anomaly"] = model.fit_predict(df_features)
    df["Time_Anomaly_Label"] = df["Time_Anomaly"].map({1: "✅ Normal", -1: "🚨 Anomali"})

    # --- Gabungkan anomali IMEI dan waktu ---
    df["IMEI_Anomaly_Flag"] = (
        (df["IMEI_Duplicate"] == "Duplicate") |
        (df["IMEI_Valid"] == "Invalid") |
        (df["IMEI_Brand_Inconsistent"] == "Inconsistent")
    )
    df["Overall_Anomaly"] = df["IMEI_Anomaly_Flag"] | (df["Time_Anomaly"] == -1)
    df["Overall_Anomaly_Label"] = df["Overall_Anomaly"].map({True: "🚨 Anomali", False: "✅ Normal"})

    # --- Filter ---
    st.sidebar.header("Filter Data")
    providers = st.sidebar.multiselect("Pilih Provider", options=df["Provider"].unique(), default=df["Provider"].unique())
    skus = st.sidebar.multiselect("Pilih SKU", options=df["SKU"].unique(), default=df["SKU"].unique())
    anomaly_filter = st.sidebar.multiselect("Filter Anomali", options=["🚨 Anomali", "✅ Normal"], default=["🚨 Anomali", "✅ Normal"])

    filtered_df = df[
        (df["Provider"].isin(providers)) &
        (df["SKU"].isin(skus)) &
        (df["Overall_Anomaly_Label"].isin(anomaly_filter))
    ]

    st.subheader(f"Data setelah filter ({len(filtered_df)} baris)")
    st.dataframe(filtered_df[[
        "Assignment ID", "Order ID", "IMEI", "Provider", "SKU", "Qty", "Duration",
        "IMEI_Duplicate", "IMEI_Valid", "IMEI_Brand_Inconsistent",
        "AssignHour", "ActivationHour",
        "Time_Anomaly_Label", "Overall_Anomaly_Label"
    ]])

    # --- Visualisasi ---
    st.subheader("Visualisasi Anomali")

    anomaly_counts = filtered_df["Overall_Anomaly_Label"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(anomaly_counts.index, anomaly_counts.values, color=["green", "red"])
    ax.set_title("Jumlah Data Normal vs Anomali")
    ax.set_ylabel("Jumlah Baris")
    st.pyplot(fig)

    # Grafik per Provider & SKU
    st.subheader("Distribusi Anomali per Provider dan SKU")
    pivot = filtered_df.pivot_table(index="Provider", columns="SKU", values="Overall_Anomaly", aggfunc='sum').fillna(0)
    st.dataframe(pivot.astype(int))

    # Download hasil
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Data Filter & Hasil Anomali (CSV)", data=csv, file_name="anomaly_filtered_results.csv", mime="text/csv")
