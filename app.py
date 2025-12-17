"""
LBSK Forecasting System - Streamlit App
Hanya menggunakan data dari CSV upload user (tanpa data training/hardcoded)
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="LBSK Forecasting System",
    page_icon="🚀",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .upload-section {
        background: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #667eea;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    try:
        peserta_model = joblib.load('xgb_peserta_optuna_best__1_.joblib')
        revenue_model = joblib.load('xgb_revenue_optuna_best__1_.joblib')
        scaler_peserta = joblib.load('scaler_peserta.joblib')
        scaler_revenue = joblib.load('scaler_revenue.joblib')
        return peserta_model, revenue_model, scaler_peserta, scaler_revenue, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False

# ============================================================================
# FEATURE NAMES
# ============================================================================
REQUIRED_FEATURES = [
    'Avg_Harga',
    'Total_Referrals',
    'Jumlah_Peserta_roll_max3',
    'Jumlah_Peserta_roll_max6',
    'Total_Revenue_roll_max3',
    'Total_Revenue_roll_max6',
    'Revenue_per_User',
    'Completion_Revenue_Interaction'
]

EXOGEN_FEATURES = [
    'Avg_Harga',
    'Total_Referrals',
    'Completion_Revenue_Interaction'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def extrapolate_next(series):
    if len(series) < 2:
        return series.iloc[-1] if not series.empty else 0
    x = np.arange(len(series))
    y = series.values
    p = np.polyfit(x, y, 1)
    return p[0] * len(series) + p[1]

def generate_future_months(last_date, n_months=6):
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date + '-01')
    future_dates = []
    for i in range(1, n_months + 1):
        future_date = last_date + relativedelta(months=i)
        future_dates.append(future_date.strftime('%Y-%m'))
    return future_dates

def generate_forecast(df, peserta_model, revenue_model, scaler_peserta, scaler_revenue, n_months=6, target='revenue'):
    extended_df = df.copy()
    extended_df['Date'] = pd.to_datetime(extended_df['Date'])
    future_dates = generate_future_months(extended_df['Date'].iloc[-1], n_months)
    forecast_peserta = []
    forecast_revenue = []
    
    for i in range(n_months):
        new_row = {'Date': pd.to_datetime(future_dates[i] + '-01')}
        
        # Extrapolasi fitur eksogen
        for feat in EXOGEN_FEATURES:
            new_row[feat] = extrapolate_next(extended_df[feat])
        
        # Hitung rolling features dari data historis
        last_peserta = extended_df['Jumlah_Peserta'].tail(6).values
        last_revenue = extended_df['Total_Revenue'].tail(6).values
        
        new_row['Jumlah_Peserta_roll_max3'] = np.max(last_peserta[-3:]) if len(last_peserta) >= 3 else np.max(last_peserta) if len(last_peserta) > 0 else 0
        new_row['Jumlah_Peserta_roll_max6'] = np.max(last_peserta) if len(last_peserta) >= 6 else np.max(last_peserta) if len(last_peserta) > 0 else 0
        new_row['Total_Revenue_roll_max3'] = np.max(last_revenue[-3:]) if len(last_revenue) >= 3 else np.max(last_revenue) if len(last_revenue) > 0 else 0
        new_row['Total_Revenue_roll_max6'] = np.max(last_revenue) if len(last_revenue) >= 6 else np.max(last_revenue) if len(last_revenue) > 0 else 0
        
        last_rev = extended_df['Total_Revenue'].iloc[-1]
        last_pes = extended_df['Jumlah_Peserta'].iloc[-1]
        new_row['Revenue_per_User'] = last_rev / last_pes if last_pes != 0 else 0
        
        # Siapkan feature vector
        features = [new_row.get(f, 0) for f in REQUIRED_FEATURES]
        features_df = pd.DataFrame([features], columns=REQUIRED_FEATURES)
        
        # Prediksi
        pred_peserta = peserta_model.predict(scaler_peserta.transform(features_df))[0]
        pred_revenue = revenue_model.predict(scaler_revenue.transform(features_df))[0]
        
        new_row['Jumlah_Peserta'] = pred_peserta
        new_row['Total_Revenue'] = pred_revenue
        
        forecast_peserta.append(pred_peserta)
        forecast_revenue.append(pred_revenue)
        
        # Tambahkan ke extended_df untuk iterasi berikutnya
        extended_df = pd.concat([extended_df, pd.DataFrame([new_row])], ignore_index=True)
    
    if target == 'revenue':
        return forecast_revenue, future_dates
    else:
        return forecast_peserta, future_dates

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🚀 LBSK Forecasting")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🏠 Home", "💰 Revenue Forecast", "👥 Peserta Forecast"])
st.sidebar.markdown("---")
st.sidebar.info("""
**Model Performance (dari validasi):**
- Peserta: R² 0.9276 | MAPE 1.78%
- Revenue: R² 0.9017 | MAPE 3.11%
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Format CSV yang dibutuhkan:**\n- Date (YYYY-MM)\n- Jumlah_Peserta\n- Total_Revenue\n- 8 kolom fitur (lihat sample)")

# ============================================================================
# SAMPLE DATA UNTUK DOWNLOAD
# ============================================================================
def get_sample_df():
    return pd.DataFrame({
        'Date': ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05'],
        'Jumlah_Peserta': [800, 820, 850, 880, 900],
        'Total_Revenue': [1200000000, 1250000000, 1300000000, 1350000000, 1400000000],
        'Avg_Harga': [1000000, 1050000, 1100000, 1150000, 1200000],
        'Total_Referrals': [50, 55, 60, 65, 70],
        'Jumlah_Peserta_roll_max3': [800, 820, 850, 880, 900],
        'Jumlah_Peserta_roll_max6': [850, 870, 900, 920, 950],
        'Total_Revenue_roll_max3': [1200000000, 1250000000, 1300000000, 1350000000, 1400000000],
        'Total_Revenue_roll_max6': [1250000000, 1300000000, 1350000000, 1400000000, 1450000000],
        'Revenue_per_User': [1500000, 1524000, 1529000, 1534000, 1556000],
        'Completion_Revenue_Interaction': [0.85, 0.86, 0.87, 0.88, 0.89]
    })

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">LBSK Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload CSV untuk melihat historical data & prediksi 6 bulan ke depan</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👥 Model Peserta")
        st.markdown("- R²: 0.9276\n- MAPE: 1.78%")
        st.success("✅ Excellent performance")
    with col2:
        st.markdown("#### 💰 Model Revenue")
        st.markdown("- R²: 0.9017\n- MAPE: 3.11%")
        st.success("✅ Strong performance")
    
    st.markdown("---")
    st.markdown("### 📥 Contoh Format CSV")
    sample_df = get_sample_df()
    st.dataframe(sample_df, use_container_width=True)
    
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample CSV", data=csv, file_name="sample_labskill_data.csv", mime="text/csv")

# ============================================================================
# REVENUE & PESERTA PAGES (sama logikanya, hanya beda target)
# ============================================================================
def forecast_page(target_name, target_col, icon):
    st.markdown(f'<h1 class="main-header">{icon} {target_name} Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload CSV untuk melihat data historis dan prediksi</p>', unsafe_allow_html=True)
    
    peserta_model, revenue_model, scaler_peserta, scaler_revenue, models_loaded = load_models()
    if not models_loaded:
        st.stop()
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload CSV Data Terbaru")
    
    uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])
    
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required = ['Date', 'Jumlah_Peserta', 'Total_Revenue'] + REQUIRED_FEATURES
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Kolom yang hilang: {', '.join(missing)}")
                df = None
            else:
                df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(x + '-01') if len(str(x)) == 7 else pd.to_datetime(x))
                df = df.sort_values('Date').reset_index(drop=True)
                st.success(f"✅ Berhasil upload {len(df)} baris data")
                with st.expander("Preview Data"):
                    st.dataframe(df.tail(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
            df = None
    else:
        st.info("Upload CSV untuk melihat grafik dan prediksi")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if df is None:
        st.stop()  # Tidak lanjut jika belum upload
    
    # Generate forecast
    if target_name == "Revenue":
        forecast_values, future_months = generate_forecast(df, peserta_model, revenue_model, scaler_peserta, scaler_revenue, target='revenue')
    else:
        forecast_values, future_months = generate_forecast(df, peserta_model, revenue_model, scaler_peserta, scaler_revenue, target='peserta')
        forecast_values = [int(round(v)) for v in forecast_values]
    
    actual_dates = df['Date'].dt.strftime('%Y-%m').tolist()
    actual_values = df[target_col].tolist()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Terakhir", f"{actual_values[-1]:,}" if target_col == 'Jumlah_Peserta' else f"IDR {actual_values[-1]:,.0f}")
    with col2:
        st.metric("Rata-rata Forecast", f"{int(np.mean(forecast_values)):,}" if target_col == 'Jumlah_Peserta' else f"IDR {np.mean(forecast_values):,.0f}")
    with col3:
        growth = (np.mean(forecast_values) - actual_values[-1]) / actual_values[-1] * 100
        st.metric("Pertumbuhan Prediksi", f"{growth:.1f}%")
    with col4:
        st.metric("Jumlah Data Historis", f"{len(actual_dates)} bulan")
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual_dates,
        y=actual_values,
        mode='lines+markers',
        name='Aktual',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    # Forecast
    blend_x = [actual_dates[-1], future_months[0]]
    blend_y = [actual_values[-1], forecast_values[0]]
    fig.add_trace(go.Scatter(x=blend_x, y=blend_y, mode='lines', line=dict(color='#ff7f0e', dash='dot'), showlegend=False, hoverinfo='skip'))
    
    fig.add_trace(go.Scatter(
        x=future_months,
        y=forecast_values,
        mode='lines+markers',
        name='Prediksi',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=7, symbol='square')
    ))
    
    # Confidence band
    factor = 1.08 if target_name == "Revenue" else 1.05
    upper = [v * factor for v in forecast_values]
    lower = [v / factor for v in forecast_values]
    fig.add_trace(go.Scatter(
        x=future_months + future_months[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(255,127,14,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f"{target_name}: Data Historis + Prediksi 6 Bulan",
        xaxis_title="Bulan",
        yaxis_title="Jumlah Peserta" if target_name == "Peserta" else "Revenue (IDR)",
        height=600,
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabel prediksi
    st.markdown("### 📋 Detail Prediksi")
    forecast_df = pd.DataFrame({
        'Bulan': future_months,
        f'Prediksi {target_name}': [int(round(v)) for v in forecast_values],
        'Batas Bawah': [int(round(v / factor)) for v in forecast_values],
        'Batas Atas': [int(round(v * factor)) for v in forecast_values]
    })
    st.dataframe(forecast_df.style.format('{:,}'), use_container_width=True, hide_index=True)
    
    csv_out = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        f"📥 Download Hasil Prediksi {target_name}",
        data=csv_out,
        file_name=f"{target_name.lower()}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ============================================================================
# ROUTING
# ============================================================================
if page == "💰 Revenue Forecast":
    forecast_page("Revenue", "Total_Revenue", "💰")
elif page == "👥 Peserta Forecast":
    forecast_page("Peserta", "Jumlah_Peserta", "👥")
