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
    
    # Ambil nilai terakhir untuk stabilisasi
    last_avg_harga = extended_df['Avg_Harga'].iloc[-1]
    last_referrals = extended_df['Total_Referrals'].iloc[-1]
    last_completion = extended_df['Completion_Revenue_Interaction'].iloc[-1]
    
    # Hitung growth rate dari 6 bulan terakhir untuk stabilisasi
    recent_6_months = extended_df.tail(6)
    if len(recent_6_months) >= 2:
        avg_harga_growth = (recent_6_months['Avg_Harga'].iloc[-1] - recent_6_months['Avg_Harga'].iloc[0]) / len(recent_6_months)
        referrals_growth = (recent_6_months['Total_Referrals'].iloc[-1] - recent_6_months['Total_Referrals'].iloc[0]) / len(recent_6_months)
        completion_growth = (recent_6_months['Completion_Revenue_Interaction'].iloc[-1] - recent_6_months['Completion_Revenue_Interaction'].iloc[0]) / len(recent_6_months)
    else:
        avg_harga_growth = 0
        referrals_growth = 0
        completion_growth = 0
    
    for i in range(n_months):
        new_row = {'Date': pd.to_datetime(future_dates[i] + '-01')}
        
        # Extrapolasi fitur eksogen dengan pertumbuhan lebih konservatif
        new_row['Avg_Harga'] = last_avg_harga + (avg_harga_growth * (i + 1))
        new_row['Total_Referrals'] = max(0, last_referrals + (referrals_growth * (i + 1)))
        new_row['Completion_Revenue_Interaction'] = np.clip(
            last_completion + (completion_growth * (i + 1)), 
            0, 1
        )
        
        # PERBAIKAN KRITIS: Hitung rolling features dari data + prediksi sebelumnya
        # Ambil 6 bulan terakhir (termasuk prediksi yang sudah dibuat)
        last_peserta = extended_df['Jumlah_Peserta'].tail(6).values
        last_revenue = extended_df['Total_Revenue'].tail(6).values
        
        # Rolling max 3 bulan
        new_row['Jumlah_Peserta_roll_max3'] = np.max(last_peserta[-3:]) if len(last_peserta) >= 3 else np.max(last_peserta)
        new_row['Total_Revenue_roll_max3'] = np.max(last_revenue[-3:]) if len(last_revenue) >= 3 else np.max(last_revenue)
        
        # Rolling max 6 bulan
        new_row['Jumlah_Peserta_roll_max6'] = np.max(last_peserta)
        new_row['Total_Revenue_roll_max6'] = np.max(last_revenue)
        
        # Revenue per User dari data terakhir (lebih stabil)
        recent_revenue_per_user = extended_df['Revenue_per_User'].tail(3).mean()
        new_row['Revenue_per_User'] = recent_revenue_per_user
        
        # Siapkan feature vector
        features = [new_row.get(f, 0) for f in REQUIRED_FEATURES]
        features_df = pd.DataFrame([features], columns=REQUIRED_FEATURES)
        
        # Prediksi dengan constraint: tidak boleh turun drastis
        pred_peserta_raw = peserta_model.predict(scaler_peserta.transform(features_df))[0]
        pred_revenue_raw = revenue_model.predict(scaler_revenue.transform(features_df))[0]
        
        # Constraint: prediksi tidak boleh turun > 20% dari nilai terakhir
        last_peserta_val = extended_df['Jumlah_Peserta'].iloc[-1]
        last_revenue_val = extended_df['Total_Revenue'].iloc[-1]
        
        # Smooth prediction: blend dengan nilai terakhir (80% prediksi + 20% last value untuk stabilitas)
        pred_peserta = pred_peserta_raw * 0.85 + last_peserta_val * 0.15
        pred_revenue = pred_revenue_raw * 0.85 + last_revenue_val * 0.15
        
        # Final constraint: minimal 70% dari nilai terakhir
        pred_peserta = max(pred_peserta, last_peserta_val * 0.7)
        pred_revenue = max(pred_revenue, last_revenue_val * 0.7)
        
        new_row['Jumlah_Peserta'] = pred_peserta
        new_row['Total_Revenue'] = pred_revenue
        
        forecast_peserta.append(pred_peserta)
        forecast_revenue.append(pred_revenue)
        
        # Tambahkan ke extended_df untuk iterasi berikutnya (KRITIS!)
        extended_df = pd.concat([extended_df, pd.DataFrame([new_row])], ignore_index=True)
    
    if target == 'revenue':
        return forecast_revenue, future_dates
    else:
        return forecast_peserta, future_dates

# ============================================================================
# SAMPLE DATA UNTUK DOWNLOAD
# ============================================================================
def get_sample_df():
    """
    Generate sample data: 20 bulan training + 5 bulan testing = 25 bulan total
    Periode: Januari 2023 - Januari 2025
    """
    dates = pd.date_range('2023-01', periods=25, freq='MS')
    date_strings = [d.strftime('%Y-%m') for d in dates]
    
    # Generate realistic progressive data
    base_peserta = 600
    peserta_growth = [base_peserta + i*15 + np.random.randint(-10, 20) for i in range(25)]
    
    base_revenue = 900000000  # 900 juta
    revenue_growth = [base_revenue + i*50000000 + np.random.randint(-10000000, 30000000) for i in range(25)]
    
    avg_harga = [revenue_growth[i] / peserta_growth[i] if peserta_growth[i] > 0 else 1500000 for i in range(25)]
    
    referrals = [30 + i*2 + np.random.randint(-3, 5) for i in range(25)]
    
    data = {
        'Date': date_strings,
        'Jumlah_Peserta': peserta_growth,
        'Total_Revenue': revenue_growth,
        'Avg_Harga': [int(h) for h in avg_harga],
        'Total_Referrals': referrals,
        'Jumlah_Peserta_roll_max3': [0]*25,  # Will be calculated below
        'Jumlah_Peserta_roll_max6': [0]*25,
        'Total_Revenue_roll_max3': [0]*25,
        'Total_Revenue_roll_max6': [0]*25,
        'Revenue_per_User': [int(revenue_growth[i] / peserta_growth[i]) if peserta_growth[i] > 0 else 0 for i in range(25)],
        'Completion_Revenue_Interaction': [0.80 + i*0.003 + np.random.uniform(-0.02, 0.02) for i in range(25)]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate rolling features properly
    for i in range(len(df)):
        if i >= 2:
            df.at[i, 'Jumlah_Peserta_roll_max3'] = df['Jumlah_Peserta'].iloc[max(0, i-2):i+1].max()
        else:
            df.at[i, 'Jumlah_Peserta_roll_max3'] = df['Jumlah_Peserta'].iloc[:i+1].max()
            
        if i >= 5:
            df.at[i, 'Jumlah_Peserta_roll_max6'] = df['Jumlah_Peserta'].iloc[max(0, i-5):i+1].max()
        else:
            df.at[i, 'Jumlah_Peserta_roll_max6'] = df['Jumlah_Peserta'].iloc[:i+1].max()
            
        if i >= 2:
            df.at[i, 'Total_Revenue_roll_max3'] = df['Total_Revenue'].iloc[max(0, i-2):i+1].max()
        else:
            df.at[i, 'Total_Revenue_roll_max3'] = df['Total_Revenue'].iloc[:i+1].max()
            
        if i >= 5:
            df.at[i, 'Total_Revenue_roll_max6'] = df['Total_Revenue'].iloc[max(0, i-5):i+1].max()
        else:
            df.at[i, 'Total_Revenue_roll_max6'] = df['Total_Revenue'].iloc[:i+1].max()
    
    return df

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
    st.markdown("### 📥 Contoh Format CSV (25 bulan: 20 training + 5 testing)")
    st.info("📊 Dataset sample ini mencakup **20 bulan data training** (Jan 2023 - Agu 2024) dan **5 bulan data testing** (Sep 2024 - Jan 2025)")
    
    sample_df = get_sample_df()
    
    # Show first 5, last 5 rows for preview
    st.markdown("**Preview: 5 baris pertama dan 5 baris terakhir**")
    preview_df = pd.concat([sample_df.head(5), sample_df.tail(5)])
    st.dataframe(preview_df, use_container_width=True)
    
    st.markdown(f"**Total: {len(sample_df)} baris data** (Jan 2023 - Jan 2025)")
    
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Sample CSV (25 bulan)", data=csv, file_name="sample_labskill_data_25months.csv", mime="text/csv")

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
    
    # Tabel prediksi - FIXED: Format setiap kolom secara terpisah
    st.markdown("### 📋 Detail Prediksi")
    forecast_df = pd.DataFrame({
        'Bulan': future_months,
        f'Prediksi {target_name}': [int(round(v)) for v in forecast_values],
        'Batas Bawah': [int(round(v / factor)) for v in forecast_values],
        'Batas Atas': [int(round(v * factor)) for v in forecast_values]
    })
    
    # Format hanya kolom numerik
    styled_df = forecast_df.style.format({
        f'Prediksi {target_name}': '{:,}',
        'Batas Bawah': '{:,}',
        'Batas Atas': '{:,}'
    })
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
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
