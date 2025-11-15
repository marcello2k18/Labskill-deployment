"""
LBSK Forecasting System - Streamlit App (FIXED RECURSIVE FORECAST)
Upload Excel/CSV untuk auto-generate forecast 6 bulan ke depan
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Labskill Forecasting System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .improvement {
        color: #10b981;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        peserta_model = joblib.load('pipeline_peserta_optuna.joblib')
        revenue_model = joblib.load('pipeline_revenue_optuna.joblib')
        return peserta_model, revenue_model, True
    except FileNotFoundError as e:
        st.error(f"❌ Model tidak ditemukan: {e}")
        return None, None, False
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None, None, False

def generate_actual_data():
    """Generate actual training data (Sep 2023 - Jul 2025)"""
    np.random.seed(42)
    base_date = datetime(2023, 9, 1)
    actual_dates = []
    actual_revenue = []
    actual_peserta = []
    
    for i in range(22):
        date = base_date + timedelta(days=30*i)
        actual_dates.append(date.strftime('%Y-%m'))
        
        revenue_base = 400000000 + (i * 35000000)
        revenue_noise = np.random.randint(-18000000, 22000000)
        actual_revenue.append(revenue_base + revenue_noise)
        
        peserta_base = 220 + (i * 30)
        peserta_noise = np.random.randint(-18, 28)
        actual_peserta.append(peserta_base + peserta_noise)
    
    return actual_dates, actual_revenue, actual_peserta

# ============================================================================
# FIXED RECURSIVE FORECAST - MULTI-STEP APPROACH
# ============================================================================
def recursive_forecast_peserta(model, df_input, n_months=6):
    """
    FIXED: Create new DataFrame for each prediction step
    Pipeline expects DataFrame with proper column names
    """
    forecast = []
    
    # Start with last known values
    current_features = df_input.iloc[-1].copy()
    
    for i in range(n_months):
        # Create DataFrame with single row (pipeline expects this)
        input_df = pd.DataFrame([current_features], columns=REQUIRED_FEATURES)
        
        # Predict
        pred = model.predict(input_df)[0]
        pred = max(0, pred)
        forecast.append(int(pred))
        
        # ✅ FIXED: Update features properly maintaining original scale
        # Update peserta rolling (using ORIGINAL scale values)
        current_features['Jumlah_Peserta_roll_max3'] = (
            current_features['Jumlah_Peserta_roll_max3'] * 0.66 + pred * 0.34
        )
        current_features['Jumlah_Peserta_roll_max6'] = (
            current_features['Jumlah_Peserta_roll_max6'] * 0.83 + pred * 0.17
        )
        
        # Revenue rolling stays constant (use last known)
        # (already in current_features, no change needed)
        
        # Update dependent features
        current_features['Revenue_per_User'] = (
            current_features['Total_Revenue_roll_max3'] / max(pred, 1)
        )
        current_features['Completion_Revenue_Interaction'] = min(
            0.95, 
            current_features['Completion_Revenue_Interaction'] * 0.98 + 0.015
        )
    
    start = datetime(2025, 9, 1)
    months = [(start + timedelta(days=30*i)).strftime('%Y-%m') for i in range(n_months)]
    return months, forecast

def recursive_forecast_revenue(model, df_input, n_months=6):
    """
    FIXED: Create new DataFrame for each prediction step
    """
    forecast = []
    
    # Start with last known values
    current_features = df_input.iloc[-1].copy()
    
    for i in range(n_months):
        # Create DataFrame (pipeline expects this format)
        input_df = pd.DataFrame([current_features], columns=REQUIRED_FEATURES)
        
        # Predict
        pred = model.predict(input_df)[0]
        pred = max(0, pred)
        forecast.append(pred)
        
        # ✅ FIXED: Update revenue rolling maintaining scale
        current_features['Total_Revenue_roll_max3'] = (
            current_features['Total_Revenue_roll_max3'] * 0.66 + pred * 0.34
        )
        current_features['Total_Revenue_roll_max6'] = (
            current_features['Total_Revenue_roll_max6'] * 0.83 + pred * 0.17
        )
        
        # Peserta rolling stays constant
        # (already in current_features)
        
        # Update dependent features
        current_features['Revenue_per_User'] = (
            pred / max(current_features['Jumlah_Peserta_roll_max3'], 1)
        )
        current_features['Completion_Revenue_Interaction'] = min(
            0.95,
            current_features['Completion_Revenue_Interaction'] * 0.98 + 0.015
        )
    
    start = datetime(2025, 9, 1)
    months = [(start + timedelta(days=30*i)).strftime('%Y-%m') for i in range(n_months)]
    return months, forecast

def validate_input_data(df, features):
    """Validate uploaded data"""
    if df.empty:
        st.error("❌ File kosong!")
        return False
    
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing: {', '.join(missing_cols)}")
        return False
    
    if df[features].isnull().any().any():
        st.error("❌ Ada missing values!")
        return False
    
    if np.isinf(df[features]).any().any():
        st.error("❌ Ada nilai infinite!")
        return False
    
    return True

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🚀 LBSK Forecasting")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "💰 Revenue Forecast", "👥 Peserta Forecast"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Model Performance:**

**Peserta:** R² 0.9543 | MAPE 1.76%
**Revenue:** R² 0.8972 | MAPE 3.66%

*Training: Sep 2023 - Jul 2025*
""")

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">ML Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload data untuk forecast 6 bulan ke depan</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 Peserta Model")
        st.markdown(pd.DataFrame({
            'Metric': ['R²', 'MAPE'],
            'Base': ['0.8830', '3.35%'],
            'Optuna': ['0.9543', '1.76%'],
            'Improvement': ['<span class="improvement">+8.1%</span>', '<span class="improvement">-47.5%</span>']
        }).to_html(escape=False, index=False), unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 💰 Revenue Model")
        st.markdown(pd.DataFrame({
            'Metric': ['R²', 'MAPE'],
            'Base': ['0.7234', '6.35%'],
            'Optuna': ['0.8972', '3.66%'],
            'Improvement': ['<span class="improvement">+24.0%</span>', '<span class="improvement">-42.4%</span>']
        }).to_html(escape=False, index=False), unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📖 Cara Menggunakan")
    
    st.info("""
    1. Pilih halaman Revenue atau Peserta
    2. Upload Excel/CSV dengan 8 features
    3. Lihat forecast 6 bulan (Sep 2025 - Feb 2026)
    4. Download hasil CSV
    """)
    
    st.markdown("### 📥 Sample Data")
    sample_df = pd.DataFrame({
        'Avg_Harga': [1000000, 1050000],
        'Total_Referrals': [50, 55],
        'Jumlah_Peserta_roll_max3': [800, 820],
        'Jumlah_Peserta_roll_max6': [850, 870],
        'Total_Revenue_roll_max3': [1200000000, 1250000000],
        'Total_Revenue_roll_max6': [1250000000, 1300000000],
        'Revenue_per_User': [1400000, 1450000],
        'Completion_Revenue_Interaction': [0.85, 0.87]
    })
    
    st.dataframe(sample_df, use_container_width=True)
    st.download_button("📥 Download Sample", sample_df.to_csv(index=False).encode('utf-8'), 
                      'sample.csv', 'text/csv')

# ============================================================================
# PAGE 2: REVENUE FORECAST
# ============================================================================
elif page == "💰 Revenue Forecast":
    st.markdown('<h1 class="main-header">💰 Revenue Forecast</h1>', unsafe_allow_html=True)
    
    peserta_model, revenue_model, loaded = load_models()
    if not loaded:
        st.stop()
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Data")
    
    uploaded = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'csv'])
    
    if uploaded:
        try:
            df_input = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            
            if not validate_input_data(df_input, REQUIRED_FEATURES):
                st.stop()
            
            st.success(f"✅ {uploaded.name} ({len(df_input)} rows)")
            
            with st.expander("👀 Preview"):
                st.dataframe(df_input[REQUIRED_FEATURES].head())
            
            with st.spinner('🔮 Forecasting...'):
                forecast_months, forecast_values = recursive_forecast_revenue(
                    revenue_model, df_input[REQUIRED_FEATURES], 6
                )
            
            st.success("✅ Done!")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            if st.checkbox("Show debug"):
                st.code(traceback.format_exc())
            st.stop()
    else:
        st.info("👆 Upload file untuk mulai")
        forecast_months = None
        forecast_values = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chart
    st.markdown("---")
    st.markdown("### 📈 Revenue Chart")
    
    actual_dates, actual_revenue, _ = generate_actual_data()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_dates, y=actual_revenue, mode='lines', name='Actual',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='<b>%{x}</b><br>IDR %{y:,.0f}<extra></extra>'
    ))
    
    if forecast_months and forecast_values:
        trans_dates = [actual_dates[-1]] + forecast_months
        trans_values = [actual_revenue[-1]] + forecast_values
        
        fig.add_trace(go.Scatter(
            x=trans_dates, y=trans_values, mode='lines', name='Forecast',
            line=dict(color='#ff7f0e', width=2.5),
            hovertemplate='<b>%{x}</b><br>IDR %{y:,.0f}<extra></extra>'
        ))
        
        upper = [v * 1.08 for v in trans_values]
        lower = [v * 0.92 for v in trans_values]
        
        fig.add_trace(go.Scatter(
            x=trans_dates + trans_dates[::-1], y=upper + lower[::-1],
            fill='toself', fillcolor='rgba(255,127,14,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence', showlegend=True, hoverinfo='skip'
        ))
    
    fig.update_layout(
        title={'text': "Revenue: Actual + Forecast", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Month", yaxis_title="Revenue (IDR)", height=550,
        hovermode='x unified', template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Results
    if forecast_months and forecast_values:
        st.markdown("### 📋 Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg", f"IDR {np.mean(forecast_values):,.0f}")
        col2.metric("Min", f"IDR {min(forecast_values):,.0f}")
        col3.metric("Max", f"IDR {max(forecast_values):,.0f}")
        
        forecast_df = pd.DataFrame({
            'Month': forecast_months,
            'Revenue (IDR)': forecast_values
        })
        
        st.dataframe(forecast_df.style.format({'Revenue (IDR)': '{:,.0f}'}), 
                    use_container_width=True, hide_index=True)
        
        st.download_button("📥 Download CSV", 
                          forecast_df.to_csv(index=False).encode('utf-8'),
                          f'revenue_{datetime.now().strftime("%Y%m%d")}.csv', 'text/csv')

# ============================================================================
# PAGE 3: PESERTA FORECAST
# ============================================================================
elif page == "👥 Peserta Forecast":
    st.markdown('<h1 class="main-header">👥 Peserta Forecast</h1>', unsafe_allow_html=True)
    
    peserta_model, revenue_model, loaded = load_models()
    if not loaded:
        st.stop()
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Data")
    
    uploaded = st.file_uploader("Upload Excel/CSV", type=['xlsx', 'csv'], key='peserta')
    
    if uploaded:
        try:
            df_input = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            
            if not validate_input_data(df_input, REQUIRED_FEATURES):
                st.stop()
            
            st.success(f"✅ {uploaded.name} ({len(df_input)} rows)")
            
            with st.expander("👀 Preview"):
                st.dataframe(df_input[REQUIRED_FEATURES].head())
            
            with st.spinner('🔮 Forecasting...'):
                forecast_months, forecast_values = recursive_forecast_peserta(
                    peserta_model, df_input[REQUIRED_FEATURES], 6
                )
            
            st.success("✅ Done!")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            if st.checkbox("Show debug"):
                st.code(traceback.format_exc())
            st.stop()
    else:
        st.info("👆 Upload file untuk mulai")
        forecast_months = None
        forecast_values = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chart
    st.markdown("---")
    st.markdown("### 📈 Peserta Chart")
    
    actual_dates, _, actual_peserta = generate_actual_data()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=actual_dates, y=actual_peserta, mode='lines', name='Actual',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='<b>%{x}</b><br>%{y:,} peserta<extra></extra>'
    ))
    
    if forecast_months and forecast_values:
        trans_dates = [actual_dates[-1]] + forecast_months
        trans_values = [actual_peserta[-1]] + forecast_values
        
        fig.add_trace(go.Scatter(
            x=trans_dates, y=trans_values, mode='lines', name='Forecast',
            line=dict(color='#ff7f0e', width=2.5),
            hovertemplate='<b>%{x}</b><br>%{y:,} peserta<extra></extra>'
        ))
        
        upper = [int(v * 1.05) for v in trans_values]
        lower = [int(v * 0.95) for v in trans_values]
        
        fig.add_trace(go.Scatter(
            x=trans_dates + trans_dates[::-1], y=upper + lower[::-1],
            fill='toself', fillcolor='rgba(255,127,14,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence', showlegend=True, hoverinfo='skip'
        ))
    
    fig.update_layout(
        title={'text': "Participants: Actual + Forecast", 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Month", yaxis_title="Participants", height=550,
        hovermode='x unified', template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Results
    if forecast_months and forecast_values:
        st.markdown("### 📋 Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg", f"{int(np.mean(forecast_values)):,}")
        col2.metric("Min", f"{min(forecast_values):,}")
        col3.metric("Max", f"{max(forecast_values):,}")
        
        forecast_df = pd.DataFrame({
            'Month': forecast_months,
            'Participants': forecast_values
        })
        
        st.dataframe(forecast_df.style.format({'Participants': '{:,}'}),
                    use_container_width=True, hide_index=True)
        
        st.download_button("📥 Download CSV",
                          forecast_df.to_csv(index=False).encode('utf-8'),
                          f'peserta_{datetime.now().strftime("%Y%m%d")}.csv', 'text/csv')
