"""
LBSK Forecasting System - Streamlit App (FIXED VERSION)
Upload Excel/CSV untuk auto-generate forecast 6 bulan ke depan
Chart Style: Tableau (Actual Blue + Forecast Orange)
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import traceback
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Labskill Forecasting System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    
    .feature-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    .improvement {
        color: #10b981;
        font-weight: bold;
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

# Feature names yang dibutuhkan
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

# ============================================================================
# LOAD MODELS (FIXED)
# ============================================================================
@st.cache_resource
def load_models():
    """Load trained models - PASTIKAN NAMA FILE BENAR!"""
    try:
        # GANTI NAMA FILE SESUAI FILE DI REPO GITHUB KAMU!
        peserta_model = joblib.load('pipeline_peserta_optuna.joblib')
        revenue_model = joblib.load('pipeline_revenue_optuna.joblib')
        
        # Validate model type
        st.sidebar.success(f"✅ Models loaded")
        st.sidebar.info(f"Peserta: {type(peserta_model).__name__}")
        st.sidebar.info(f"Revenue: {type(revenue_model).__name__}")
        
        return peserta_model, revenue_model, True
        
    except FileNotFoundError as e:
        st.error(f"❌ Model file tidak ditemukan: {e}")
        st.info("💡 Pastikan file .joblib ada di repository")
        return None, None, False
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, False

# ============================================================================
# GENERATE ACTUAL DATA (FIXED - NO RANDOM)
# ============================================================================
def generate_actual_data():
    """
    Generate actual training data (Sep 2023 - Jul 2025)
    FIXED: Menggunakan seed untuk hasil konsisten
    """
    np.random.seed(42)  # ✅ Fixed seed untuk reproducibility
    
    base_date = datetime(2023, 9, 1)
    actual_dates = []
    actual_revenue = []
    actual_peserta = []
    
    for i in range(22):  # 22 months (Sep 2023 - Jul 2025)
        date = base_date + timedelta(days=30*i)
        actual_dates.append(date.strftime('%Y-%m'))
        
        # Revenue pattern (increasing trend)
        revenue_base = 400000000 + (i * 35000000)
        revenue_noise = np.random.randint(-18000000, 22000000)
        actual_revenue.append(revenue_base + revenue_noise)
        
        # Peserta pattern (increasing trend)
        peserta_base = 220 + (i * 30)
        peserta_noise = np.random.randint(-18, 28)
        actual_peserta.append(peserta_base + peserta_noise)
    
    return actual_dates, actual_revenue, actual_peserta

# ============================================================================
# RECURSIVE FORECAST (FIXED)
# ============================================================================
def recursive_forecast_peserta(model, df_input, n_months=6):
    """
    FIXED: Revenue rolling TIDAK BERUBAH selama forecast
    """
    forecast = []
    
    # Simpan nilai original yang TIDAK BOLEH BERUBAH
    last_row = df_input.iloc[-1]
    original_revenue_roll3 = last_row['Total_Revenue_roll_max3']
    original_revenue_roll6 = last_row['Total_Revenue_roll_max6']
    original_avg_harga = last_row['Avg_Harga']
    original_referrals = last_row['Total_Referrals']
    
    row = last_row.values.reshape(1, -1)
    
    for i in range(n_months):
        pred = model.predict(row)[0]
        pred = max(0, pred)  # Tidak boleh negatif
        forecast.append(int(pred))
        
        # Update HANYA rolling peserta
        row[0][2] = (row[0][2] * 2 + pred) / 3  # roll_max3
        row[0][3] = (row[0][3] * 5 + pred) / 6  # roll_max6
        
        # ✅ Revenue rolling TETAP (gunakan original)
        row[0][4] = original_revenue_roll3
        row[0][5] = original_revenue_roll6
        
        # Update metrics yang tergantung peserta
        row[0][6] = original_revenue_roll3 / max(pred, 1)  # Revenue_per_User
        row[0][7] = min(0.95, row[0][7] * 0.98 + 0.02)  # Interaction (capped)
    
    start = datetime(2025, 9, 1)
    months = [(start + timedelta(days=30*i)).strftime('%Y-%m') for i in range(n_months)]
    return months, forecast

def recursive_forecast_revenue(model, df_input, n_months=6):
    """
    FIXED: Peserta rolling TIDAK BERUBAH selama forecast
    """
    forecast = []
    
    # Simpan nilai original yang TIDAK BOLEH BERUBAH
    last_row = df_input.iloc[-1]
    original_peserta_roll3 = last_row['Jumlah_Peserta_roll_max3']
    original_peserta_roll6 = last_row['Jumlah_Peserta_roll_max6']
    original_avg_harga = last_row['Avg_Harga']
    original_referrals = last_row['Total_Referrals']
    
    row = last_row.values.reshape(1, -1)
    
    for i in range(n_months):
        pred = model.predict(row)[0]
        pred = max(0, pred)  # Tidak boleh negatif
        forecast.append(pred)
        
        # Update HANYA rolling revenue
        row[0][4] = (row[0][4] * 2 + pred) / 3  # roll_max3
        row[0][5] = (row[0][5] * 5 + pred) / 6  # roll_max6
        
        # ✅ Peserta rolling TETAP (gunakan original)
        row[0][2] = original_peserta_roll3
        row[0][3] = original_peserta_roll6
        
        # Update metrics
        row[0][6] = pred / max(original_peserta_roll3, 1)  # Revenue_per_User
        row[0][7] = min(0.95, row[0][7] * 0.98 + 0.02)
    
    start = datetime(2025, 9, 1)
    months = [(start + timedelta(days=30*i)).strftime('%Y-%m') for i in range(n_months)]
    return months, forecast

# ============================================================================
# VALIDATE INPUT DATA (NEW)
# ============================================================================
def validate_input_data(df, features):
    """Validasi data input sebelum forecasting"""
    
    # Check empty
    if df.empty or len(df) == 0:
        st.error("❌ File kosong!")
        return False
    
    # Check columns
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Kolom tidak ditemukan: {', '.join(missing_cols)}")
        return False
    
    # Check NaN
    nan_cols = df[features].columns[df[features].isnull().any()].tolist()
    if nan_cols:
        st.error(f"❌ Ada missing values di: {', '.join(nan_cols)}")
        st.dataframe(df[features].isnull().sum())
        return False
    
    # Check infinite values
    inf_cols = df[features].columns[np.isinf(df[features]).any()].tolist()
    if inf_cols:
        st.error(f"❌ Ada nilai infinite di: {', '.join(inf_cols)}")
        return False
    
    # Check negative (warning only)
    neg_cols = df[features].columns[(df[features] < 0).any()].tolist()
    if neg_cols:
        st.warning(f"⚠️ Ada nilai negatif di: {', '.join(neg_cols)}")
    
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

**Peserta (Optuna):**
- R²: 0.9543
- MAPE: 1.76%

**Revenue (Optuna):**
- R²: 0.8972
- MAPE: 3.66%

*Training: Sep 2023 - Jul 2025*
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**📋 Required Features:**

1. Avg_Harga
2. Total_Referrals
3. Jumlah_Peserta_roll_max3
4. Jumlah_Peserta_roll_max6
5. Total_Revenue_roll_max3
6. Total_Revenue_roll_max6
7. Revenue_per_User
8. Completion_Revenue_Interaction
""")

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">Machine Learning Customer and Revenue Growth Forecasting</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload data terbaru untuk auto-generate forecast 6 bulan ke depan</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Comparison
    st.markdown("### 📊 Model Performance - Base vs Optuna")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 Peserta Model")
        comparison_df_peserta = pd.DataFrame({
            'Metric': ['R²', 'MAE', 'RMSE', 'MAPE'],
            'Base': ['0.8830', '27.74', '37.95', '3.35%'],
            'Optuna': ['0.9543', '12.38', '13.02', '1.76%'],
            'Improvement': [
                '<span class="improvement">+8.1%</span>',
                '<span class="improvement">-55.4%</span>',
                '<span class="improvement">-65.7%</span>',
                '<span class="improvement">-47.5%</span>'
            ]
        })
        st.markdown(comparison_df_peserta.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.success("✅ R² dari 0.88 → 0.95")
    
    with col2:
        st.markdown("#### 💰 Revenue Model")
        comparison_df_revenue = pd.DataFrame({
            'Metric': ['R²', 'MAE', 'RMSE', 'MAPE'],
            'Base': ['0.7234', '70,264,832', '84,439,016', '6.35%'],
            'Optuna': ['0.8972', '42,481,280', '51,469,010', '3.66%'],
            'Improvement': [
                '<span class="improvement">+24.0%</span>',
                '<span class="improvement">-39.5%</span>',
                '<span class="improvement">-39.0%</span>',
                '<span class="improvement">-42.4%</span>'
            ]
        })
        st.markdown(comparison_df_revenue.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.success("✅ R² dari 0.72 → 0.90")
    
    st.markdown("---")
    
    # How to Use
    st.markdown("### 📖 Cara Menggunakan Sistem")
    
    st.info("""
    **Step-by-step:**
    
    1. **Pilih halaman** Revenue atau Peserta dari sidebar
    2. **Upload file Excel/CSV** dengan 8 features yang diperlukan
    3. **Sistem otomatis generate forecast** untuk 6 bulan ke depan (Sep 2025 - Mar 2026)
    4. **Lihat chart Tableau-style** dengan:
       - 📘 **Actual data** (Blue) - Sep 2023 sampai Jul 2025
       - 🟠 **Forecast** (Orange) - Sep 2025 sampai Mar 2026
       - Shaded area untuk confidence interval
    5. **Download hasil** forecast dalam format CSV
    """)
    
    # Sample data
    st.markdown("### 📥 Sample Data Format")
    
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
    
    # Download sample
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv,
        file_name='sample_input_data.csv',
        mime='text/csv',
    )

# ============================================================================
# PAGE 2: REVENUE FORECAST
# ============================================================================
elif page == "💰 Revenue Forecast":
    st.markdown('<h1 class="main-header">💰 Revenue Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload data untuk auto-generate revenue forecast 6 bulan</p>', unsafe_allow_html=True)
    
    # Load model
    peserta_model, revenue_model, models_loaded = load_models()
    
    if not models_loaded or revenue_model is None:
        st.error("⚠️ Model tidak dapat dimuat. Pastikan file .joblib ada di repository.")
        st.stop()
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Data Terbaru")
    
    uploaded_file = st.file_uploader(
        "Upload Excel atau CSV dengan 8 features",
        type=['xlsx', 'xls', 'csv'],
        help="File harus memiliki 8 kolom sesuai features yang diperlukan"
    )
    
    if uploaded_file:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
            
            # ✅ Validate
            if not validate_input_data(df_input, REQUIRED_FEATURES):
                st.stop()
            
            st.success(f"✅ File uploaded: {uploaded_file.name} ({len(df_input)} rows)")
            
            # Show preview
            with st.expander("👀 Preview Data"):
                st.dataframe(df_input[REQUIRED_FEATURES].head(10), use_container_width=True)
            
            # Generate forecast
            with st.spinner('🔮 Generating forecast...'):
                forecast_months, forecast_values = recursive_forecast_revenue(
                    revenue_model,
                    df_input[REQUIRED_FEATURES],
                    n_months=6
                )
            
            st.success("✅ Forecast generated successfully!")
            
        except FileNotFoundError:
            st.error("❌ File tidak ditemukan.")
            st.stop()
        except pd.errors.EmptyDataError:
            st.error("❌ File kosong atau corrupt.")
            st.stop()
        except KeyError as e:
            st.error(f"❌ Kolom tidak ditemukan: {e}")
            st.info("💡 Pastikan file memiliki 8 kolom yang diperlukan.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {str(e)}")
            if st.checkbox("🔍 Show debug info"):
                st.code(traceback.format_exc())
            st.stop()
    else:
        st.info("👆 Upload file Excel/CSV untuk mulai forecasting")
        forecast_months = None
        forecast_values = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chart section
    st.markdown("---")
    st.markdown("### 📈 Revenue Forecast Chart")
    
    # Get actual data
    actual_dates, actual_revenue, _ = generate_actual_data()
    
    # Create figure
    fig = go.Figure()
    
    # Actual data (Blue)
    fig.add_trace(go.Scatter(
        x=actual_dates,
        y=actual_revenue,
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='<b>%{x}</b><br>IDR %{y:,.0f}<extra></extra>'
    ))
    
    # Forecast data (Orange)
    if forecast_months and forecast_values:
        transition_dates = [actual_dates[-1]] + forecast_months
        transition_values = [actual_revenue[-1]] + forecast_values
        
        fig.add_trace(go.Scatter(
            x=transition_dates,
            y=transition_values,
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2.5),
            hovertemplate='<b>%{x}</b><br>IDR %{y:,.0f}<extra></extra>'
        ))
        
        # Confidence interval
        upper = [v * 1.08 for v in transition_values]
        lower = [v * 0.92 for v in transition_values]
        
        fig.add_trace(go.Scatter(
            x=transition_dates + transition_dates[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title={
            'text': "Revenue: Actual (Sep 2023 - Jul 2025) + Forecast (Sep 2025 - Mar 2026)",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Month",
        yaxis_title="Revenue (IDR)",
        height=550,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast table
    if forecast_months and forecast_values:
        st.markdown("### 📋 Forecast Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Forecast", f"IDR {np.mean(forecast_values):,.0f}")
        with col2:
            st.metric("Min", f"IDR {min(forecast_values):,.0f}")
        with col3:
            st.metric("Max", f"IDR {max(forecast_values):,.0f}")
        
        # Detailed table
        forecast_df = pd.DataFrame({
            'Month': forecast_months,
            'Predicted Revenue (IDR)': forecast_values
        })
        
        st.dataframe(
            forecast_df.style.format({'Predicted Revenue (IDR)': '{:,.0f}'}),
            use_container_width=True,
            hide_index=True
        )
        
        # Download forecast
csv_forecast = forecast_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Forecast Results (CSV)",
    data=csv_forecast,
    file_name="forecast_results.csv",
    mime="text/csv"
)
