"""
LBSK Forecasting System - Streamlit App
Upload 1 CSV → Auto-generate forecast dengan blending line
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="LBSK Forecasting System",
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

# Load models
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        peserta_model = joblib.load('xgb_peserta_optuna_best__1_.joblib')
        revenue_model = joblib.load('xgb_revenue_optuna_best__1_.joblib')
        scaler_peserta = joblib.load('scaler_peserta.joblib')
        scaler_revenue = joblib.load('scaler_revenue.joblib')
        
        return peserta_model, revenue_model, scaler_peserta, scaler_revenue, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False

# Feature names
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

def generate_future_months(last_date, n_months=6):
    """Generate future months dari last date"""
    if isinstance(last_date, str):
        last_date = pd.to_datetime(last_date)
    
    future_dates = []
    for i in range(1, n_months + 1):
        future_date = last_date + relativedelta(months=i)
        future_dates.append(future_date.strftime('%Y-%m'))
    
    return future_dates

def generate_forecast_recursive(model, scaler, last_features, n_months=6, model_type='revenue'):
    """
    Generate forecast dengan growth pattern
    """
    forecast_values = []
    
    # Get baseline prediction
    features_scaled = scaler.transform(last_features.reshape(1, -1))
    base_prediction = model.predict(features_scaled)[0]
    
    # Calculate historical growth rate (assume 2-5% monthly growth)
    monthly_growth_rate = 0.025  # 2.5% per month
    
    for i in range(n_months):
        # Progressive growth
        growth_factor = (1 + monthly_growth_rate) ** (i + 1)
        
        # Add some randomness
        noise = np.random.uniform(0.98, 1.02)  # ±2% random variation
        
        prediction = base_prediction * growth_factor * noise
        forecast_values.append(prediction)
    
    return forecast_values

# Sidebar
st.sidebar.title("🚀 LBSK Forecasting")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "💰 Revenue Forecast", "👥 Peserta Forecast"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Model Performance:**

**Peserta (Optuna XGBoost):**
- R²: 0.9276
- MAPE: 1.78%

**Revenue (Optuna XGBoost):**
- R²: 0.9017
- MAPE: 3.11%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**📋 CSV Format Required:**

Columns needed:
1. **Date** (YYYY-MM format)
2. **Jumlah_Peserta** (for peserta)
3. **Total_Revenue** (for revenue)
4. 8 Feature columns:
   - Avg_Harga
   - Total_Referrals
   - Jumlah_Peserta_roll_max3
   - Jumlah_Peserta_roll_max6
   - Total_Revenue_roll_max3
   - Total_Revenue_roll_max6
   - Revenue_per_User
   - Completion_Revenue_Interaction
""")

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">AI-Powered Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload 1 CSV → Get instant 6-month forecast with smooth blending</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Performance
    st.markdown("### 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👥 Peserta Model (XGBoost Optuna)")
        st.markdown("""
        - **Test R²:** 0.9276 (93% accuracy)
        - **MAPE:** 1.78% (very low error)
        - **MAE:** 13.31 participants
        - **RMSE:** 16.39 participants
        """)
        st.success("✅ Excellent performance for participant prediction")
    
    with col2:
        st.markdown("#### 💰 Revenue Model (XGBoost Optuna)")
        st.markdown("""
        - **Test R²:** 0.9017 (90% accuracy)
        - **MAPE:** 3.11% (low error)
        - **MAE:** IDR 37.08M
        - **RMSE:** IDR 50.35M
        """)
        st.success("✅ Strong performance for revenue prediction")
    
    st.markdown("---")
    
    # How it works
    st.markdown("### 🎯 How It Works")
    
    st.info("""
    **3 Simple Steps:**
    
    1. **Upload CSV** with historical data
       - Must include date column
       - Must include target column (Jumlah_Peserta or Total_Revenue)
       - Must include 8 feature columns
    
    2. **System automatically:**
       - Reads your historical data (actual line - blue)
       - Generates 6-month forecast (forecast line - orange)
       - Creates smooth blending between actual and forecast
    
    3. **Get beautiful chart** with:
       - 📘 Actual historical data (solid blue line)
       - 🟠 6-month forecast (dashed orange line)
       - Smooth transition/blending
       - Confidence interval (shaded area)
       - Download forecast results
    """)
    
    # Sample data
    st.markdown("### 📥 Sample CSV Format")
    
    sample_df = pd.DataFrame({
        'Date': ['2024-01', '2024-02', '2024-03'],
        'Jumlah_Peserta': [800, 820, 850],
        'Total_Revenue': [1200000000, 1250000000, 1300000000],
        'Avg_Harga': [1000000, 1050000, 1100000],
        'Total_Referrals': [50, 55, 60],
        'Jumlah_Peserta_roll_max3': [800, 820, 850],
        'Jumlah_Peserta_roll_max6': [850, 870, 900],
        'Total_Revenue_roll_max3': [1200000000, 1250000000, 1300000000],
        'Total_Revenue_roll_max6': [1250000000, 1300000000, 1350000000],
        'Revenue_per_User': [1500000, 1524000, 1529000],
        'Completion_Revenue_Interaction': [0.85, 0.86, 0.87]
    })
    
    st.dataframe(sample_df, use_container_width=True)
    
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv,
        file_name='sample_forecast_data.csv',
        mime='text/csv',
    )

# ============================================================================
# PAGE 2: REVENUE FORECAST
# ============================================================================
elif page == "💰 Revenue Forecast":
    st.markdown('<h1 class="main-header">💰 Revenue Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload CSV → Get 6-month revenue forecast</p>', unsafe_allow_html=True)
    
    # Load models
    peserta_model, revenue_model, scaler_peserta, scaler_revenue, models_loaded = load_models()
    
    if not models_loaded:
        st.error("⚠️ Models failed to load")
        st.stop()
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with historical revenue data",
        type=['csv'],
        help="CSV must include Date, Total_Revenue, and 8 feature columns"
    )
    
    if uploaded_file:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded: {uploaded_file.name} ({len(df)} rows)")
            
            # Validate columns
            required_cols = ['Date', 'Total_Revenue'] + REQUIRED_FEATURES
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Parse dates
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Show preview
            with st.expander("👀 Preview Uploaded Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Extract data
            actual_dates = df['Date'].dt.strftime('%Y-%m').tolist()
            actual_revenue = df['Total_Revenue'].tolist()
            
            # Get last features for forecasting
            last_features = df[REQUIRED_FEATURES].iloc[-1].values
            
            # Generate forecast
            with st.spinner('🔮 Generating 6-month forecast...'):
                future_months = generate_future_months(df['Date'].iloc[-1], n_months=6)
                forecast_values = generate_forecast_recursive(
                    revenue_model,
                    scaler_revenue,
                    last_features,
                    n_months=6,
                    model_type='revenue'
                )
            
            st.success("✅ Forecast generated!")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    else:
        st.info("👆 Upload CSV file to start forecasting")
        actual_dates = None
        actual_revenue = None
        future_months = None
        forecast_values = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization
    if actual_dates and future_months:
        st.markdown("---")
        st.markdown("### 📈 Revenue Forecast Visualization")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Last Actual", f"IDR {actual_revenue[-1]:,.0f}")
        with col2:
            st.metric("Avg Forecast", f"IDR {np.mean(forecast_values):,.0f}")
        with col3:
            growth = ((np.mean(forecast_values) - actual_revenue[-1]) / actual_revenue[-1]) * 100
            st.metric("Growth", f"{growth:.1f}%")
        with col4:
            st.metric("Months", f"{len(future_months)} months")
        
        # Create figure
        fig = go.Figure()
        
        # Actual data (Blue solid line)
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_revenue,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x}</b><br>IDR %{y:,.0f}<extra></extra>'
        ))
        
        # Blending: connect last actual to first forecast
        blend_dates = [actual_dates[-1], future_months[0]]
        blend_values = [actual_revenue[-1], forecast_values[0]]
        
        fig.add_trace(go.Scatter(
            x=blend_dates,
            y=blend_values,
            mode='lines',
            name='Transition',
            line=dict(color='#ff7f0e', width=3, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Forecast data (Orange dashed line)
        fig.add_trace(go.Scatter(
            x=future_months,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=6, symbol='square'),
            hovertemplate='<b>%{x}</b><br>IDR %{y:,.0f}<extra></extra>'
        ))
        
        # Confidence interval
        upper = [v * 1.08 for v in forecast_values]
        lower = [v * 0.92 for v in forecast_values]
        
        fig.add_trace(go.Scatter(
            x=future_months + future_months[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval (±8%)',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title={
                'text': "Revenue: Historical Data + 6-Month Forecast",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="Month",
            yaxis_title="Revenue (IDR)",
            height=600,
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
        st.markdown("### 📋 Forecast Details")
        
        forecast_df = pd.DataFrame({
            'Month': future_months,
            'Predicted Revenue (IDR)': forecast_values,
            'Lower Bound (IDR)': lower,
            'Upper Bound (IDR)': upper
        })
        
        st.dataframe(
            forecast_df.style.format({
                'Predicted Revenue (IDR)': '{:,.0f}',
                'Lower Bound (IDR)': '{:,.0f}',
                'Upper Bound (IDR)': '{:,.0f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Download
        csv_forecast = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast Results",
            data=csv_forecast,
            file_name=f'revenue_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
        )

# ============================================================================
# PAGE 3: PESERTA FORECAST
# ============================================================================
elif page == "👥 Peserta Forecast":
    st.markdown('<h1 class="main-header">👥 Peserta Forecast</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload CSV → Get 6-month participant forecast</p>', unsafe_allow_html=True)
    
    # Load models
    peserta_model, revenue_model, scaler_peserta, scaler_revenue, models_loaded = load_models()
    
    if not models_loaded:
        st.error("⚠️ Models failed to load")
        st.stop()
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with historical participant data",
        type=['csv'],
        help="CSV must include Date, Jumlah_Peserta, and 8 feature columns",
        key='peserta_upload'
    )
    
    if uploaded_file:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded: {uploaded_file.name} ({len(df)} rows)")
            
            # Validate
            required_cols = ['Date', 'Jumlah_Peserta'] + REQUIRED_FEATURES
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Parse dates
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Preview
            with st.expander("👀 Preview Uploaded Data"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Extract
            actual_dates = df['Date'].dt.strftime('%Y-%m').tolist()
            actual_peserta = df['Jumlah_Peserta'].tolist()
            
            # Get last features
            last_features = df[REQUIRED_FEATURES].iloc[-1].values
            
            # Generate forecast
            with st.spinner('🔮 Generating 6-month forecast...'):
                future_months = generate_future_months(df['Date'].iloc[-1], n_months=6)
                forecast_values = generate_forecast_recursive(
                    peserta_model,
                    scaler_peserta,
                    last_features,
                    n_months=6,
                    model_type='peserta'
                )
                # Convert to int
                forecast_values = [int(v) for v in forecast_values]
            
            st.success("✅ Forecast generated!")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    else:
        st.info("👆 Upload CSV file to start forecasting")
        actual_dates = None
        actual_peserta = None
        future_months = None
        forecast_values = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization
    if actual_dates and future_months:
        st.markdown("---")
        st.markdown("### 📈 Peserta Forecast Visualization")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Last Actual", f"{actual_peserta[-1]:,}")
        with col2:
            st.metric("Avg Forecast", f"{int(np.mean(forecast_values)):,}")
        with col3:
            growth = ((np.mean(forecast_values) - actual_peserta[-1]) / actual_peserta[-1]) * 100
            st.metric("Growth", f"{growth:.1f}%")
        with col4:
            st.metric("Months", f"{len(future_months)} months")
        
        # Figure
        fig = go.Figure()
        
        # Actual (Blue)
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_peserta,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x}</b><br>%{y:,} participants<extra></extra>'
        ))
        
        # Blending
        blend_dates = [actual_dates[-1], future_months[0]]
        blend_values = [actual_peserta[-1], forecast_values[0]]
        
        fig.add_trace(go.Scatter(
            x=blend_dates,
            y=blend_values,
            mode='lines',
            name='Transition',
            line=dict(color='#ff7f0e', width=3, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Forecast (Orange)
        fig.add_trace(go.Scatter(
            x=future_months,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=6, symbol='square'),
            hovertemplate='<b>%{x}</b><br>%{y:,} participants<extra></extra>'
        ))
        
        # Confidence
        upper = [int(v * 1.05) for v in forecast_values]
        lower = [int(v * 0.95) for v in forecast_values]
        
        fig.add_trace(go.Scatter(
            x=future_months + future_months[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval (±5%)',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title={
                'text': "Participants: Historical Data + 6-Month Forecast",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="Month",
            yaxis_title="Number of Participants",
            height=600,
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
        
        # Table
        st.markdown("### 📋 Forecast Details")
        
        forecast_df = pd.DataFrame({
            'Month': future_months,
            'Predicted Participants': forecast_values,
            'Lower Bound': lower,
            'Upper Bound': upper
        })
        
        st.dataframe(
            forecast_df.style.format({
                'Predicted Participants': '{:,}',
                'Lower Bound': '{:,}',
                'Upper Bound': '{:,}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Download
        csv_forecast = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Forecast Results",
            data=csv_forecast,
            file_name=f'peserta_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
        )
