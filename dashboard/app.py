"""
Enhanced Interactive Streamlit Dashboard for Aadhaar Enrollment Data Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# Set default Plotly template to dark
pio.templates.default = "plotly_dark"

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="UIDAI Enrollment Analytics",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Color Palette (Dark Mode Optimized)
COLORS = {
    'primary': '#4dabf7',        # Bright Blue
    'secondary': '#a0a0a0',      # Light Grey
    'success': '#00b894',        # Green
    'warning': '#fdcb6e',        # Yellow
    'danger': '#ff7675',         # Red
    'background': '#0e1117',     # Dark Background
    'card_bg': '#262730',        # Card Background
    'text': '#fafafa',           # Main Text
    'chart_colors': px.colors.qualitative.Pastel
}

# Custom CSS
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #fafafa !important;
        font-weight: 700;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #363945;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 8px 3px rgba(0,0,0,0.3);
        transform: translateY(-2px);
        border-color: #4dabf7;
    }
    div[data-testid="metric-container"] > div {
        color: #fafafa;
    }
    /* Metric Label */
    div[data-testid="metric-container"] label {
        color: #a0a0a0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #363945;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #a0a0a0;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262730;
        color: #4dabf7;
        border-bottom: 2px solid #4dabf7;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #363945;
    }
    
    /* Charts Container */
    .js-plotly-plot {
        border-radius: 8px;
        background-color: #262730 !important;
        padding: 10px; 
        border: 1px solid #363945;
    }
    
    /* Dataframes */
    div[data-testid="stDataFrame"] {
        background-color: #262730;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading & Processing
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load and cache the cleaned data."""
    import os
    
    # Try multiple paths to find the data file
    if os.path.exists('data/aadhar_enrollment_cleaned.csv'):
        path = 'data/aadhar_enrollment_cleaned.csv'
    elif os.path.exists('../data/aadhar_enrollment_cleaned.csv'):
        path = '../data/aadhar_enrollment_cleaned.csv'
    else:
        st.error("❌ Data file not found! Please run 'main.py' first to generate the dataset in data/ folder.")
        st.stop()
        
    try:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if 'day_name' in df.columns:
            df['day_name'] = pd.Categorical(df['day_name'], categories=day_order, ordered=True)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# -----------------------------------------------------------------------------
# Chart Helper Functions
# -----------------------------------------------------------------------------
def format_large_number(num):
    if num >= 1_000_000_0:
        return f"{num/10000000:.2f} Cr"
    elif num >= 100_000:
        return f"{num/100000:.2f} L"
    elif num >= 1_000:
        return f"{num/1000:.2f} K"
    return f"{num:.0f}"

def create_trend_chart(df, granularity='D'):
    """Create interactive trend chart with selectable granularity."""
    if granularity == 'W':
        data = df.resample('W', on='date')['total_enrollments'].sum().reset_index()
    elif granularity == 'M':
        data = df.resample('M', on='date')['total_enrollments'].sum().reset_index()
    else:
        data = df.groupby('date')['total_enrollments'].sum().reset_index()
    
    # Calculate Moving Average
    data['MA'] = data['total_enrollments'].rolling(window=7 if granularity=='D' else 3).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['date'], y=data['total_enrollments'],
        mode='lines', name='Enrollments',
        line=dict(color=COLORS['primary'], width=2),
        fill='tozeroy', fillcolor='rgba(26, 115, 232, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=data['date'], y=data['MA'],
        mode='lines', name=f"Moving Average",
        line=dict(color=COLORS['warning'], width=2, dash='dot')
    ))
    
    fig.update_layout(
        title='<b>Enrollment Trend Over Time</b>',
        xaxis_title='Date',
        yaxis_title='Total Enrollments',
        hovermode='x unified',
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_demographics_sunburst(df):
    """Create a sunburst chart for Age Group distribution."""
    # Summarize data
    age_groups = {
        '0-5 Years': df['age_0_5'].sum(),
        '5-17 Years': df['age_5_17'].sum(),
        '18+ Years': df['age_18_greater'].sum()
    }
    
    labels = list(age_groups.keys())
    values = list(age_groups.values())
    parents = ["Total Data"] * 3
    
    fig = px.sunburst(
        names=labels,
        parents=[""] * len(labels),
        values=values,
        title="<b>Age Group Breakdown</b>",
        color=labels,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textinfo="label+percent entry")
    fig.update_layout(height=400, margin=dict(t=50, l=0, r=0, b=0))
    return fig

def create_state_comparison(df):
    """Create a stacked bar chart comparing age demographics across top states."""
    top_states = df.groupby('state')['total_enrollments'].sum().nlargest(10).index
    filtered = df[df['state'].isin(top_states)]
    
    state_demo = filtered.groupby('state')[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
    
    # Normalize for 100% stacked bar
    state_demo['total'] = state_demo['age_0_5'] + state_demo['age_5_17'] + state_demo['age_18_greater']
    
    fig = go.Figure()
    for col, name, color in zip(['age_0_5', 'age_5_17', 'age_18_greater'], 
                               ['0-5 Years', '5-17 Years', '18+ Years'],
                               ['#4285f4', '#34a853', '#fbbc04']):
        fig.add_trace(go.Bar(
            y=state_demo['state'],
            x=state_demo[col],
            name=name,
            orientation='h',
            marker_color=color
        ))
    
    fig.update_layout(
        barmode='stack',
        title='<b>Demographic Composition by Top States</b>',
        xaxis_title='Number of Enrollments',
        yaxis={'categoryorder':'total ascending'},
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def create_district_performance(df, state=None):
    """TreeMap of District Performance."""
    if state and state != "All States":
        data = df[df['state'] == state]
        path = ['state', 'district', 'pincode'] # Drill down to pincode if state selected
        title = f'Enrollment Distribution in {state}'
    else:
        data = df
        path = ['state', 'district']
        title = 'Enrollment Distribution Comparison'
        
    # Limit to top 200 records for performance if showing all states
    if not state or state == "All States":
        data = data.groupby(path)['total_enrollments'].sum().reset_index().nlargest(200, 'total_enrollments')
    else:
        data = data.groupby(path)['total_enrollments'].sum().reset_index()
        
    fig = px.treemap(
        data,
        path=path,
        values='total_enrollments',
        color='total_enrollments',
        color_continuous_scale='Blues',
        title=f'<b>{title}</b>'
    )
    fig.update_layout(height=600)
    return fig

def create_heatmap_grid(df):
    """Heatmap of Day vs Month."""
    pivot = df.pivot_table(
        index='day_name', 
        columns='month_name', 
        values='total_enrollments', 
        aggfunc='mean'
    )
    
    # Sort months correctly
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    available_months = [m for m in month_order if m in pivot.columns]
    pivot = pivot[available_months]
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Day of Week", color="Avg Enrollments"),
        x=available_months,
        y=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        color_continuous_scale='Viridis',
        title="<b>Average Daily Enrollments Heatmap</b>"
    )
    fig.update_layout(height=400)
    return fig

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Load Data
    df = load_data()
    if df is None:
        return

    # Sidebar Controls
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        
        # Date Filter
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.date_input(
            "📅 Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        st.markdown("---")
        
        # State Filter
        all_states = sorted(df['state'].unique().tolist())
        selected_states = st.multiselect(
            "🗺️ Select States",
            options=all_states,
            default=None,
            placeholder="All States (Default)"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info(
            "This dashboard visualizes Aadhaar enrollment data, providing insights "
            "into demographic trends, geographic performance, and operational efficiency."
        )

    # Filtering Logic
    filtered_df = df.copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
        
    if selected_states:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]

    # Main Header
    st.title("🇮🇳 UIDAI Aadhaar Enrollment Analytics")
    st.markdown(f"**Analysis Period:** {date_range[0]} to {date_range[1]} | **Regions:** {'All' if not selected_states else ', '.join(selected_states)}")
    st.markdown("---")

    # High-level Metrics (KPIs)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_enrollment = filtered_df['total_enrollments'].sum()
    avg_daily = filtered_df.groupby('date')['total_enrollments'].sum().mean()
    active_districts = filtered_df['district'].nunique()
    top_pincode = filtered_df.groupby('pincode')['total_enrollments'].sum().idxmax()
    
    # Calculate Growth (vs previous period of same length - simplified to just show current)
    # Could imply complex logic, but sticking to simple clean display
    
    with kpi1:
        st.metric("Total Enrollments", format_large_number(total_enrollment), delta="Total Volume")
    with kpi2:
        st.metric("Avg Daily Enrollments", f"{avg_daily:,.0f}", delta="Throughput")
    with kpi3:
        st.metric("Active Districts", active_districts, delta="Coverage")
    with kpi4:
        st.metric("Top Pincode", str(top_pincode), delta="Highest Volume")

    st.markdown("---")

    # Navigation Tabs
    tab_overview, tab_geo, tab_demo, tab_trends, tab_data = st.tabs([
        "🏠 Overview", 
        "🗺️ Geographic", 
        "👥 Demographics", 
        "📅 Trends", 
        "📋 Data & Export"
    ])

    # ------------------ TAB: OVERVIEW ------------------
    with tab_overview:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(create_trend_chart(filtered_df, 'D'), width="stretch")
            
        with col2:
            st.plotly_chart(create_demographics_sunburst(filtered_df), width="stretch")
            
        # Quick State Summary
        st.subheader("🏆 Top Performing States")
        top_states_bar = px.bar(
            filtered_df.groupby('state')['total_enrollments'].sum().nlargest(10).reset_index(),
            x='total_enrollments',
            y='state',
            orientation='h',
            text_auto='.2s',
            title="",
            color='total_enrollments',
            color_continuous_scale='Blues'
        )
        top_states_bar.update_layout(height=400, xaxis_title="Total Enrollments", yaxis_title=None, showlegend=False)
        st.plotly_chart(top_states_bar, width="stretch")

    # ------------------ TAB: GEOGRAPHIC ------------------
    with tab_geo:
        st.subheader("📍 State & District Deep Dive")
        
        col_ctrl, col_map = st.columns([1, 3])
        with col_ctrl:
            drill_state = st.selectbox("Select State to Drill Down", ["All States"] + all_states)
            
        st.plotly_chart(create_district_performance(filtered_df, drill_state), width="stretch")
        
        # State Comparison Logic
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Highest Growth Districts")
            # Simply showing top districts by volume here as "Growth" requires time comparison logic
            top_dist = filtered_df.groupby(['state', 'district'])['total_enrollments'].sum().nlargest(10).reset_index()
            fig_top_dist = px.bar(top_dist, x='total_enrollments', y='district', color='state', orientation='h', title="Top 10 Districts by Volume")
            st.plotly_chart(fig_top_dist, width="stretch")
            
        with col2:
            st.subheader("Lowest Performing Districts (Bottom 10)")
            bot_dist = filtered_df.groupby(['state', 'district'])['total_enrollments'].sum().nsmallest(10).reset_index()
            fig_bot_dist = px.bar(bot_dist, x='total_enrollments', y='district', color='state', orientation='h', title="Bottom 10 Districts by Volume")
            st.plotly_chart(fig_bot_dist, width="stretch")

    # ------------------ TAB: DEMOGRAPHICS ------------------
    with tab_demo:
        st.subheader("👥 Population Analysis")
        
        st.plotly_chart(create_state_comparison(filtered_df), width="stretch")
        
        col1, col2 = st.columns(2)
        with col1:
            # Distribution of Age 18+
            fig_hist = px.histogram(filtered_df, x="age_18_greater", nbins=50, title="<b>Distribution of Adult Enrollments (Daily)</b>", color_discrete_sequence=['#34a853'])
            st.plotly_chart(fig_hist, width="stretch")
        
        with col2:
            # Correlation Matrix (Simplified)
            corr_df = filtered_df[['age_0_5', 'age_5_17', 'age_18_greater', 'total_enrollments']].corr()
            fig_corr = px.imshow(corr_df, text_auto=True, title="<b>Correlation: Age Groups vs Total</b>", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, width="stretch")

    # ------------------ TAB: TRENDS ------------------
    with tab_trends:
        st.subheader("📅 Temporal Patterns")
        
        # Seasonality / Weekday Pattern
        col1, col2 = st.columns(2)
        
        with col1:
            avg_by_day = filtered_df.groupby('day_name')['total_enrollments'].mean().reset_index()
            fig_day = px.bar(avg_by_day, x='day_name', y='total_enrollments', title="<b>Average Enrollments by Day of Week</b>", color='total_enrollments', color_continuous_scale='Viridis')
            st.plotly_chart(fig_day, width="stretch")
            
        with col2:
            st.markdown("#### 🌡️ Efficiency Heatmap")
            st.plotly_chart(create_heatmap_grid(filtered_df), width="stretch")
            
        # Monthly View
        st.subheader("Monthly Trajectory")
        monthly_trend = filtered_df.resample('M', on='date')['total_enrollments'].sum().reset_index()
        fig_monthly = px.area(monthly_trend, x='date', y='total_enrollments', title="Monthly Enrollment Volume", markers=True)
        st.plotly_chart(fig_monthly, width="stretch")

    # ------------------ TAB: DATA & EXPORT ------------------
    with tab_data:
        st.subheader("📋 Raw Data Explorer")
        
        with st.expander("🔎 Advanced Data Filters"):
            min_val = int(filtered_df['total_enrollments'].min())
            max_val = int(filtered_df['total_enrollments'].max())
            enrollment_range = st.slider("Filter by Enrollment Count", min_val, max_val, (min_val, max_val))
        
        data_view = filtered_df[
            (filtered_df['total_enrollments'] >= enrollment_range[0]) & 
            (filtered_df['total_enrollments'] <= enrollment_range[1])
        ]
        
        st.dataframe(
            data_view.sort_values(by='date', ascending=False),
            column_config={
                "date": "Date",
                "state": "State",
                "district": "District",
                "total_enrollments": st.column_config.NumberColumn("Total", format="%d"),
            },
            width="stretch",
            height=600
        )
        
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=data_view.to_csv(index=False).encode('utf-8'),
            file_name=f'uidai_enrollment_export_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
