# 🇮🇳 UIDAI Aadhaar Enrollment Analytics System

A comprehensive, production-grade analytics platform designed to process, analyze, and visualize Aadhaar enrollment data across India. This system leverages advanced machine learning, time-series forecasting, and anomaly detection to provide actionable insights into enrollment trends, operational efficiency, and demographic coverage.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## 🚀 Key Capabilities

This system goes beyond basic descriptive statistics to offer predictive and prescriptive analytics:

### 1. 🔍 Advanced Anomaly Detection
*   **Statistical Anomalies**: Uses Z-score analysis to detect statistically significant spikes or drops in daily enrollment numbers (±3σ).
*   **Geographic Outliers**: Identifies states and districts performing significantly above or below the national average.
*   **Holiday Impact**: Analyzes the specific impact of national holidays on enrollment volumes.

### 2. 🔮 Predictive Forecasting
*   **30-Day Forecast**: Prophet/ARIMA-based time-series modeling to predict daily enrollments for the next month with confidence intervals.
*   **Milestone Tracking**: Projections for when key enrollment milestones (e.g., next 10 million) will be achieved based on current run rates.

### 3. 🤖 Machine Learning & Clustering
*   **District Segmentation**: K-Means clustering to classify districts into "High Growth," "Steady State," or "Underserved" categories based on multidimensional features.
*   **Risk Scoring**: Random Forest models to calculate a "Risk Score" for each district, identifying areas prone to operational bottlenecks or low coverage.
*   **Feature Importance**: specific analysis of which factors (age demographics, location, timing) drive enrollment numbers.

### 4. 📈 Operational Insights
*   **Seasonality Analysis**: Calculates monthly seasonality indices to recommend resource scaling (e.g., "Increase staff in October").
*   **Capacity Utilization**: Models center utilization rates to identify over-burdened or under-utilized infrastructure.
*   **Strategic Recommendations**: Auto-generated, priority-ranked action items for administrators (e.g., "Target District X for camp mode").

### 5. 📊 Comprehensive Visualization Suite
*   **Static Reports**: Generates 40+ high-definition static charts in `images/` including:
    *   **Pareto Charts**: 80/20 analysis of state contributions.
    *   **Treemaps**: Hierarchical view of State > District enrollments.
    *   **Sunburst Charts**: Nested demographic breakdowns.
    *   **Control Charts**: SPC (Statistical Process Control) for monitoring stability.
    *   **Waterfall Charts**: Month-over-month growth decomposition.

### 6. 🖥️ Interactive Dashboard
A fully responsive Streamlit dashboard featuring:
*   **Dark Mode UI**: Professional, eye-strain-free interface.
*   **Drill-Down Capabilities**: Navigate from National -> State -> District levels.
*   **Dynamic Filtering**: Filter by date range, state, and demographic groups.
*   **Geo-Spatial Intelligence**: Interactive maps and heatmaps of enrollment density.

---

## 📂 Project Architecture

The project follows a modular, production-ready structure:

```
uidai_analytics/
├── data/                          # Centralized Data Repository
│   ├── api_data_*.csv             # Raw input files
│   ├── aadhar_enrollment_cleaned.csv # Processed master dataset
│   ├── anomalies_*.csv            # Anomaly detection reports
│   └── forecast_*.csv             # Predictive model outputs
│
├── images/                        # Visualization Artifacts
│   ├── graph_*.png                # Standard descriptive plots (1-25)
│   ├── insight_*.png              # Analytical insight plots (1-10)
│   └── graph_26-32_*.png          # Advanced charts (Treemap, Radar, etc.)
│
├── src/                           # Core Logic Modules
│   ├── data_processing.py         # ETL Pipeline (formerly analysis.py)
│   ├── static_visualizations.py   # Base plotting library
│   ├── advanced_visualizations.py # Complex chart generation
│   ├── anomaly_detection.py       # Outlier detection algorithms
│   ├── forecasting.py             # Time-series models
│   ├── clustering.py              # ML segmentation logic
│   ├── ml_models.py               # Predictive modeling
│   └── operational_insights.py    # Business logic engine
│
├── dashboard/                     # Web Application
│   └── app.py                     # Streamlit entry point
│
├── main.py                        # Master Orchestrator Script
└── requirements.txt               # Dependency definition
```

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8 or higher
*   pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/AbhimanRajCoder/uidai_hackthaon.git
cd uidai_hackthaon
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Usage Guide

### 1. Run the Full Analysis Pipeline
Execute the master script to process data, run ML models, and generate all static reports.
```bash
python3 main.py
```
**What happens:**
1.  Raw data is read from `data/`.
2.  Data is cleaned and merged into `data/aadhar_enrollment_cleaned.csv`.
3.  ML models (Anomaly, Forecasting, Clustering) execute.
4.  Insight reports are saved as CSVs in `data/`.
5.  40+ visualization images are generated in `images/`.

### 2. Launch the Interactive Dashboard
Start the web interface to explore the data interactively.
```bash
python3 main.py --dashboard
```
*   Or manually: `streamlit run dashboard/app.py`
*   Access the dashboard at: `http://localhost:8501`

### 3. Generate Visualizations Only
If the data is already processed and you just want to regenerate charts:
```bash
python3 main.py --graphs-only
```

---

## 📊 Dashboard Modules

1.  **Overview**: Executive summary, key metrics (Total Enrollments, Growth Rate), and high-level trends.
2.  **Geographic**: State and District performance leaderboards, drill-down tools, and comparative analysis.
3.  **Demographics**: Age group distribution, gender analysis (if available), and population pyramid insights.
4.  **Trends**: Daily/Weekly/Monthly trajectory analysis, seasonality heatmaps, and day-of-week efficiency.
5.  **Data & Export**: Raw data viewer with download capabilities for custom analysis.

---

## 📝 Generated Reports (in `data/`)

*   `anomalies_zscore.csv`: Dates with statistically abnormal enrollment volumes.
*   `forecast_30day.csv`: Daily prediction values for the upcoming month.
*   `district_risk_scores.csv`: Risk assessment for every district (0-100 score).
*   `seasonality_index.csv`: Monthly multipliers indicating peak/off-peak seasons.
*   `recommendations.csv`: Automated strategic advice based on data patterns.
*   `underserved_districts.csv`: List of districts needing targeted intervention.

---

**Developed for UIDAI Hackathon 2026**
*Team LUMEN*
