"""
MAIN.PY - UIDAI Aadhaar Enrollment Analysis
Master script that orchestrates all analysis modules.

Usage:
    python main.py              # Run complete analysis pipeline
    python main.py --quick      # Run only essential modules
    python main.py --dashboard  # Launch interactive dashboard
"""

import os
import sys
import time
import argparse
from datetime import datetime
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_FILE = 'data/aadhar_enrollment_cleaned.csv'
RAW_FILES = [
    'data/api_data_aadhar_enrolment_0_500000.csv',
    'data/api_data_aadhar_enrolment_500000_1000000.csv',
    'data/api_data_aadhar_enrolment_1000000_1006029.csv'
]


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ██╗   ██╗██╗██████╗  █████╗ ██╗                               ║
║   ██║   ██║██║██╔══██╗██╔══██╗██║                               ║
║   ██║   ██║██║██║  ██║███████║██║                               ║
║   ██║   ██║██║██║  ██║██╔══██║██║                               ║
║   ╚██████╔╝██║██████╔╝██║  ██║██║                               ║
║    ╚═════╝ ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝                               ║
║                                                                  ║
║   AADHAAR ENROLLMENT DATA ANALYSIS SYSTEM                       ║
║   Hackathon 2026 - Team LUMEN                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_section(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_status(status, message):
    """Print status message."""
    icons = {'OK': '[OK]', 'ERROR': '[X]', 'INFO': '[i]', 'SKIP': '[--]'}
    print(f"  {icons.get(status, '[?]')} {message}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load the cleaned data file."""
    print_section("LOADING DATA")
    
    if not os.path.exists(DATA_FILE):
        print_status('INFO', f'{DATA_FILE} not found. Running data cleaning...')
        run_data_cleaning()
    
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        print_status('OK', f'Loaded {len(df):,} records from {DATA_FILE}')
        return df
    except Exception as e:
        print_status('ERROR', f'Failed to load data: {e}')
        return None


def run_data_cleaning():
    """Run the data cleaning pipeline."""
    print_status('INFO', 'Running src.data_processing for data cleaning...')
    
    try:
        from src import data_processing
        data_processing.main()
    except ImportError:
        print_status('ERROR', 'src/data_processing.py not found!')
        sys.exit(1)


# =============================================================================
# MODULE RUNNERS
# =============================================================================

def run_original_graphs(df):
    """Run the original static visualizations."""
    print_section("STATIC VISUALIZATIONS (1-25)")
    
    try:
        from src import static_visualizations
        static_visualizations.generate_all_graphs()
        print_status('OK', '25 static graphs generated')
        return True
    except ImportError:
        print_status('SKIP', 'src/static_visualizations.py not found')
        return False
    except Exception as e:
        print_status('ERROR', f'Graph generation failed: {e}')
        return False


def run_anomaly_detection(df):
    """Run anomaly detection module."""
    print_section("ANOMALY DETECTION")
    
    try:
        from src.anomaly_detection import generate_anomaly_report
        results = generate_anomaly_report(df)
        
        # Save outputs
        if results.get('zscore_anomalies') is not None and len(results['zscore_anomalies']) > 0:
            results['zscore_anomalies'].to_csv('data/anomalies_zscore.csv', index=False)
        if results.get('geo_anomalies') is not None and len(results['geo_anomalies']) > 0:
            results['geo_anomalies'].to_csv('data/anomalies_geographic.csv', index=False)
        if results.get('holiday_impact') is not None and len(results['holiday_impact']) > 0:
            results['holiday_impact'].to_csv('data/anomalies_holiday_impact.csv', index=False)
            
        print_status('OK', 'Anomaly detection complete')
        return results
    except Exception as e:
        print_status('ERROR', f'Anomaly detection failed: {e}')
        return None


def run_forecasting(df):
    """Run forecasting module."""
    print_section("FORECASTING")
    
    try:
        from src.forecasting import generate_forecast_report
        results = generate_forecast_report(df)
        
        # Save outputs
        if results.get('seasonal_forecast') is not None:
            results['seasonal_forecast'].to_csv('data/forecast_30day.csv')
        if results.get('milestones') is not None:
            results['milestones'].to_csv('data/milestone_projections.csv', index=False)
            
        print_status('OK', 'Forecasting complete')
        return results
    except Exception as e:
        print_status('ERROR', f'Forecasting failed: {e}')
        return None


def run_clustering(df):
    """Run clustering module."""
    print_section("CLUSTERING ANALYSIS")
    
    try:
        from src.clustering import generate_clustering_report
        results = generate_clustering_report(df)
        
        # Save outputs
        if results.get('district_clusters') is not None:
            results['district_clusters'].to_csv('data/district_clusters.csv', index=False)
        if results.get('underserved') is not None:
            results['underserved'].to_csv('data/underserved_districts.csv', index=False)
            
        print_status('OK', 'Clustering complete')
        return results
    except Exception as e:
        print_status('ERROR', f'Clustering failed: {e}')
        return None


def run_ml_models(df):
    """Run machine learning models."""
    print_section("MACHINE LEARNING MODELS")
    
    try:
        from src.ml_models import generate_ml_report
        results = generate_ml_report(df)
        
        # Save outputs
        if results.get('district_risk') is not None:
            results['district_risk'].to_csv('data/district_risk_scores.csv', index=False)
        if results.get('feature_importance') is not None:
            results['feature_importance'].to_csv('data/feature_importance.csv', index=False)
            
        print_status('OK', 'ML analysis complete')
        return results
    except Exception as e:
        print_status('ERROR', f'ML analysis failed: {e}')
        return None


def run_operational_insights(df):
    """Run operational insights module."""
    print_section("OPERATIONAL INSIGHTS")
    
    try:
        from src.operational_insights import generate_operational_report
        results = generate_operational_report(df)
        
        # Save outputs
        if results.get('seasonality') is not None:
            results['seasonality'].to_csv('data/seasonality_index.csv', index=False)
        if results.get('recommendations') is not None:
            results['recommendations'].to_csv('data/recommendations.csv', index=False)
        if results.get('capacity') is not None:
            results['capacity'].to_csv('data/capacity_utilization.csv', index=False)
            
        print_status('OK', 'Operational insights complete')
        return results
    except Exception as e:
        print_status('ERROR', f'Operational insights failed: {e}')
        return None


def run_advanced_visualizations(df):
    """Run advanced visualizations module."""
    print_section("ADVANCED VISUALIZATIONS (26-32)")
    
    try:
        from src.advanced_visualizations import generate_advanced_visualizations
        generate_advanced_visualizations(df)
        print_status('OK', '7 advanced visualizations generated')
        return True
    except Exception as e:
        print_status('ERROR', f'Advanced visualizations failed: {e}')
        return False


def run_insight_visualizations():
    """Run insight visualizations from output CSVs."""
    print_section("INSIGHT VISUALIZATIONS (From Output CSVs)")
    
    try:
        from src.insight_visualizations import generate_all_insight_visualizations
        generate_all_insight_visualizations()
        print_status('OK', 'Insight visualizations generated')
        return True
    except Exception as e:
        print_status('ERROR', f'Insight visualizations failed: {e}')
        return False


def run_dashboard():
    """Launch the Streamlit dashboard."""
    print_section("LAUNCHING DASHBOARD")
    
    dashboard_path = os.path.join('dashboard', 'app.py')
    if os.path.exists(dashboard_path):
        print_status('INFO', 'Starting Streamlit server...')
        print_status('INFO', 'Open http://localhost:8501 in your browser')
        os.system(f'streamlit run {dashboard_path}')
    else:
        print_status('ERROR', 'Dashboard app.py not found!')


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary(elapsed_time):
    """Print final summary."""
    print_section("ANALYSIS COMPLETE")
    
    print(f"\n  Total Time: {elapsed_time:.1f} seconds")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n  Generated Output Files:")
    
    output_files = [
        # Data
        ('data/aadhar_enrollment_cleaned.csv', 'Cleaned dataset'),
        # Anomalies
        ('data/anomalies_zscore.csv', 'Statistical anomalies'),
        ('data/anomalies_geographic.csv', 'Geographic outliers'),
        ('data/anomalies_holiday_impact.csv', 'Holiday impact analysis'),
        # Forecasts
        ('data/forecast_30day.csv', '30-day forecast'),
        ('data/milestone_projections.csv', 'Milestone projections'),
        # Clustering
        ('data/district_clusters.csv', 'District cluster assignments'),
        ('data/underserved_districts.csv', 'Underserved regions'),
        # ML
        ('data/district_risk_scores.csv', 'District risk scores'),
        ('data/feature_importance.csv', 'Feature importance'),
        # Operational
        ('data/seasonality_index.csv', 'Monthly seasonality'),
        ('data/recommendations.csv', 'Action recommendations'),
        ('data/capacity_utilization.csv', 'Capacity analysis'),
    ]
    
    for filename, description in output_files:
        if os.path.exists(filename):
            print_status('OK', f'{filename} - {description}')
        else:
            print_status('SKIP', f'{filename}')
    
    print("\n  Generated Visualizations:")
    
    # Count graph files
    # Count graph files in images/ directory
    if os.path.exists('images'):
        original_graphs = len([f for f in os.listdir('images') if f.startswith('graph_') and f.endswith('.png') and int(f.split('_')[1]) <= 25])
        advanced_graphs = len([f for f in os.listdir('images') if f.startswith('graph_') and f.endswith('.png') and int(f.split('_')[1]) > 25])
        insight_graphs = len([f for f in os.listdir('images') if f.startswith('insight_') and f.endswith('.png')])
    else:
        original_graphs = 0
        advanced_graphs = 0
        insight_graphs = 0
    
    print_status('OK', f'{original_graphs} original graphs (graph_01 - graph_25)')
    print_status('OK', f'{advanced_graphs} advanced graphs (graph_26 - graph_32)')
    print_status('OK', f'{insight_graphs} insight graphs (insight_01 - insight_10)')
    
    print("\n  Next Steps:")
    print("    1. Review generated CSV files for insights")
    print("    2. Check images/ folder for all visualizations")
    print("    3. Run 'python main.py --dashboard' for interactive dashboard")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='UIDAI Aadhaar Enrollment Analysis')
    parser.add_argument('--quick', action='store_true', help='Run only essential modules')
    parser.add_argument('--dashboard', action='store_true', help='Launch interactive dashboard')
    parser.add_argument('--graphs-only', action='store_true', help='Generate only visualizations')
    parser.add_argument('--no-graphs', action='store_true', help='Skip graph generation')
    args = parser.parse_args()
    
    # Launch dashboard if requested
    if args.dashboard:
        run_dashboard()
        return
    
    # Start
    start_time = time.time()
    print_banner()
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    if df is None:
        print_status('ERROR', 'Cannot proceed without data!')
        sys.exit(1)
    
    # Graphs only mode
    if args.graphs_only:
        run_original_graphs(df)
        run_advanced_visualizations(df)
        run_insight_visualizations()
        print_summary(time.time() - start_time)
        return
    
    # Run modules
    if not args.no_graphs:
        run_original_graphs(df)
    
    run_anomaly_detection(df)
    run_forecasting(df)
    
    if not args.quick:
        run_clustering(df)
        run_ml_models(df)
        run_operational_insights(df)
        
        if not args.no_graphs:
            run_advanced_visualizations(df)
            run_insight_visualizations()
    
    # Summary
    print_summary(time.time() - start_time)


if __name__ == "__main__":
    main()

