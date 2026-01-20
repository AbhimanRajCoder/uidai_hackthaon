"""
Operational Insights Module for Aadhaar Enrollment Data
KPIs, capacity utilization, efficiency metrics, and recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def format_number(val):
    """Format large numbers."""
    if val >= 1e7:
        return f'{val/1e7:.1f}Cr'
    elif val >= 1e5:
        return f'{val/1e5:.1f}L'
    elif val >= 1e3:
        return f'{val/1e3:.1f}K'
    return f'{val:.0f}'


# =============================================================================
# KEY PERFORMANCE INDICATORS
# =============================================================================

def calculate_overall_kpis(df):
    """Calculate overall operational KPIs."""
    print("Calculating overall KPIs...")
    
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby('date')['total_enrollments'].sum()
    
    kpis = {
        'total_enrollments': df['total_enrollments'].sum(),
        'total_records': len(df),
        'date_range_start': df['date'].min(),
        'date_range_end': df['date'].max(),
        'total_days': df['date'].nunique(),
        'total_states': df['state'].nunique(),
        'total_districts': df['district'].nunique(),
        'total_pincodes': df['pincode'].nunique(),
        
        # Daily metrics
        'avg_daily_enrollments': daily.mean(),
        'median_daily_enrollments': daily.median(),
        'max_daily_enrollments': daily.max(),
        'min_daily_enrollments': daily.min(),
        'std_daily_enrollments': daily.std(),
        
        # Age distribution
        'pct_age_0_5': df['age_0_5'].sum() / df['total_enrollments'].sum() * 100,
        'pct_age_5_17': df['age_5_17'].sum() / df['total_enrollments'].sum() * 100,
        'pct_age_18_plus': df['age_18_greater'].sum() / df['total_enrollments'].sum() * 100,
    }
    
    # Derived KPIs
    kpis['coefficient_of_variation'] = kpis['std_daily_enrollments'] / kpis['avg_daily_enrollments'] * 100
    kpis['enrollments_per_state'] = kpis['total_enrollments'] / kpis['total_states']
    kpis['enrollments_per_district'] = kpis['total_enrollments'] / kpis['total_districts']
    kpis['enrollments_per_pincode'] = kpis['total_enrollments'] / kpis['total_pincodes']
    
    return kpis


def calculate_temporal_kpis(df):
    """Calculate time-based KPIs."""
    print("Calculating temporal KPIs...")
    
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    daily = df.groupby(['date', 'is_weekend'])['total_enrollments'].sum().reset_index()
    
    weekday_avg = daily[~daily['is_weekend']]['total_enrollments'].mean()
    weekend_avg = daily[daily['is_weekend']]['total_enrollments'].mean()
    
    monthly = df.groupby('month')['total_enrollments'].sum()
    dow = df.groupby('day_of_week')['total_enrollments'].sum()
    
    kpis = {
        'weekday_avg': weekday_avg,
        'weekend_avg': weekend_avg,
        'weekend_efficiency': weekend_avg / weekday_avg * 100 if weekday_avg > 0 else 0,
        'best_month': monthly.idxmax(),
        'worst_month': monthly.idxmin(),
        'best_day_of_week': dow.idxmax(),
        'worst_day_of_week': dow.idxmin(),
        'monthly_variance': monthly.std() / monthly.mean() * 100,
    }
    
    return kpis


def calculate_geographic_kpis(df):
    """Calculate geographic KPIs."""
    print("Calculating geographic KPIs...")
    
    state_totals = df.groupby('state')['total_enrollments'].sum()
    district_totals = df.groupby('district')['total_enrollments'].sum()
    
    # Concentration metrics
    top_5_states = state_totals.nlargest(5).sum()
    top_10_states = state_totals.nlargest(10).sum()
    total = state_totals.sum()
    
    # Gini coefficient (inequality)
    sorted_values = np.sort(state_totals.values)
    n = len(sorted_values)
    cumulative = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values)) - (n + 1) * np.sum(sorted_values)) / (n * np.sum(sorted_values))
    
    kpis = {
        'top_state': state_totals.idxmax(),
        'top_state_share': state_totals.max() / total * 100,
        'top_5_states_share': top_5_states / total * 100,
        'top_10_states_share': top_10_states / total * 100,
        'state_gini_coefficient': gini,
        'avg_per_state': state_totals.mean(),
        'avg_per_district': district_totals.mean(),
        'top_district': district_totals.idxmax(),
    }
    
    return kpis


# =============================================================================
# CAPACITY UTILIZATION
# =============================================================================

def estimate_capacity_utilization(df, assumed_centers_per_district=5, capacity_per_center=100):
    """Estimate enrollment center capacity utilization."""
    print("Estimating capacity utilization...")
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Daily by district
    district_daily = df.groupby(['district', 'date'])['total_enrollments'].sum().reset_index()
    
    # Assumed capacity
    daily_capacity = assumed_centers_per_district * capacity_per_center
    district_daily['utilization'] = district_daily['total_enrollments'] / daily_capacity * 100
    
    # Aggregate
    district_util = district_daily.groupby('district').agg({
        'total_enrollments': 'mean',
        'utilization': ['mean', 'max']
    }).reset_index()
    district_util.columns = ['district', 'avg_daily', 'avg_utilization', 'peak_utilization']
    
    # Categorize
    district_util['capacity_status'] = pd.cut(
        district_util['avg_utilization'],
        bins=[0, 50, 80, 100, float('inf')],
        labels=['Under-utilized', 'Optimal', 'Near Capacity', 'Over Capacity']
    )
    
    summary = district_util['capacity_status'].value_counts()
    
    print(f"  Capacity Status Distribution:")
    for status, count in summary.items():
        print(f"    {status}: {count} districts")
    
    return district_util


def analyze_weekday_weekend_efficiency(df):
    """Compare weekday vs weekend operational efficiency."""
    print("Analyzing weekday vs weekend efficiency...")
    
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['day_name'] = df['date'].dt.day_name()
    
    daily = df.groupby(['date', 'is_weekend', 'day_name'])['total_enrollments'].sum().reset_index()
    
    weekday_data = daily[~daily['is_weekend']]
    weekend_data = daily[daily['is_weekend']]
    
    comparison = {
        'weekday': {
            'avg': weekday_data['total_enrollments'].mean(),
            'total': weekday_data['total_enrollments'].sum(),
            'days': len(weekday_data),
        },
        'weekend': {
            'avg': weekend_data['total_enrollments'].mean(),
            'total': weekend_data['total_enrollments'].sum(),
            'days': len(weekend_data),
        }
    }
    
    comparison['efficiency_gap'] = (comparison['weekday']['avg'] - comparison['weekend']['avg']) / comparison['weekday']['avg'] * 100
    comparison['weekend_potential'] = (comparison['weekday']['avg'] - comparison['weekend']['avg']) * comparison['weekend']['days']
    
    # Day-wise breakdown
    day_breakdown = daily.groupby('day_name')['total_enrollments'].mean().sort_values(ascending=False)
    
    print(f"  Weekday avg: {format_number(comparison['weekday']['avg'])}")
    print(f"  Weekend avg: {format_number(comparison['weekend']['avg'])}")
    print(f"  Efficiency gap: {comparison['efficiency_gap']:.1f}%")
    print(f"  Weekend potential: {format_number(comparison['weekend_potential'])}")
    
    return comparison, day_breakdown


# =============================================================================
# SEASONALITY ANALYSIS
# =============================================================================

def calculate_seasonality_index(df):
    """Calculate seasonality index for resource planning."""
    print("Calculating seasonality index...")
    
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name().str[:3]
    
    monthly = df.groupby(['month', 'month_name'])['total_enrollments'].sum().reset_index()
    overall_avg = monthly['total_enrollments'].mean()
    
    monthly['seasonality_index'] = (monthly['total_enrollments'] / overall_avg * 100).round(2)
    monthly.rename(columns={'total_enrollments': 'total'}, inplace=True)
    
    monthly['recommendation'] = monthly['seasonality_index'].apply(
        lambda x: 'Reduce Staff' if x < 90 else ('Add Staff' if x > 110 else 'Normal')
    )
    
    peak_idx = monthly['seasonality_index'].idxmax()
    low_idx = monthly['seasonality_index'].idxmin()
    
    print(f"  Peak month: {monthly.loc[peak_idx, 'month_name']} ({monthly.loc[peak_idx, 'seasonality_index']:.0f})")
    print(f"  Low month: {monthly.loc[low_idx, 'month_name']} ({monthly.loc[low_idx, 'seasonality_index']:.0f})")
    
    return monthly


# =============================================================================
# RECOMMENDATIONS ENGINE
# =============================================================================

def generate_recommendations(df, kpis):
    """Generate actionable recommendations based on analysis."""
    print("\nGenerating recommendations...")
    
    recommendations = []
    
    # Weekend efficiency
    if kpis.get('weekend_efficiency', 100) < 70:
        recommendations.append({
            'category': 'Operations',
            'priority': 'High',
            'title': 'Improve Weekend Operations',
            'description': f"Weekend enrollments are only {kpis['weekend_efficiency']:.0f}% of weekday levels. Consider extended hours or awareness campaigns.",
            'potential_impact': 'Increase weekly enrollments by 10-15%'
        })
    
    # Geographic concentration
    if kpis.get('top_5_states_share', 0) > 60:
        recommendations.append({
            'category': 'Geographic',
            'priority': 'Medium',
            'title': 'Expand Reach in Underserved States',
            'description': f"Top 5 states contribute {kpis['top_5_states_share']:.0f}% of enrollments. Focus on increasing capacity in smaller states.",
            'potential_impact': 'More balanced national coverage'
        })
    
    # Volatility
    if kpis.get('coefficient_of_variation', 0) > 30:
        recommendations.append({
            'category': 'Planning',
            'priority': 'Medium',
            'title': 'Implement Demand Smoothing',
            'description': f"High daily variation (CV={kpis['coefficient_of_variation']:.0f}%) indicates unpredictable demand. Consider appointment systems.",
            'potential_impact': 'Reduce wait times and improve resource utilization'
        })
    
    # Age distribution
    if kpis.get('pct_age_0_5', 0) < 25:
        recommendations.append({
            'category': 'Outreach',
            'priority': 'Medium',
            'title': 'Child Enrollment Campaigns',
            'description': f"Children (0-5) represent only {kpis['pct_age_0_5']:.1f}% of enrollments. Partner with schools and hospitals.",
            'potential_impact': 'Increase child enrollment coverage'
        })
    
    recommendations.append({
        'category': 'Technology',
        'priority': 'High',
        'title': 'Deploy Predictive Demand System',
        'description': 'Use ML forecasting to predict daily demand and optimize staffing.',
        'potential_impact': 'Reduce wait times by 20-30%'
    })
    
    recommendations.append({
        'category': 'Monitoring',
        'priority': 'Low',
        'title': 'Implement Real-Time Dashboard',
        'description': 'Deploy interactive dashboard for monitoring KPIs across all centers.',
        'potential_impact': 'Faster response to operational issues'
    })
    
    return pd.DataFrame(recommendations)


# =============================================================================
# COMPREHENSIVE REPORT
# =============================================================================

def generate_operational_report(df):
    """Generate comprehensive operational insights report."""
    print("\n" + "="*50 + "\nOPERATIONAL INSIGHTS REPORT\n" + "="*50)
    
    results = {}
    
    # Overall KPIs
    print("\n[1/6] Overall KPIs")
    results['overall_kpis'] = calculate_overall_kpis(df)
    kpis = results['overall_kpis']
    print(f"  Total Enrollments: {format_number(kpis['total_enrollments'])}")
    print(f"  Daily Average: {format_number(kpis['avg_daily_enrollments'])}")
    print(f"  States: {kpis['total_states']}, Districts: {kpis['total_districts']}")
    
    # Temporal KPIs
    print("\n[2/6] Temporal KPIs")
    results['temporal_kpis'] = calculate_temporal_kpis(df)
    
    # Geographic KPIs
    print("\n[3/6] Geographic KPIs")
    results['geographic_kpis'] = calculate_geographic_kpis(df)
    
    # Capacity utilization
    print("\n[4/6] Capacity Utilization")
    results['capacity'] = estimate_capacity_utilization(df)
    
    # Weekday/Weekend analysis
    print("\n[5/6] Weekday/Weekend Analysis")
    results['efficiency'], results['day_breakdown'] = analyze_weekday_weekend_efficiency(df)
    
    # Seasonality
    print("\n[6/6] Seasonality Analysis")
    results['seasonality'] = calculate_seasonality_index(df)
    
    # Merge all KPIs for recommendations
    all_kpis = {**results['overall_kpis'], **results['temporal_kpis'], **results['geographic_kpis']}
    results['recommendations'] = generate_recommendations(df, all_kpis)
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    for _, rec in results['recommendations'].iterrows():
        print(f"\n[{rec['priority']}] {rec['title']}")
        print(f"  {rec['description']}")
    
    return results


def main():
    """Main function."""
    print("Loading data...")
    try:
        df = pd.read_csv('aadhar_enrollment_cleaned.csv')
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print("Error: Run analysis.py first!")
        return None
    
    results = generate_operational_report(df)
    
    # Save results
    results['seasonality'].to_csv('seasonality_index.csv', index=False)
    results['recommendations'].to_csv('recommendations.csv', index=False)
    results['capacity'].to_csv('capacity_utilization.csv', index=False)
    
    print("\nSaved: seasonality_index.csv, recommendations.csv, capacity_utilization.csv")
    
    return results


if __name__ == "__main__":
    main()
