"""
Anomaly Detection Module for Aadhaar Enrollment Data
Implements various statistical methods to detect unusual patterns in enrollment data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# STATISTICAL ANOMALY DETECTION
# =============================================================================

def detect_anomalies_zscore(df, column='total_enrollments', threshold=3.0):
    """
    Detect anomalies using Z-score method.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with date and enrollment columns
    column : str
        Column name to analyze for anomalies
    threshold : float
        Z-score threshold (default 3.0 = 99.7% confidence)
    
    Returns:
    --------
    DataFrame with anomaly dates, values, and z-scores
    """
    print("Detecting anomalies using Z-score method...")
    
    # Aggregate daily enrollments
    daily = df.groupby('date')[column].sum().reset_index()
    daily.columns = ['date', 'value']
    
    # Calculate Z-scores
    mean_val = daily['value'].mean()
    std_val = daily['value'].std()
    daily['z_score'] = (daily['value'] - mean_val) / std_val
    daily['is_anomaly'] = np.abs(daily['z_score']) > threshold
    daily['anomaly_type'] = daily.apply(
        lambda x: 'High' if x['z_score'] > threshold else ('Low' if x['z_score'] < -threshold else 'Normal'),
        axis=1
    )
    
    anomalies = daily[daily['is_anomaly']].copy()
    anomalies['deviation_pct'] = ((anomalies['value'] - mean_val) / mean_val * 100).round(2)
    
    print(f"  Found {len(anomalies)} anomalies (threshold: {threshold} sigma)")
    print(f"  - High anomalies: {len(anomalies[anomalies['anomaly_type'] == 'High'])}")
    print(f"  - Low anomalies: {len(anomalies[anomalies['anomaly_type'] == 'Low'])}")
    
    return anomalies, daily


def detect_anomalies_iqr(df, column='total_enrollments', multiplier=1.5):
    """
    Detect anomalies using Interquartile Range (IQR) method.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with date and enrollment columns
    column : str
        Column name to analyze
    multiplier : float
        IQR multiplier (1.5 = standard, 3.0 = extreme outliers only)
    
    Returns:
    --------
    DataFrame with anomaly dates and values
    """
    print("Detecting anomalies using IQR method...")
    
    daily = df.groupby('date')[column].sum().reset_index()
    daily.columns = ['date', 'value']
    
    Q1 = daily['value'].quantile(0.25)
    Q3 = daily['value'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    daily['is_anomaly'] = (daily['value'] < lower_bound) | (daily['value'] > upper_bound)
    daily['anomaly_type'] = daily.apply(
        lambda x: 'High' if x['value'] > upper_bound else ('Low' if x['value'] < lower_bound else 'Normal'),
        axis=1
    )
    
    anomalies = daily[daily['is_anomaly']].copy()
    
    print(f"  IQR bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    print(f"  Found {len(anomalies)} anomalies")
    
    return anomalies, daily, (lower_bound, upper_bound)


def detect_consecutive_anomalies(df, column='total_enrollments', window=7, threshold=2.0, min_consecutive=3):
    """
    Detect consecutive days with sustained deviation from moving average.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    column : str
        Column name to analyze
    window : int
        Moving average window size
    threshold : float
        Standard deviation threshold
    min_consecutive : int
        Minimum consecutive days to flag
    
    Returns:
    --------
    DataFrame with consecutive anomaly periods
    """
    print(f"Detecting consecutive anomalies (min {min_consecutive} days)...")
    
    daily = df.groupby('date')[column].sum().reset_index()
    daily.columns = ['date', 'value']
    daily = daily.sort_values('date')
    
    # Calculate rolling statistics
    daily['ma'] = daily['value'].rolling(window=window, min_periods=1).mean()
    daily['std'] = daily['value'].rolling(window=window, min_periods=1).std()
    daily['deviation'] = (daily['value'] - daily['ma']) / daily['std']
    daily['is_deviant'] = np.abs(daily['deviation']) > threshold
    
    # Find consecutive runs
    daily['run_id'] = (daily['is_deviant'] != daily['is_deviant'].shift()).cumsum()
    
    runs = daily[daily['is_deviant']].groupby('run_id').agg({
        'date': ['min', 'max', 'count'],
        'deviation': 'mean',
        'value': 'mean'
    }).reset_index()
    
    runs.columns = ['run_id', 'start_date', 'end_date', 'duration', 'avg_deviation', 'avg_value']
    consecutive_anomalies = runs[runs['duration'] >= min_consecutive]
    
    print(f"  Found {len(consecutive_anomalies)} consecutive anomaly periods")
    
    return consecutive_anomalies, daily


# =============================================================================
# SEASONAL DECOMPOSITION
# =============================================================================

def seasonal_decomposition_analysis(df, column='total_enrollments', period=7):
    """
    Perform seasonal decomposition using moving average method.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    column : str
        Column to decompose
    period : int
        Seasonality period (7 for weekly, 30 for monthly)
    
    Returns:
    --------
    Trend, seasonal, and residual components
    """
    print(f"Performing seasonal decomposition (period={period})...")
    
    daily = df.groupby('date')[column].sum().reset_index()
    daily.columns = ['date', 'value']
    daily = daily.sort_values('date').set_index('date')
    
    # Calculate trend using centered moving average
    trend = daily['value'].rolling(window=period, center=True, min_periods=1).mean()
    
    # Detrend the series
    detrended = daily['value'] - trend
    
    # Calculate seasonal component (average of each day of week/month)
    daily['day_of_week'] = daily.index.dayofweek
    seasonal_means = detrended.groupby(daily['day_of_week']).mean()
    seasonal = daily['day_of_week'].map(seasonal_means)
    
    # Residual = Original - Trend - Seasonal
    residual = daily['value'] - trend - seasonal
    
    # Detect anomalies in residuals
    residual_std = residual.std()
    residual_mean = residual.mean()
    anomaly_threshold = 2.5 * residual_std
    
    result_df = pd.DataFrame({
        'date': daily.index,
        'original': daily['value'].values,
        'trend': trend.values,
        'seasonal': seasonal.values,
        'residual': residual.values,
        'is_residual_anomaly': np.abs(residual.values - residual_mean) > anomaly_threshold
    })
    
    residual_anomalies = result_df[result_df['is_residual_anomaly']]
    print(f"  Found {len(residual_anomalies)} residual anomalies")
    
    return result_df, residual_anomalies


# =============================================================================
# GEOGRAPHIC ANOMALIES
# =============================================================================

def detect_geographic_anomalies(df, level='state', threshold=2.0):
    """
    Identify states/districts with abnormal enrollment patterns.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    level : str
        'state' or 'district'
    threshold : float
        Standard deviation threshold
    
    Returns:
    --------
    DataFrame with geographic anomalies
    """
    print(f"Detecting geographic anomalies at {level} level...")
    
    # Aggregate by geographic level
    geo_data = df.groupby(level).agg({
        'total_enrollments': 'sum',
        'date': 'nunique',
        'pincode': 'nunique'
    }).reset_index()
    
    geo_data.columns = [level, 'total_enrollments', 'active_days', 'unique_pincodes']
    
    # Calculate per-day and per-pincode averages
    geo_data['enrollments_per_day'] = geo_data['total_enrollments'] / geo_data['active_days']
    geo_data['enrollments_per_pincode'] = geo_data['total_enrollments'] / geo_data['unique_pincodes']
    
    # Calculate Z-scores for both metrics
    for metric in ['enrollments_per_day', 'enrollments_per_pincode']:
        mean_val = geo_data[metric].mean()
        std_val = geo_data[metric].std()
        geo_data[f'{metric}_zscore'] = (geo_data[metric] - mean_val) / std_val
    
    # Flag anomalies
    geo_data['is_high_performer'] = (geo_data['enrollments_per_day_zscore'] > threshold)
    geo_data['is_low_performer'] = (geo_data['enrollments_per_day_zscore'] < -threshold)
    geo_data['is_anomaly'] = geo_data['is_high_performer'] | geo_data['is_low_performer']
    
    anomalies = geo_data[geo_data['is_anomaly']].copy()
    
    print(f"  High performers: {geo_data['is_high_performer'].sum()}")
    print(f"  Low performers: {geo_data['is_low_performer'].sum()}")
    
    return anomalies, geo_data


def detect_pincode_anomalies(df, top_n=50):
    """
    Identify pincodes with unusually high or low enrollments.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    top_n : int
        Number of top/bottom pincodes to return
    
    Returns:
    --------
    DataFrame with pincode anomalies
    """
    print("Detecting pincode-level anomalies...")
    
    pincode_data = df.groupby(['pincode', 'state', 'district']).agg({
        'total_enrollments': 'sum',
        'date': 'nunique'
    }).reset_index()
    
    pincode_data.columns = ['pincode', 'state', 'district', 'total_enrollments', 'active_days']
    pincode_data['daily_avg'] = pincode_data['total_enrollments'] / pincode_data['active_days']
    
    # Calculate percentiles
    pincode_data['percentile'] = pincode_data['total_enrollments'].rank(pct=True) * 100
    
    # Get extremes
    top_pincodes = pincode_data.nlargest(top_n, 'total_enrollments')
    bottom_pincodes = pincode_data.nsmallest(top_n, 'total_enrollments')
    
    print(f"  Top pincode: {top_pincodes.iloc[0]['pincode']} with {top_pincodes.iloc[0]['total_enrollments']:,} enrollments")
    
    return top_pincodes, bottom_pincodes, pincode_data


# =============================================================================
# HOLIDAY/EVENT IMPACT ANALYSIS
# =============================================================================

def get_indian_holidays_2025():
    """
    Return a dictionary of major Indian holidays in 2025.
    """
    return {
        '2025-01-14': 'Makar Sankranti',
        '2025-01-26': 'Republic Day',
        '2025-03-14': 'Holi',
        '2025-04-14': 'Ambedkar Jayanti',
        '2025-04-18': 'Good Friday',
        '2025-05-12': 'Buddha Purnima',
        '2025-08-15': 'Independence Day',
        '2025-08-16': 'Janmashtami',
        '2025-10-02': 'Gandhi Jayanti',
        '2025-10-20': 'Dussehra',
        '2025-11-01': 'Diwali',
        '2025-11-05': 'Bhai Dooj',
        '2025-12-25': 'Christmas',
    }


def analyze_holiday_impact(df, column='total_enrollments'):
    """
    Analyze the impact of holidays on enrollment numbers.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with date column
    column : str
        Column to analyze
    
    Returns:
    --------
    DataFrame with holiday impact analysis
    """
    print("Analyzing holiday impact on enrollments...")
    
    holidays = get_indian_holidays_2025()
    
    daily = df.groupby('date')[column].sum().reset_index()
    daily.columns = ['date', 'value']
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Calculate baseline (non-holiday average)
    daily['is_holiday'] = daily['date'].dt.strftime('%Y-%m-%d').isin(holidays.keys())
    baseline = daily[~daily['is_holiday']]['value'].mean()
    
    # Analyze each holiday
    holiday_impact = []
    for date_str, name in holidays.items():
        date = pd.to_datetime(date_str)
        
        # Get enrollment on holiday
        holiday_data = daily[daily['date'] == date]
        if len(holiday_data) > 0:
            holiday_value = holiday_data['value'].values[0]
            impact_pct = ((holiday_value - baseline) / baseline) * 100
            
            # Get surrounding days for context
            before = daily[(daily['date'] >= date - timedelta(days=3)) & (daily['date'] < date)]['value'].mean()
            after = daily[(daily['date'] > date) & (daily['date'] <= date + timedelta(days=3))]['value'].mean()
            
            holiday_impact.append({
                'date': date,
                'holiday': name,
                'enrollments': holiday_value,
                'baseline': baseline,
                'impact_pct': round(impact_pct, 2),
                'avg_before': before,
                'avg_after': after,
                'recovery_days': 1 if after > before * 0.9 else 2
            })
    
    impact_df = pd.DataFrame(holiday_impact)
    
    if len(impact_df) > 0:
        avg_impact = impact_df['impact_pct'].mean()
        print(f"  Average holiday impact: {avg_impact:.1f}%")
        print(f"  Holidays analyzed: {len(impact_df)}")
    
    return impact_df, baseline


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_anomaly_report(df):
    """
    Generate a comprehensive anomaly detection report.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame
    
    Returns:
    --------
    Dictionary with all anomaly analysis results
    """
    print("\n" + "="*60)
    print("ANOMALY DETECTION REPORT")
    print("="*60 + "\n")
    
    results = {}
    
    # 1. Z-score anomalies
    print("\n[1/6] Z-Score Anomaly Detection")
    results['zscore_anomalies'], results['daily_zscore'] = detect_anomalies_zscore(df)
    
    # 2. IQR anomalies
    print("\n[2/6] IQR Anomaly Detection")
    results['iqr_anomalies'], results['daily_iqr'], results['iqr_bounds'] = detect_anomalies_iqr(df)
    
    # 3. Consecutive anomalies
    print("\n[3/6] Consecutive Anomaly Detection")
    results['consecutive_anomalies'], results['daily_consecutive'] = detect_consecutive_anomalies(df)
    
    # 4. Seasonal decomposition
    print("\n[4/6] Seasonal Decomposition")
    results['decomposition'], results['residual_anomalies'] = seasonal_decomposition_analysis(df)
    
    # 5. Geographic anomalies
    print("\n[5/6] Geographic Anomaly Detection")
    results['geo_anomalies'], results['geo_data'] = detect_geographic_anomalies(df, level='state')
    
    # 6. Holiday impact
    print("\n[6/6] Holiday Impact Analysis")
    results['holiday_impact'], results['baseline'] = analyze_holiday_impact(df)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Z-score anomalies: {len(results['zscore_anomalies'])}")
    print(f"Total IQR anomalies: {len(results['iqr_anomalies'])}")
    print(f"Consecutive anomaly periods: {len(results['consecutive_anomalies'])}")
    print(f"Residual anomalies: {len(results['residual_anomalies'])}")
    print(f"Geographic anomalies: {len(results['geo_anomalies'])}")
    print(f"Holidays analyzed: {len(results['holiday_impact'])}")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run anomaly detection analysis."""
    
    # Load data
    print("Loading cleaned data...")
    try:
        df = pd.read_csv('aadhar_enrollment_cleaned.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df):,} records")
    except FileNotFoundError:
        print("Error: aadhar_enrollment_cleaned.csv not found!")
        print("Please run analysis.py first to generate the cleaned dataset.")
        return None
    
    # Run comprehensive analysis
    results = generate_anomaly_report(df)
    
    # Save key results
    if results['zscore_anomalies'] is not None and len(results['zscore_anomalies']) > 0:
        results['zscore_anomalies'].to_csv('anomalies_zscore.csv', index=False)
        print("\nSaved: anomalies_zscore.csv")
    
    if results['geo_anomalies'] is not None and len(results['geo_anomalies']) > 0:
        results['geo_anomalies'].to_csv('anomalies_geographic.csv', index=False)
        print("Saved: anomalies_geographic.csv")
    
    if results['holiday_impact'] is not None and len(results['holiday_impact']) > 0:
        results['holiday_impact'].to_csv('anomalies_holiday_impact.csv', index=False)
        print("Saved: anomalies_holiday_impact.csv")
    
    print("\nAnomaly detection complete!")
    return results


if __name__ == "__main__":
    results = main()
