"""
Time Series Forecasting Module for Aadhaar Enrollment Data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def prepare_time_series(df, column='total_enrollments'):
    """Prepare time series data for forecasting."""
    daily = df.groupby('date')[column].sum().reset_index()
    daily.columns = ['date', 'value']
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').set_index('date')
    return daily


def moving_average_forecast(series, periods=30, window=7):
    """Forecast using simple moving average."""
    print(f"Generating {window}-day MA forecast for {periods} days...")
    last_ma = series['value'].rolling(window=window).mean().iloc[-1]
    future_dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=periods)
    recent_trend = (series['value'].iloc[-1] - series['value'].iloc[-window]) / window
    forecast_values = [last_ma + (i * recent_trend * 0.5) for i in range(1, periods + 1)]
    
    return pd.DataFrame({
        'date': future_dates, 'forecast': forecast_values,
        'lower_bound': [v * 0.85 for v in forecast_values],
        'upper_bound': [v * 1.15 for v in forecast_values]
    }).set_index('date')


def exponential_smoothing_forecast(series, periods=30, alpha=0.3, beta=0.1):
    """Double Exponential Smoothing (Holt's method) forecast."""
    print(f"Generating Holt's forecast for {periods} days...")
    values = series['value'].values
    level, trend = values[0], values[1] - values[0] if len(values) > 1 else 0
    
    for t in range(1, len(values)):
        new_level = alpha * values[t] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
    
    future_dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=periods)
    forecast_values = [max(0, level + (i + 1) * trend) for i in range(periods)]
    
    # Use percentage-based bounds for more realistic intervals
    return pd.DataFrame({
        'date': future_dates, 
        'forecast': forecast_values,
        'lower_bound': [max(0, v * 0.7) for v in forecast_values],
        'upper_bound': [v * 1.3 for v in forecast_values]
    }).set_index('date')


def seasonal_forecast(series, periods=30, seasonal_period=7):
    """Forecast using seasonal decomposition."""
    print(f"Generating seasonal forecast for {periods} days...")
    values = series['value'].values
    n = len(values)
    
    trend = pd.Series(values).rolling(window=seasonal_period, center=True, min_periods=1).mean().values
    detrended = values - trend
    
    seasonal_factors = []
    for i in range(seasonal_period):
        indices = range(i, n, seasonal_period)
        factor = np.mean([detrended[j] for j in indices if j < n])
        seasonal_factors.append(factor)
    seasonal_factors = np.array(seasonal_factors) - np.mean(seasonal_factors)
    
    recent_trend = (trend[-1] - trend[-seasonal_period]) / seasonal_period if n > seasonal_period else 0
    future_dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=periods)
    
    forecast_values = []
    for i in range(periods):
        trend_comp = trend[-1] + (i + 1) * recent_trend
        seasonal_idx = (series.index[-1].dayofweek + i + 1) % seasonal_period
        forecast_values.append(max(0, trend_comp + seasonal_factors[seasonal_idx]))
    
    residual_std = np.std(values - trend)
    
    return pd.DataFrame({
        'date': future_dates, 'forecast': forecast_values,
        'lower_bound': [max(0, v - 1.96 * residual_std) for v in forecast_values],
        'upper_bound': [v + 1.96 * residual_std for v in forecast_values]
    }).set_index('date'), seasonal_factors


def forecast_by_state(df, periods=30, top_n=10):
    """Generate forecasts for top states."""
    print(f"Generating state-wise forecasts...")
    top_states = df.groupby('state')['total_enrollments'].sum().nlargest(top_n).index.tolist()
    state_forecasts = {}
    
    for state in top_states:
        state_series = prepare_time_series(df[df['state'] == state])
        if len(state_series) > 14:
            forecast, _ = seasonal_forecast(state_series, periods=periods)
            state_forecasts[state] = forecast
    return state_forecasts


def project_milestones(df, milestones=[25_000_000, 50_000_000, 100_000_000]):
    """Project when cumulative enrollments will hit milestones."""
    print("Projecting enrollment milestones...")
    daily = df.groupby('date')['total_enrollments'].sum().reset_index()
    daily.columns = ['date', 'value']
    daily = daily.sort_values('date')
    daily['cumulative'] = daily['value'].cumsum()
    
    current = daily['cumulative'].iloc[-1]
    last_date = pd.to_datetime(daily['date'].iloc[-1])
    avg_daily = daily.tail(30)['value'].mean()
    
    projections = []
    for m in milestones:
        if m <= current:
            projections.append({'milestone': m, 'status': 'Achieved'})
        else:
            days = int(np.ceil((m - current) / avg_daily))
            projections.append({
                'milestone': m, 'status': 'Projected',
                'projected_date': last_date + timedelta(days=days), 'days_from_now': days
            })
    return pd.DataFrame(projections), daily


def generate_forecast_report(df, periods=30):
    """Generate comprehensive forecasting report."""
    print("\n" + "="*50 + "\nFORECASTING REPORT\n" + "="*50)
    
    series = prepare_time_series(df)
    results = {
        'time_series': series,
        'ma_forecast': moving_average_forecast(series, periods),
        'exp_forecast': exponential_smoothing_forecast(series, periods),
    }
    
    seasonal_fcst, factors = seasonal_forecast(series, periods)
    results['seasonal_forecast'] = seasonal_fcst
    results['seasonal_factors'] = factors
    results['state_forecasts'] = forecast_by_state(df, periods)
    results['milestones'], results['cumulative'] = project_milestones(df)
    
    print(f"\nNext {periods} days: Avg={seasonal_fcst['forecast'].mean():,.0f}, Total={seasonal_fcst['forecast'].sum():,.0f}")
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
    
    results = generate_forecast_report(df)
    results['seasonal_forecast'].to_csv('forecast_30day.csv')
    results['milestones'].to_csv('milestone_projections.csv', index=False)
    print("\nForecasting complete! Saved forecast_30day.csv, milestone_projections.csv")
    return results


if __name__ == "__main__":
    main()
