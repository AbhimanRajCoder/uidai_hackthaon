"""
Machine Learning Models for Aadhaar Enrollment Data
Classification, regression, and feature importance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def prepare_ml_features(df):
    """Prepare features for ML models."""
    print("Preparing ML features...")
    
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    
    # Time features
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['day_of_month'] = data['date'].dt.day
    data['quarter'] = data['date'].dt.quarter
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['is_month_start'] = (data['day_of_month'] <= 5).astype(int)
    data['is_month_end'] = (data['day_of_month'] >= 26).astype(int)
    
    # Encode categorical
    le_state = LabelEncoder()
    data['state_encoded'] = le_state.fit_transform(data['state'].fillna('Unknown'))
    
    return data, le_state


def train_demand_predictor(df, target='total_enrollments'):
    """Train model to predict enrollment demand using time-series split."""
    print("\nTraining demand prediction model...")
    
    data, _ = prepare_ml_features(df)
    
    # Aggregate by date and sort chronologically
    daily = data.groupby('date').agg({
        'total_enrollments': 'sum',
        'day_of_week': 'first',
        'month': 'first',
        'quarter': 'first',
        'is_weekend': 'first',
        'is_month_start': 'first',
        'is_month_end': 'first'
    }).reset_index().sort_values('date')
    
    # Add lag features
    daily['lag_1'] = daily['total_enrollments'].shift(1)
    daily['lag_7'] = daily['total_enrollments'].shift(7)
    daily['lag_14'] = daily['total_enrollments'].shift(14)
    daily['rolling_7_mean'] = daily['total_enrollments'].rolling(7).mean()
    daily['rolling_14_mean'] = daily['total_enrollments'].rolling(14).mean()
    daily = daily.dropna()
    
    feature_cols = ['day_of_week', 'month', 'quarter', 'is_weekend', 
                    'is_month_start', 'is_month_end', 'lag_1', 'lag_7', 'lag_14',
                    'rolling_7_mean', 'rolling_14_mean']
    
    X = daily[feature_cols]
    y = daily['total_enrollments']
    
    # TIME-SERIES SPLIT (not random) for proper evaluation
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Use more trees and tuned parameters
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=10,
        min_samples_split=5,
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  MAE: {mae:,.0f}")
    print(f"  R2 Score: {r2:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  Feature Importance:")
    for _, row in importance.head().iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return model, importance, {'mae': mae, 'r2': r2}


def train_surge_classifier(df, threshold_percentile=90):
    """Train classifier to predict surge days."""
    print("\nTraining surge day classifier...")
    
    data, _ = prepare_ml_features(df)
    
    daily = data.groupby('date').agg({
        'total_enrollments': 'sum',
        'day_of_week': 'first',
        'month': 'first',
        'quarter': 'first',
        'is_weekend': 'first'
    }).reset_index()
    
    # Define surge
    threshold = daily['total_enrollments'].quantile(threshold_percentile / 100)
    daily['is_surge'] = (daily['total_enrollments'] >= threshold).astype(int)
    
    # Add lag features
    daily['lag_1'] = daily['total_enrollments'].shift(1)
    daily['rolling_3_mean'] = daily['total_enrollments'].rolling(3).mean()
    daily = daily.dropna()
    
    feature_cols = ['day_of_week', 'month', 'quarter', 'is_weekend', 'lag_1', 'rolling_3_mean']
    
    X = daily[feature_cols]
    y = daily['is_surge']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(f"  Surge threshold: {threshold:,.0f} enrollments")
    print(f"  Surge days in data: {daily['is_surge'].sum()}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Surge']))
    
    return model, threshold


def train_district_risk_scorer(df):
    """Score districts by service delivery risk using quantile-based scoring."""
    print("\nCalculating district risk scores...")
    
    # Filter out invalid state/district entries (numeric values)
    df_clean = df[df['state'].apply(lambda x: isinstance(x, str) and not str(x).replace('.','').isdigit())].copy()
    df_clean = df_clean[df_clean['district'].apply(lambda x: isinstance(x, str) and not str(x).replace('.','').isdigit())]
    
    district_data = df_clean.groupby(['state', 'district']).agg({
        'total_enrollments': ['sum', 'mean', 'std'],
        'date': 'nunique',
        'pincode': 'nunique'
    }).reset_index()
    
    district_data.columns = ['state', 'district', 'total', 'daily_mean', 'daily_std', 'active_days', 'pincodes']
    
    # Risk factors (normalized to 0-1 scale)
    district_data['cv'] = district_data['daily_std'] / district_data['daily_mean']  # Volatility
    district_data['cv'] = district_data['cv'].clip(upper=district_data['cv'].quantile(0.95))  # Cap outliers
    district_data['cv_normalized'] = (district_data['cv'] - district_data['cv'].min()) / (district_data['cv'].max() - district_data['cv'].min())
    
    district_data['coverage'] = district_data['pincodes'] / district_data['pincodes'].max()
    district_data['volume_score'] = district_data['total'] / district_data['total'].max()
    
    # Composite risk score (higher = more risk) - balanced formula
    district_data['risk_score'] = (
        district_data['cv_normalized'].fillna(0.5) * 0.35 +  # High volatility = risk
        (1 - district_data['coverage']) * 0.35 +  # Low coverage = risk
        (1 - district_data['volume_score']) * 0.30  # Low volume = risk
    )
    
    # QUANTILE-BASED BINNING for balanced distribution
    district_data['risk_category'] = pd.qcut(
        district_data['risk_score'].rank(method='first'),
        q=3,
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    risk_summary = district_data.groupby('risk_category', observed=True).size()
    print(f"  Risk Distribution:")
    for cat, count in risk_summary.items():
        print(f"    {cat}: {count} districts")
    
    return district_data.sort_values('risk_score', ascending=False)


def analyze_feature_importance(df):
    """Analyze feature importance for enrollments."""
    print("\nAnalyzing feature importance...")
    
    data, _ = prepare_ml_features(df)
    
    # Aggregate features
    daily = data.groupby('date').agg({
        'total_enrollments': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum',
        'day_of_week': 'first',
        'month': 'first',
        'is_weekend': 'first'
    }).reset_index()
    
    # Age ratios
    total = daily['age_0_5'] + daily['age_5_17'] + daily['age_18_greater']
    daily['pct_children'] = daily['age_0_5'] / total * 100
    daily['pct_youth'] = daily['age_5_17'] / total * 100
    
    feature_cols = ['day_of_week', 'month', 'is_weekend', 'pct_children', 'pct_youth']
    
    X = daily[feature_cols].fillna(0)
    y = daily['total_enrollments']
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("  Key Drivers of Enrollment:")
    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"    {row['feature']}: {row['importance']:.3f} {bar}")
    
    return importance


def generate_ml_report(df):
    """Generate comprehensive ML analysis report."""
    print("\n" + "="*50 + "\nMACHINE LEARNING REPORT\n" + "="*50)
    
    results = {}
    
    print("\n[1/4] Demand Prediction Model")
    results['demand_model'], results['demand_importance'], results['demand_metrics'] = train_demand_predictor(df)
    
    print("\n[2/4] Surge Classification Model")
    results['surge_model'], results['surge_threshold'] = train_surge_classifier(df)
    
    print("\n[3/4] District Risk Scoring")
    results['district_risk'] = train_district_risk_scorer(df)
    
    print("\n[4/4] Feature Importance Analysis")
    results['feature_importance'] = analyze_feature_importance(df)
    
    print("\n" + "="*50 + "\nML ANALYSIS COMPLETE\n" + "="*50)
    
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
    
    results = generate_ml_report(df)
    
    # Save results
    results['district_risk'].to_csv('district_risk_scores.csv', index=False)
    results['feature_importance'].to_csv('feature_importance.csv', index=False)
    print("\nSaved: district_risk_scores.csv, feature_importance.csv")
    
    return results


if __name__ == "__main__":
    main()
