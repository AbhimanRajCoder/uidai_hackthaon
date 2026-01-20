"""
Clustering Module for Aadhaar Enrollment Data
Geographic and behavioral clustering for service optimization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def prepare_clustering_features(df, level='district'):
    """Prepare features for clustering analysis."""
    print(f"Preparing clustering features at {level} level...")
    
    # Filter out invalid state/district entries (numeric values, nulls)
    df_clean = df.copy()
    df_clean = df_clean[df_clean['state'].apply(lambda x: isinstance(x, str) and not str(x).replace('.','').isdigit())]
    df_clean = df_clean[df_clean['district'].apply(lambda x: isinstance(x, str) and not str(x).replace('.','').isdigit())]
    
    agg_funcs = {
        'total_enrollments': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum', 
        'age_18_greater': 'sum',
        'date': 'nunique',
        'pincode': 'nunique'
    }
    
    if level == 'district':
        grouped = df_clean.groupby(['state', 'district']).agg(agg_funcs).reset_index()
        grouped.rename(columns={'date': 'active_days', 'pincode': 'unique_pincodes'}, inplace=True)
    else:
        grouped = df_clean.groupby('state').agg(agg_funcs).reset_index()
        grouped.rename(columns={'date': 'active_days', 'pincode': 'unique_pincodes'}, inplace=True)
    
    # Calculate derived features
    total = grouped['total_enrollments']
    grouped['pct_children'] = (grouped['age_0_5'] / total * 100).fillna(0)
    grouped['pct_youth'] = (grouped['age_5_17'] / total * 100).fillna(0)
    grouped['pct_adults'] = (grouped['age_18_greater'] / total * 100).fillna(0)
    grouped['daily_avg'] = grouped['total_enrollments'] / grouped['active_days']
    grouped['per_pincode_avg'] = grouped['total_enrollments'] / grouped['unique_pincodes']
    
    return grouped


def find_optimal_clusters(features, max_k=10):
    """Find optimal number of clusters using silhouette score."""
    print("Finding optimal number of clusters...")
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    scores = []
    for k in range(2, min(max_k + 1, len(features))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        scores.append({'k': k, 'silhouette': score})
    
    scores_df = pd.DataFrame(scores)
    optimal_k = scores_df.loc[scores_df['silhouette'].idxmax(), 'k']
    print(f"  Optimal clusters: {int(optimal_k)}")
    
    return int(optimal_k), scores_df


def cluster_regions(df, level='district', n_clusters=None):
    """Cluster regions based on enrollment patterns."""
    print(f"\nClustering {level}s by enrollment patterns...")
    
    data = prepare_clustering_features(df, level)
    
    feature_cols = ['pct_children', 'pct_youth', 'pct_adults', 'daily_avg', 'per_pincode_avg']
    features = data[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    if n_clusters is None:
        n_clusters, _ = find_optimal_clusters(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(scaled)
    
    # Cluster profiles - use a column that always exists
    count_col = 'district' if level == 'district' else 'state'
    profiles = data.groupby('cluster').agg({
        'total_enrollments': ['sum', 'mean'],
        'pct_children': 'mean',
        'pct_youth': 'mean',
        'pct_adults': 'mean',
        'daily_avg': 'mean',
        count_col: 'count'
    }).round(2)
    
    profiles.columns = ['total_sum', 'total_mean', 'avg_pct_children', 
                        'avg_pct_youth', 'avg_pct_adults', 'avg_daily', 'count']
    
    # Name clusters
    cluster_names = []
    for idx, row in profiles.iterrows():
        if row['avg_daily'] > profiles['avg_daily'].quantile(0.75):
            name = 'High Volume'
        elif row['avg_daily'] < profiles['avg_daily'].quantile(0.25):
            name = 'Low Volume'
        elif row['avg_pct_children'] > 40:
            name = 'Child-Focused'
        elif row['avg_pct_adults'] > 50:
            name = 'Adult-Dominated'
        else:
            name = 'Balanced'
        cluster_names.append(name)
    
    profiles['cluster_name'] = cluster_names
    data['cluster_name'] = data['cluster'].map(dict(enumerate(cluster_names)))
    
    print(f"  Created {n_clusters} clusters")
    for i, name in enumerate(cluster_names):
        count = len(data[data['cluster'] == i])
        print(f"    Cluster {i} ({name}): {count} {level}s")
    
    return data, profiles, kmeans


def identify_underserved_regions(df, population_data=None):
    """Identify underserved regions based on enrollment patterns."""
    print("\nIdentifying underserved regions...")
    
    data = prepare_clustering_features(df, level='district')
    
    # Calculate underserved score
    data['daily_avg_normalized'] = (data['daily_avg'] - data['daily_avg'].min()) / \
                                   (data['daily_avg'].max() - data['daily_avg'].min())
    data['coverage_score'] = data['unique_pincodes'] / data['unique_pincodes'].max()
    
    # Lower score = more underserved
    data['service_score'] = (data['daily_avg_normalized'] + data['coverage_score']) / 2
    data['is_underserved'] = data['service_score'] < data['service_score'].quantile(0.25)
    
    underserved = data[data['is_underserved']].sort_values('service_score')
    
    print(f"  Found {len(underserved)} underserved districts")
    print(f"  Top 5 most underserved:")
    for _, row in underserved.head().iterrows():
        print(f"    - {row['district']}, {row['state']}: score={row['service_score']:.3f}")
    
    return underserved, data


def pincode_clustering(df, n_clusters=5):
    """Cluster pincodes for service zone optimization."""
    print("\nClustering pincodes for service zones...")
    
    pincode_data = df.groupby(['pincode', 'state', 'district']).agg({
        'total_enrollments': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum',
        'date': 'nunique'
    }).reset_index()
    
    pincode_data['daily_avg'] = pincode_data['total_enrollments'] / pincode_data['date']
    total = pincode_data['total_enrollments']
    pincode_data['pct_children'] = (pincode_data['age_0_5'] / total * 100).fillna(0)
    pincode_data['pct_youth'] = (pincode_data['age_5_17'] / total * 100).fillna(0)
    
    features = pincode_data[['daily_avg', 'pct_children', 'pct_youth']].fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pincode_data['cluster'] = kmeans.fit_predict(scaled)
    
    cluster_summary = pincode_data.groupby('cluster').agg({
        'pincode': 'count',
        'total_enrollments': 'sum',
        'daily_avg': 'mean'
    }).round(2)
    
    print(f"  Created {n_clusters} pincode clusters")
    
    return pincode_data, cluster_summary


def generate_clustering_report(df):
    """Generate comprehensive clustering report."""
    print("\n" + "="*50 + "\nCLUSTERING REPORT\n" + "="*50)
    
    results = {}
    
    # State clustering
    print("\n[1/4] State-level clustering")
    results['state_clusters'], results['state_profiles'], _ = cluster_regions(df, level='state', n_clusters=4)
    
    # District clustering
    print("\n[2/4] District-level clustering")
    results['district_clusters'], results['district_profiles'], _ = cluster_regions(df, level='district', n_clusters=5)
    
    # Underserved identification
    print("\n[3/4] Underserved region identification")
    results['underserved'], results['all_districts'] = identify_underserved_regions(df)
    
    # Pincode clustering
    print("\n[4/4] Pincode clustering")
    results['pincode_clusters'], results['pincode_summary'] = pincode_clustering(df)
    
    print("\n" + "="*50 + "\nCLUSTERING COMPLETE\n" + "="*50)
    
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
    
    results = generate_clustering_report(df)
    
    # Save results
    results['district_clusters'].to_csv('district_clusters.csv', index=False)
    results['underserved'].to_csv('underserved_districts.csv', index=False)
    print("\nSaved: district_clusters.csv, underserved_districts.csv")
    
    return results


if __name__ == "__main__":
    main()
