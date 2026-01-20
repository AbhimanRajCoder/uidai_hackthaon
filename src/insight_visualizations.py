"""
Output Data Visualizations - Unique Graphs for Analysis Results
Generates visualizations from all output CSV files with insights naming.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Premium style settings
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#333333',
    'axes.titlecolor': '#1a1a1a',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.dpi': 150,
})

# Color palette
PALETTE = {
    'blue': '#1a73e8',
    'green': '#34a853',
    'red': '#ea4335',
    'yellow': '#fbbc04',
    'purple': '#9334e6',
    'teal': '#00bcd4',
    'orange': '#ff5722',
    'pink': '#e91e63',
    'indigo': '#3f51b5',
    'cyan': '#00acc1'
}


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
# ANOMALY VISUALIZATIONS
# =============================================================================

def viz_zscore_anomalies():
    """Visualize Z-score anomalies timeline."""
    if not os.path.exists('data/anomalies_zscore.csv'):
        print("  [SKIP] data/anomalies_zscore.csv not found")
        return
    
    df = pd.read_csv('data/anomalies_zscore.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot anomaly points
    colors = [PALETTE['red'] if t == 'High' else PALETTE['blue'] for t in df['anomaly_type']]
    sizes = np.abs(df['z_score']) * 100
    
    scatter = ax.scatter(df['date'], df['value'], c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=2)
    
    # Add z-score labels
    for _, row in df.iterrows():
        ax.annotate(f"z={row['z_score']:.1f}", 
                   (row['date'], row['value']),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontsize=8, color=PALETTE['red'])
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Enrollment Value', fontsize=12)
    ax.set_title('INSIGHT: Statistical Anomalies Detected (Z-Score > 3σ)', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))
    
    # Legend
    high = mpatches.Patch(color=PALETTE['red'], label='High Anomaly (Surge)')
    low = mpatches.Patch(color=PALETTE['blue'], label='Low Anomaly (Dip)')
    ax.legend(handles=[high, low], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('images/insight_01_anomaly_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_01_anomaly_timeline.png")


def viz_geographic_anomalies():
    """Visualize geographic performance anomalies."""
    if not os.path.exists('data/anomalies_geographic.csv'):
        print("  [SKIP] data/anomalies_geographic.csv not found")
        return
    
    df = pd.read_csv('data/anomalies_geographic.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # High performers
    high = df[df['is_high_performer'] == True].nlargest(10, 'enrollments_per_day')
    ax1 = axes[0]
    bars1 = ax1.barh(high['state'], high['enrollments_per_day'], color=PALETTE['green'], alpha=0.8)
    ax1.set_xlabel('Daily Avg Enrollments', fontsize=11)
    ax1.set_title('Top Performing States', fontsize=12, fontweight='bold', color=PALETTE['green'])
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))
    
    # Low performers
    low = df[df['is_low_performer'] == True].nsmallest(10, 'enrollments_per_day')
    if len(low) > 0:
        ax2 = axes[1]
        bars2 = ax2.barh(low['state'], low['enrollments_per_day'], color=PALETTE['red'], alpha=0.8)
        ax2.set_xlabel('Daily Avg Enrollments', fontsize=11)
        ax2.set_title('Underperforming States', fontsize=12, fontweight='bold', color=PALETTE['red'])
    else:
        axes[1].text(0.5, 0.5, 'No Low Performers Found', ha='center', va='center', fontsize=12)
        axes[1].axis('off')
    
    fig.suptitle('INSIGHT: Geographic Performance Outliers', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('images/insight_02_geographic_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_02_geographic_performance.png")


def viz_holiday_impact():
    """Visualize holiday impact on enrollments."""
    if not os.path.exists('data/anomalies_holiday_impact.csv'):
        print("  [SKIP] data/anomalies_holiday_impact.csv not found")
        return
    
    df = pd.read_csv('data/anomalies_holiday_impact.csv')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create bar chart
    x = range(len(df))
    colors = [PALETTE['red'] if imp < 0 else PALETTE['green'] for imp in df['impact_pct']]
    
    bars = ax.bar(x, df['impact_pct'], color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add labels
    for i, (bar, holiday) in enumerate(zip(bars, df['holiday'])):
        impact = df['impact_pct'].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2, impact + (2 if impact >= 0 else -5),
               f'{impact:.1f}%', ha='center', va='bottom' if impact >= 0 else 'top',
               fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df['holiday'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Impact on Enrollments (%)', fontsize=12)
    ax.set_title('INSIGHT: Holiday Impact on Enrollment Numbers', fontsize=14, fontweight='bold')
    
    # Add average impact
    avg_impact = df['impact_pct'].mean()
    ax.axhline(y=avg_impact, color=PALETTE['purple'], linestyle='--', label=f'Avg Impact: {avg_impact:.1f}%')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('images/insight_03_holiday_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_03_holiday_impact.png")


# =============================================================================
# FORECAST VISUALIZATIONS
# =============================================================================

def viz_forecast_30day():
    """Visualize 30-day enrollment forecast."""
    if not os.path.exists('data/forecast_30day.csv'):
        print("  [SKIP] data/forecast_30day.csv not found")
        return
    
    df = pd.read_csv('data/forecast_30day.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Confidence band
    ax.fill_between(df['date'], df['lower_bound'], df['upper_bound'], 
                   alpha=0.2, color=PALETTE['blue'], label='95% Confidence')
    
    # Forecast line
    ax.plot(df['date'], df['forecast'], color=PALETTE['blue'], linewidth=2.5, 
           marker='o', markersize=4, label='Forecast')
    
    # Highlight weekends
    for _, row in df.iterrows():
        if row['date'].dayofweek >= 5:
            ax.axvspan(row['date'] - pd.Timedelta(hours=12), 
                      row['date'] + pd.Timedelta(hours=12), 
                      alpha=0.1, color='gray')
    
    # Stats annotation
    avg_forecast = df['forecast'].mean()
    total_forecast = df['forecast'].sum()
    
    textstr = f'30-Day Forecast\nTotal: {format_number(total_forecast)}\nDaily Avg: {format_number(avg_forecast)}'
    props = dict(boxstyle='round', facecolor=PALETTE['blue'], alpha=0.1)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, fontweight='bold')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Predicted Enrollments', fontsize=12)
    ax.set_title('INSIGHT: 30-Day Enrollment Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))
    
    plt.tight_layout()
    plt.savefig('images/insight_04_forecast_30day.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_04_forecast_30day.png")


def viz_milestone_projections():
    """Visualize milestone projection timeline."""
    if not os.path.exists('data/milestone_projections.csv'):
        print("  [SKIP] data/milestone_projections.csv not found")
        return
    
    df = pd.read_csv('data/milestone_projections.csv')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create timeline
    y_positions = range(len(df))
    colors = [PALETTE['green'] if s == 'Achieved' else PALETTE['blue'] for s in df['status']]
    
    for i, row in df.iterrows():
        milestone_label = f"{row['milestone']/1e6:.0f}M" if row['milestone'] >= 1e6 else f"{row['milestone']/1e3:.0f}K"
        
        # Draw circle
        circle = plt.Circle((0.3, i), 0.15, color=colors[i], alpha=0.8)
        ax.add_patch(circle)
        
        # Milestone label
        ax.text(0.3, i, milestone_label, ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
        
        # Status and date
        status_text = row['status']
        if 'days_from_now' in df.columns and pd.notna(row.get('days_from_now')):
            status_text += f" ({int(row['days_from_now'])} days)"
        
        ax.text(0.6, i, status_text, ha='left', va='center', fontsize=11,
               color=colors[i], fontweight='bold')
    
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('INSIGHT: Enrollment Milestone Projections', fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    achieved = mpatches.Patch(color=PALETTE['green'], label='Achieved')
    projected = mpatches.Patch(color=PALETTE['blue'], label='Projected')
    ax.legend(handles=[achieved, projected], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('images/insight_05_milestones.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_05_milestones.png")


# =============================================================================
# RISK & ML VISUALIZATIONS
# =============================================================================

def viz_district_risk():
    """Visualize district risk distribution."""
    if not os.path.exists('data/district_risk_scores.csv'):
        print("  [SKIP] data/district_risk_scores.csv not found")
        return
    
    df = pd.read_csv('data/district_risk_scores.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Risk distribution pie
    ax1 = axes[0]
    risk_counts = df['risk_category'].value_counts()
    colors_map = {'Low Risk': PALETTE['green'], 'Medium Risk': PALETTE['yellow'], 'High Risk': PALETTE['red']}
    colors = [colors_map.get(cat, PALETTE['blue']) for cat in risk_counts.index]
    
    wedges, texts, autotexts = ax1.pie(risk_counts.values, labels=risk_counts.index, 
                                        colors=colors, autopct='%1.1f%%',
                                        pctdistance=0.75, startangle=90)
    
    centre_circle = plt.Circle((0, 0), 0.5, fc='white')
    ax1.add_patch(centre_circle)
    ax1.set_title('District Risk Distribution', fontsize=12, fontweight='bold')
    
    # Top high-risk districts
    ax2 = axes[1]
    high_risk = df[df['risk_category'] == 'High Risk'].nlargest(15, 'risk_score')
    
    if len(high_risk) > 0:
        bars = ax2.barh(range(len(high_risk)), high_risk['risk_score'], color=PALETTE['red'], alpha=0.7)
        ax2.set_yticks(range(len(high_risk)))
        ax2.set_yticklabels(high_risk['district'].str[:20], fontsize=9)
        ax2.set_xlabel('Risk Score', fontsize=11)
        ax2.set_title('Top 15 High-Risk Districts', fontsize=12, fontweight='bold', color=PALETTE['red'])
    
    fig.suptitle('INSIGHT: District Service Delivery Risk Assessment', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('images/insight_06_district_risk.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_06_district_risk.png")


def viz_feature_importance():
    """Visualize feature importance analysis."""
    if not os.path.exists('data/feature_importance.csv'):
        print("  [SKIP] data/feature_importance.csv not found")
        return
    
    df = pd.read_csv('data/feature_importance.csv')
    df = df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Horizontal bar chart
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))
    bars = ax.barh(df['feature'], df['importance'], color=colors, edgecolor='white', linewidth=1)
    
    # Add percentage labels
    for bar, imp in zip(bars, df['importance']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{imp*100:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('INSIGHT: Key Drivers of Enrollment Demand', fontsize=14, fontweight='bold')
    ax.set_xlim(0, df['importance'].max() * 1.2)
    
    plt.tight_layout()
    plt.savefig('images/insight_07_key_drivers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_07_key_drivers.png")


# =============================================================================
# OPERATIONAL VISUALIZATIONS
# =============================================================================

def viz_seasonality():
    """Visualize seasonality index."""
    if not os.path.exists('data/seasonality_index.csv'):
        print("  [SKIP] data/seasonality_index.csv not found")
        return
    
    df = pd.read_csv('data/seasonality_index.csv')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color by recommendation
    color_map = {'Reduce Staff': PALETTE['blue'], 'Normal': PALETTE['green'], 'Add Staff': PALETTE['red']}
    colors = [color_map.get(rec, PALETTE['blue']) for rec in df['recommendation']]
    
    bars = ax.bar(df['month_name'], df['seasonality_index'], color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Reference line at 100
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, label='Baseline (100)')
    
    # Add value labels
    for bar, val in zip(bars, df['seasonality_index']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Seasonality Index', fontsize=12)
    ax.set_title('INSIGHT: Monthly Seasonality Index for Resource Planning', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(df['seasonality_index']) * 1.15)
    
    # Legend
    reduce = mpatches.Patch(color=PALETTE['blue'], label='Reduce Staff (<90)')
    normal = mpatches.Patch(color=PALETTE['green'], label='Normal (90-110)')
    add = mpatches.Patch(color=PALETTE['red'], label='Add Staff (>110)')
    ax.legend(handles=[reduce, normal, add], loc='upper right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('images/insight_08_seasonality.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_08_seasonality.png")


def viz_capacity():
    """Visualize capacity utilization."""
    if not os.path.exists('data/capacity_utilization.csv'):
        print("  [SKIP] data/capacity_utilization.csv not found")
        return
    
    df = pd.read_csv('data/capacity_utilization.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Capacity status distribution
    ax1 = axes[0]
    status_counts = df['capacity_status'].value_counts()
    colors_map = {
        'Under-utilized': PALETTE['blue'],
        'Optimal': PALETTE['green'],
        'Near Capacity': PALETTE['yellow'],
        'Over Capacity': PALETTE['red']
    }
    colors = [colors_map.get(s, PALETTE['blue']) for s in status_counts.index]
    
    wedges, texts, autotexts = ax1.pie(status_counts.values, labels=status_counts.index,
                                        colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Capacity Utilization Distribution', fontsize=12, fontweight='bold')
    
    # Utilization histogram
    ax2 = axes[1]
    ax2.hist(df['avg_utilization'].dropna(), bins=20, color=PALETTE['teal'], alpha=0.7, edgecolor='white')
    ax2.axvline(x=80, color=PALETTE['red'], linestyle='--', label='Target (80%)')
    ax2.axvline(x=df['avg_utilization'].mean(), color=PALETTE['purple'], linestyle='--', 
               label=f'Avg ({df["avg_utilization"].mean():.0f}%)')
    ax2.set_xlabel('Utilization %', fontsize=11)
    ax2.set_ylabel('Number of Districts', fontsize=11)
    ax2.set_title('Utilization Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    
    fig.suptitle('INSIGHT: Enrollment Center Capacity Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('images/insight_09_capacity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_09_capacity.png")


def viz_recommendations():
    """Visualize recommendations summary."""
    if not os.path.exists('data/recommendations.csv'):
        print("  [SKIP] data/recommendations.csv not found")
        return
    
    df = pd.read_csv('data/recommendations.csv')
    
    fig, ax = plt.subplots(figsize=(14, len(df) * 1.2 + 2))
    
    priority_colors = {'High': PALETTE['red'], 'Medium': PALETTE['yellow'], 'Low': PALETTE['green']}
    
    for i, row in df.iterrows():
        y_pos = len(df) - i - 1
        
        # Priority badge
        color = priority_colors.get(row['priority'], PALETTE['blue'])
        rect = plt.Rectangle((0, y_pos - 0.4), 0.08, 0.8, color=color, alpha=0.8)
        ax.add_patch(rect)
        
        # Title
        ax.text(0.12, y_pos + 0.2, row['title'], fontsize=12, fontweight='bold', va='center')
        
        # Description
        ax.text(0.12, y_pos - 0.15, row['description'][:80] + '...', fontsize=9, va='center', color='gray')
        
        # Category badge
        ax.text(0.95, y_pos, f"[{row['category']}]", fontsize=9, va='center', ha='right',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(df))
    ax.axis('off')
    ax.set_title('INSIGHT: Strategic Recommendations', fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    high = mpatches.Patch(color=PALETTE['red'], label='High Priority')
    medium = mpatches.Patch(color=PALETTE['yellow'], label='Medium Priority')
    low = mpatches.Patch(color=PALETTE['green'], label='Low Priority')
    ax.legend(handles=[high, medium, low], loc='upper right', bbox_to_anchor=(1, 1.1))
    
    plt.tight_layout()
    plt.savefig('images/insight_10_recommendations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] images/insight_10_recommendations.png")


# =============================================================================
# MAIN
# =============================================================================

def generate_all_insight_visualizations():
    """Generate all insight visualizations from output CSVs."""
    print("\n" + "="*60)
    print("  GENERATING INSIGHT VISUALIZATIONS FROM OUTPUT DATA")
    print("="*60 + "\n")
    
    # Anomaly insights
    print("[1/10] Anomaly Insights")
    viz_zscore_anomalies()
    viz_geographic_anomalies()
    viz_holiday_impact()
    
    # Forecast insights
    print("\n[2/10] Forecast Insights")
    viz_forecast_30day()
    viz_milestone_projections()
    
    # Risk & ML insights
    print("\n[3/10] Risk & ML Insights")
    viz_district_risk()
    viz_feature_importance()
    
    # Operational insights
    print("\n[4/10] Operational Insights")
    viz_seasonality()
    viz_capacity()
    viz_recommendations()
    
    print("\n" + "="*60)
    print("  INSIGHT VISUALIZATIONS COMPLETE!")
    print("="*60)
    
    # Count generated files
    insight_files = [f for f in os.listdir('images') if f.startswith('insight_') and f.endswith('.png')]
    print(f"\n  Generated {len(insight_files)} insight visualizations (insight_01 to insight_10)")


def main():
    """Main function."""
    generate_all_insight_visualizations()


if __name__ == "__main__":
    main()
