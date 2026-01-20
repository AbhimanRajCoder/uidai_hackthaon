"""
Aadhar Enrollment Data Visualization - Enhanced Version
Premium styling with clear captions and accurate data representation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# PREMIUM STYLE CONFIGURATION
# =============================================================================

# Color Palettes
COLORS = {
    'primary': '#1a73e8',
    'secondary': '#34a853',
    'accent': '#ea4335',
    'warning': '#fbbc04',
    'dark': '#202124',
    'light': '#f8f9fa',
    'gradient': ['#667eea', '#764ba2'],
    'age_0_5': '#FF6B6B',
    'age_5_17': '#4ECDC4',
    'age_18': '#45B7D1',
    'states': plt.cm.viridis(np.linspace(0.15, 0.85, 15)),
}

# Premium style settings
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#333333',
    'axes.titlecolor': '#1a1a1a',
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'font.family': 'sans-serif',
    'font.size': 10,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#e0e0e0',
    'figure.dpi': 150,
})


def add_caption(fig, caption_text, y_pos=-0.08):
    """Add a caption below the figure."""
    fig.text(0.5, y_pos, caption_text, ha='center', va='top', fontsize=9,
             style='italic', color='#555555', wrap=True,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor='#e0e0e0'))


def format_number(val):
    """Format large numbers with K/M/L suffix."""
    if val >= 1e7:
        return f'{val/1e7:.1f}Cr'
    elif val >= 1e5:
        return f'{val/1e5:.1f}L'
    elif val >= 1e3:
        return f'{val/1e3:.1f}K'
    return f'{val:.0f}'


def load_data():
    """Load the cleaned data or merge raw files."""
    try:
        if not os.path.exists('data/aadhar_enrollment_cleaned.csv'):
            print("[!] Cleaned data not found. Running data cleaning first...")
            # Since we are inside src, we import relative or absolute
            import data_processing 
            data_processing.main()
        
        df = pd.read_csv('data/aadhar_enrollment_cleaned.csv')
        df['date'] = pd.to_datetime(df['date'])
        print("[OK] Loaded: data/aadhar_enrollment_cleaned.csv")
    except FileNotFoundError:
        print("[WARNING] Loading raw files...")
        files = [
            'api_data_aadhar_enrolment_1000000_1006029.csv'
        ]
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df['total_enrollments'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name()
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['quarter'] = df['date'].dt.quarter
    return df


# =============================================================================
# TIME SERIES GRAPHS
# =============================================================================

def plot_daily_enrollments(df):
    """Daily enrollment trends with 7-day moving average."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    daily = df.groupby('date')['total_enrollments'].sum().sort_index()
    ma7 = daily.rolling(window=7, min_periods=1).mean()
    
    ax.fill_between(daily.index, daily.values, alpha=0.3, color=COLORS['primary'], label='Daily')
    ax.plot(daily.index, daily.values, linewidth=0.8, color=COLORS['primary'], alpha=0.6)
    ax.plot(ma7.index, ma7.values, linewidth=2.5, color=COLORS['accent'], label='7-Day Moving Avg')
    
    ax.set_title('Daily Aadhar Enrollments Over Time', fontsize=16, pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    ax.legend(loc='upper right', frameon=True)
    
    total = daily.sum()
    avg_daily = daily.mean()
    add_caption(fig, f'Caption: Total enrollments: {format_number(total)} | Average daily: {format_number(avg_daily)} | Data from {daily.index.min().strftime("%d %b %Y")} to {daily.index.max().strftime("%d %b %Y")}')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_01_daily_enrollments.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_01_daily_enrollments.png")


def plot_monthly_enrollments(df):
    """Monthly enrollment bar chart with trend line."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    monthly = df.groupby(df['date'].dt.to_period('M'))['total_enrollments'].sum()
    months = [m.to_timestamp() for m in monthly.index]
    
    bars = ax.bar(months, monthly.values, width=20, color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.plot(months, monthly.values, color=COLORS['accent'], linewidth=2.5, marker='o', markersize=6, zorder=5)
    
    # Highlight max/min
    max_idx = monthly.idxmax()
    min_idx = monthly.idxmin()
    ax.bar(max_idx.to_timestamp(), monthly[max_idx], width=20, color=COLORS['secondary'], alpha=0.9, label=f'Peak: {max_idx.strftime("%b %Y")}')
    ax.bar(min_idx.to_timestamp(), monthly[min_idx], width=20, color=COLORS['accent'], alpha=0.9, label=f'Low: {min_idx.strftime("%b %Y")}')
    
    ax.set_title('Monthly Aadhar Enrollments', fontsize=16, pad=15)
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    ax.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    
    add_caption(fig, f'Caption: Highest enrollment in {max_idx.strftime("%B %Y")} ({format_number(monthly[max_idx])}) | Lowest in {min_idx.strftime("%B %Y")} ({format_number(monthly[min_idx])})')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_02_monthly_enrollments.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_02_monthly_enrollments.png")


def plot_weekly_trend(df):
    """Weekly trend with 4-week moving average."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    weekly = df.groupby(df['date'].dt.to_period('W'))['total_enrollments'].sum()
    weeks = [w.to_timestamp() for w in weekly.index]
    ma4 = weekly.rolling(window=4, min_periods=1).mean()
    
    ax.plot(weeks, weekly.values, linewidth=1.5, color=COLORS['primary'], alpha=0.7, marker='o', markersize=3, label='Weekly')
    ax.plot(weeks, ma4.values, linewidth=2.5, color=COLORS['accent'], linestyle='-', label='4-Week Moving Avg')
    ax.fill_between(weeks, weekly.values, alpha=0.2, color=COLORS['primary'])
    
    ax.set_title('Weekly Enrollment Trend with Moving Average', fontsize=16, pad=15)
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    ax.legend(loc='upper right')
    
    add_caption(fig, f'Caption: Total weeks analyzed: {len(weekly)} | Weekly average: {format_number(weekly.mean())} enrollments')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_03_weekly_trend.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_03_weekly_trend.png")


# =============================================================================
# GEOGRAPHIC DISTRIBUTION
# =============================================================================

def plot_top_states_bar(df, n=15):
    """Top states by enrollment with percentage labels."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    state_data = df.groupby('state')['total_enrollments'].sum().nlargest(n).sort_values()
    total = df['total_enrollments'].sum()
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(state_data)))
    bars = ax.barh(state_data.index, state_data.values, color=colors, edgecolor='white', height=0.7)
    
    for bar, val in zip(bars, state_data.values):
        pct = (val / total) * 100
        ax.text(val + total*0.005, bar.get_y() + bar.get_height()/2, f'{format_number(val)} ({pct:.1f}%)', 
                va='center', fontsize=9, fontweight='bold')
    
    ax.set_title(f'Top {n} States by Aadhar Enrollments', fontsize=16, pad=15)
    ax.set_xlabel('Total Enrollments')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    
    top3_pct = (state_data.nlargest(3).sum() / total) * 100
    add_caption(fig, f'Caption: Top 3 states contribute {top3_pct:.1f}% of total enrollments | Data covers {df["state"].nunique()} states/UTs')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_04_top_states.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_04_top_states.png")


def plot_top_districts_bar(df, n=20):
    """Top districts with state context."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    district_data = df.groupby(['state', 'district'])['total_enrollments'].sum().nlargest(n).sort_values()
    labels = [f"{d}\n({s})" for s, d in district_data.index]
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(district_data)))
    bars = ax.barh(range(len(district_data)), district_data.values, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(district_data)))
    ax.set_yticklabels(labels, fontsize=8)
    
    ax.set_title(f'Top {n} Districts by Aadhar Enrollments', fontsize=16, pad=15)
    ax.set_xlabel('Total Enrollments')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    
    top_district = district_data.idxmax()
    add_caption(fig, f'Caption: Highest enrollment district: {top_district[1]} ({top_district[0]}) with {format_number(district_data.max())} enrollments')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('images/graph_05_top_districts.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_05_top_districts.png")


def plot_state_enrollment_pie(df, n=10):
    """State enrollment donut chart."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    state_data = df.groupby('state')['total_enrollments'].sum().nlargest(n)
    others = df['total_enrollments'].sum() - state_data.sum()
    if others > 0:
        state_data['Others'] = others
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(state_data)))
    wedges, texts, autotexts = ax.pie(state_data.values, labels=state_data.index, autopct='%1.1f%%',
                                       colors=colors, startangle=90, pctdistance=0.75,
                                       wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
    
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    centre_circle = plt.Circle((0, 0), 0.35, fc='white')
    ax.add_patch(centre_circle)
    ax.text(0, 0, f'{format_number(df["total_enrollments"].sum())}\nTotal', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.set_title(f'Enrollment Distribution by State', fontsize=16, pad=15)
    
    add_caption(fig, f'Caption: Top {n} states shown individually, remaining {df["state"].nunique() - n} states grouped as "Others"')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('images/graph_06_state_pie_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_06_state_pie_chart.png")


# =============================================================================
# AGE GROUP ANALYSIS
# =============================================================================

def plot_age_group_distribution(df):
    """Age group comparison bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    age_0_5 = df['age_0_5'].sum()
    age_5_17 = df['age_5_17'].sum()
    age_18_plus = df['age_18_greater'].sum()
    total = age_0_5 + age_5_17 + age_18_plus
    
    age_data = {'Age 0-5\n(Children)': age_0_5, 
                'Age 5-17\n(Youth)': age_5_17, 
                'Age 18+\n(Adults)': age_18_plus}
    
    colors = [COLORS['age_0_5'], COLORS['age_5_17'], COLORS['age_18']]
    bars = ax.bar(age_data.keys(), age_data.values(), color=colors, edgecolor='white', linewidth=2, width=0.6)
    
    for bar, val in zip(bars, age_data.values()):
        pct = (val / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, 
                f'{format_number(val)}\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_title('Total Enrollments by Age Group', fontsize=16, pad=15)
    ax.set_ylabel('Total Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    ax.set_ylim(0, max(age_data.values()) * 1.15)
    
    add_caption(fig, f'Caption: Adults (18+) form {(age_18_plus / total * 100):.1f}% of all enrollments | Youth (5-17): {(age_5_17 / total * 100):.1f}% | Children (0-5): {(age_0_5 / total * 100):.1f}%')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_07_age_group_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_07_age_group_distribution.png")


def plot_age_group_pie(df):
    """Age group donut chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = ['Age 0-5 (Children)', 'Age 5-17 (Youth)', 'Age 18+ (Adults)']
    values = [df['age_0_5'].sum(), df['age_5_17'].sum(), df['age_18_greater'].sum()]
    colors = [COLORS['age_0_5'], COLORS['age_5_17'], COLORS['age_18']]
    
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors,
                                       startangle=90, pctdistance=0.75,
                                       wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
    
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    centre_circle = plt.Circle((0, 0), 0.35, fc='white')
    ax.add_patch(centre_circle)
    ax.text(0, 0, f'{format_number(sum(values))}\nTotal', ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.set_title('Age Group Distribution', fontsize=16, pad=15)
    
    add_caption(fig, f'Caption: Age distribution of Aadhar enrollments across {len(df):,} records')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('images/graph_08_age_group_pie.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_08_age_group_pie.png")


def plot_age_group_trend(df):
    """Age group trends over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    monthly = df.groupby(df['date'].dt.to_period('M')).agg({
        'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'
    })
    months = [m.to_timestamp() for m in monthly.index]
    
    ax.plot(months, monthly['age_0_5'], label='Age 0-5', linewidth=2.5, marker='o', markersize=5, color=COLORS['age_0_5'])
    ax.plot(months, monthly['age_5_17'], label='Age 5-17', linewidth=2.5, marker='s', markersize=5, color=COLORS['age_5_17'])
    ax.plot(months, monthly['age_18_greater'], label='Age 18+', linewidth=2.5, marker='^', markersize=5, color=COLORS['age_18'])
    
    ax.set_title('Monthly Enrollment Trends by Age Group', fontsize=16, pad=15)
    ax.set_xlabel('Month')
    ax.set_ylabel('Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    ax.legend(loc='upper left', frameon=True)
    plt.xticks(rotation=45, ha='right')
    
    add_caption(fig, f'Caption: Monthly breakdown of age-wise enrollment patterns from {months[0].strftime("%b %Y")} to {months[-1].strftime("%b %Y")}')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_09_age_group_trend.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_09_age_group_trend.png")


def plot_age_group_stacked(df):
    """Stacked area chart of age composition."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    monthly = df.groupby(df['date'].dt.to_period('M')).agg({
        'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'
    })
    months = [m.to_timestamp() for m in monthly.index]
    
    ax.stackplot(months, monthly['age_0_5'], monthly['age_5_17'], monthly['age_18_greater'],
                 labels=['Age 0-5', 'Age 5-17', 'Age 18+'],
                 colors=[COLORS['age_0_5'], COLORS['age_5_17'], COLORS['age_18']], alpha=0.85)
    
    ax.set_title('Age Group Composition Over Time (Stacked Area)', fontsize=16, pad=15)
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    ax.legend(loc='upper left')
    plt.xticks(rotation=45, ha='right')
    
    add_caption(fig, f'Caption: Stacked view showing cumulative contribution of each age group to total monthly enrollments')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_10_age_group_stacked.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_10_age_group_stacked.png")


# =============================================================================
# TEMPORAL PATTERNS
# =============================================================================

def plot_day_of_week(df):
    """Enrollments by day of week."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_data = df.groupby('day_name')['total_enrollments'].sum().reindex(day_order)
    avg = day_data.mean()
    
    colors = ['#2ecc71' if v >= avg else '#e74c3c' for v in day_data.values]
    bars = ax.bar(day_data.index, day_data.values, color=colors, edgecolor='white', linewidth=2)
    ax.axhline(y=avg, color='#333', linestyle='--', linewidth=2, label=f'Average: {format_number(avg)}')
    
    ax.set_title('Enrollments by Day of Week', fontsize=16, pad=15)
    ax.set_ylabel('Total Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    ax.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    
    max_day = day_data.idxmax()
    min_day = day_data.idxmin()
    add_caption(fig, f'Caption: Highest on {max_day} ({format_number(day_data[max_day])}) | Lowest on {min_day} ({format_number(day_data[min_day])}) | Green = Above Average')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_11_day_of_week.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_11_day_of_week.png")


def plot_month_distribution(df):
    """Enrollments by month."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    month_data = df.groupby('month_name')['total_enrollments'].sum().reindex(month_order).dropna()
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(month_data)))
    bars = ax.bar(month_data.index, month_data.values, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_title('Enrollments by Month', fontsize=16, pad=15)
    ax.set_ylabel('Total Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    plt.xticks(rotation=45, ha='right')
    
    max_month = month_data.idxmax()
    add_caption(fig, f'Caption: Peak enrollment month: {max_month} ({format_number(month_data[max_month])}) | Total months with data: {len(month_data)}')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_12_month_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_12_month_distribution.png")


def plot_quarterly_trend(df):
    """Quarterly enrollment analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    quarterly = df.groupby('quarter')['total_enrollments'].sum()
    labels = ['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax.bar([labels[i-1] for i in quarterly.index], quarterly.values, 
                  color=[colors[i-1] for i in quarterly.index], edgecolor='white', linewidth=2, width=0.6)
    
    for bar, val in zip(bars, quarterly.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + quarterly.max()*0.02,
                format_number(val), ha='center', fontsize=12, fontweight='bold')
    
    ax.set_title('Enrollments by Quarter', fontsize=16, pad=15)
    ax.set_ylabel('Total Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    ax.set_ylim(0, quarterly.max() * 1.15)
    
    best_q = quarterly.idxmax()
    add_caption(fig, f'Caption: Q{best_q} has highest enrollments ({format_number(quarterly[best_q])}) | Q{quarterly.idxmin()} has lowest ({format_number(quarterly.min())})')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_13_quarterly.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_13_quarterly.png")


# =============================================================================
# HEATMAPS
# =============================================================================

def plot_heatmap_day_month(df):
    """Day vs Month heatmap."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    pivot = df.pivot_table(values='total_enrollments', index='day_name', columns='month_name', aggfunc='sum')
    pivot = pivot.reindex(index=[d for d in day_order if d in pivot.index])
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
    
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Enrollments', 'format': mticker.FuncFormatter(lambda x, _: format_number(x))})
    
    ax.set_title('Enrollment Heatmap: Day of Week vs Month', fontsize=16, pad=15)
    
    max_val = pivot.max().max()
    max_pos = pivot.stack().idxmax()
    add_caption(fig, f'Caption: Highest enrollments on {max_pos[0]}, {max_pos[1]} ({format_number(max_val)}) | Darker colors = Higher enrollments')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('images/graph_14_heatmap_day_month.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_14_heatmap_day_month.png")


def plot_heatmap_state_month(df, n=15):
    """State vs Month heatmap."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    top_states = df.groupby('state')['total_enrollments'].sum().nlargest(n).index
    df_top = df[df['state'].isin(top_states)]
    
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    pivot = df_top.pivot_table(values='total_enrollments', index='state', columns='month_name', aggfunc='sum')
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Blues', ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Enrollments'})
    
    ax.set_title(f'Top {n} States: Monthly Enrollment Heatmap', fontsize=16, pad=15)
    
    add_caption(fig, f'Caption: Heatmap showing monthly enrollment patterns for top {n} states by total enrollments')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('images/graph_15_heatmap_state_month.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_15_heatmap_state_month.png")


# =============================================================================
# DISTRIBUTIONS
# =============================================================================

def plot_enrollment_histogram(df):
    """Distribution histograms for enrollment counts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    data_info = [
        ('total_enrollments', 'Total Enrollments', COLORS['primary']),
        ('age_0_5', 'Age 0-5', COLORS['age_0_5']),
        ('age_5_17', 'Age 5-17', COLORS['age_5_17']),
        ('age_18_greater', 'Age 18+', COLORS['age_18'])
    ]
    
    for ax, (col, title, color) in zip(axes.flat, data_info):
        ax.hist(df[col], bins=50, color=color, edgecolor='white', alpha=0.8)
        ax.set_title(f'Distribution: {title}', fontweight='bold')
        ax.set_xlabel('Enrollments per Record')
        ax.set_ylabel('Frequency')
        ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col].mean():.1f}')
        ax.axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[col].median():.1f}')
        ax.legend(fontsize=8)
    
    plt.suptitle('Enrollment Distribution Histograms', fontsize=16, fontweight='bold', y=1.02)
    add_caption(fig, f'Caption: Distribution of enrollment counts per record | Red line = Mean, Green line = Median')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig('images/graph_16_enrollment_histogram.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_16_enrollment_histogram.png")


def plot_boxplot_age_groups(df):
    """Box plot for age group distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [df['age_0_5'], df['age_5_17'], df['age_18_greater']]
    labels = ['Age 0-5', 'Age 5-17', 'Age 18+']
    colors = [COLORS['age_0_5'], COLORS['age_5_17'], COLORS['age_18']]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Enrollment Distribution by Age Group (Box Plot)', fontsize=16, pad=15)
    ax.set_ylabel('Enrollments per Record')
    
    add_caption(fig, f'Caption: Box plots showing median, quartiles, and outliers for each age group | Notch indicates 95% CI for median')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_17_boxplot_age_groups.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_17_boxplot_age_groups.png")


def plot_boxplot_top_states(df, n=10):
    """Box plot by top states."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top_states = df.groupby('state')['total_enrollments'].sum().nlargest(n).index
    df_top = df[df['state'].isin(top_states)]
    order = df_top.groupby('state')['total_enrollments'].median().sort_values(ascending=False).index
    
    sns.boxplot(data=df_top, x='state', y='total_enrollments', order=order, palette='viridis', ax=ax)
    
    ax.set_title(f'Enrollment Distribution: Top {n} States', fontsize=16, pad=15)
    ax.set_xlabel('State')
    ax.set_ylabel('Enrollments per Record')
    plt.xticks(rotation=45, ha='right')
    
    add_caption(fig, f'Caption: Box plots showing enrollment distribution per record for top {n} states | Ordered by median enrollment')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('images/graph_18_boxplot_states.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_18_boxplot_states.png")


# =============================================================================
# CORRELATIONS
# =============================================================================

def plot_correlation_matrix(df):
    """Correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cols = ['age_0_5', 'age_5_17', 'age_18_greater', 'total_enrollments']
    corr = df[cols].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='coolwarm', center=0, square=True,
                linewidths=2, linecolor='white', ax=ax, cbar_kws={'label': 'Correlation'})
    
    ax.set_title('Correlation Matrix: Age Groups', fontsize=16, pad=15)
    
    add_caption(fig, f'Caption: Pearson correlation coefficients between age group enrollments | Values range from -1 (negative) to +1 (positive)')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('images/graph_19_correlation_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_19_correlation_matrix.png")


def plot_scatter_age_groups(df):
    """Scatter plots between age groups."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sample = df.sample(min(5000, len(df)), random_state=42)
    
    pairs = [('age_0_5', 'age_5_17', '#3498db'), ('age_0_5', 'age_18_greater', '#e74c3c'), ('age_5_17', 'age_18_greater', '#2ecc71')]
    labels = ['Age 0-5', 'Age 5-17', 'Age 18+']
    
    for ax, (x, y, c) in zip(axes, pairs):
        ax.scatter(sample[x], sample[y], alpha=0.4, c=c, s=15, edgecolors='white', linewidth=0.3)
        ax.set_xlabel(x.replace('age_', 'Age ').replace('_greater', '+').replace('_', '-'))
        ax.set_ylabel(y.replace('age_', 'Age ').replace('_greater', '+').replace('_', '-'))
        corr = df[[x, y]].corr().iloc[0, 1]
        ax.set_title(f'r = {corr:.3f}', fontweight='bold')
    
    plt.suptitle('Age Group Correlations (Sample: 5,000 records)', fontsize=16, fontweight='bold', y=1.02)
    add_caption(fig, f'Caption: Scatter plots showing relationships between age groups | r = Pearson correlation coefficient')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig('images/graph_20_scatter_age_groups.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_20_scatter_age_groups.png")


# =============================================================================
# TOP N ANALYSIS
# =============================================================================

def plot_top_pincodes(df, n=20):
    """Top pincodes bar chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    pincode_data = df.groupby('pincode')['total_enrollments'].sum().nlargest(n).sort_values()
    
    colors = plt.cm.magma(np.linspace(0.3, 0.8, len(pincode_data)))
    bars = ax.barh(pincode_data.index.astype(str), pincode_data.values, color=colors, edgecolor='white')
    
    ax.set_title(f'Top {n} Pincodes by Enrollments', fontsize=16, pad=15)
    ax.set_xlabel('Total Enrollments')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    
    top_pin = pincode_data.idxmax()
    add_caption(fig, f'Caption: Pincode {top_pin} has highest enrollments ({format_number(pincode_data.max())}) | Total unique pincodes: {df["pincode"].nunique():,}')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('images/graph_21_top_pincodes.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_21_top_pincodes.png")


def plot_state_age_composition(df, n=10):
    """Stacked bar for state age composition."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    top_states = df.groupby('state')['total_enrollments'].sum().nlargest(n).index
    state_age = df[df['state'].isin(top_states)].groupby('state').agg({'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'})
    state_age = state_age.loc[state_age.sum(axis=1).sort_values(ascending=False).index]
    
    x = range(len(state_age))
    ax.bar(x, state_age['age_0_5'], label='Age 0-5', color=COLORS['age_0_5'], edgecolor='white')
    ax.bar(x, state_age['age_5_17'], bottom=state_age['age_0_5'], label='Age 5-17', color=COLORS['age_5_17'], edgecolor='white')
    ax.bar(x, state_age['age_18_greater'], bottom=state_age['age_0_5'] + state_age['age_5_17'], label='Age 18+', color=COLORS['age_18'], edgecolor='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels(state_age.index, rotation=45, ha='right')
    ax.set_title(f'Age Group Composition: Top {n} States', fontsize=16, pad=15)
    ax.set_ylabel('Total Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    ax.legend(loc='upper right')
    
    add_caption(fig, f'Caption: Stacked bars showing age group breakdown for each of the top {n} states by enrollment')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('images/graph_22_state_age_composition.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_22_state_age_composition.png")


# =============================================================================
# CUMULATIVE & GROWTH
# =============================================================================

def plot_cumulative_enrollments(df):
    """Cumulative enrollment line chart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    daily = df.groupby('date')['total_enrollments'].sum().sort_index().cumsum()
    
    ax.fill_between(daily.index, daily.values, alpha=0.3, color=COLORS['primary'])
    ax.plot(daily.index, daily.values, linewidth=2.5, color=COLORS['primary'])
    
    # Milestones
    milestones = [1e6, 5e6, 10e6, 20e6]
    for m in milestones:
        if daily.max() >= m:
            idx = daily[daily >= m].index[0]
            ax.axhline(y=m, color='gray', linestyle=':', alpha=0.5)
            ax.scatter([idx], [m], color=COLORS['accent'], s=100, zorder=5)
            ax.annotate(f'{format_number(m)}', (idx, m), textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax.set_title('Cumulative Aadhar Enrollments Over Time', fontsize=16, pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Enrollments')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    
    add_caption(fig, f'Caption: Running total of enrollments | Final cumulative total: {format_number(daily.iloc[-1])} enrollments')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_23_cumulative_enrollments.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_23_cumulative_enrollments.png")


def plot_growth_rate(df):
    """Month-over-month growth rate."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    monthly = df.groupby(df['date'].dt.to_period('M'))['total_enrollments'].sum()
    months = [m.to_timestamp() for m in monthly.index]
    growth = monthly.pct_change() * 100
    
    colors = ['#2ecc71' if g >= 0 else '#e74c3c' for g in growth.fillna(0)]
    bars = ax.bar(months, growth.fillna(0).values, width=20, color=colors, edgecolor='white')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_title('Month-over-Month Growth Rate (%)', fontsize=16, pad=15)
    ax.set_xlabel('Month')
    ax.set_ylabel('Growth Rate (%)')
    plt.xticks(rotation=45, ha='right')
    
    avg_growth = growth.dropna().mean()
    max_growth = growth.max()
    min_growth = growth.min()
    add_caption(fig, f'Caption: Green = Positive growth, Red = Negative growth | Avg: {avg_growth:.1f}% | Peak: +{max_growth:.1f}% | Lowest: {min_growth:.1f}%')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('images/graph_24_growth_rate.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_24_growth_rate.png")


# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================

def plot_summary_dashboard(df):
    """Comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('white')
    
    # Title
    fig.suptitle('AADHAR ENROLLMENT DATA - EXECUTIVE DASHBOARD', fontsize=22, fontweight='bold', y=0.98, color='#1a1a1a')
    
    # 1. Age Pie
    ax1 = fig.add_subplot(3, 3, 1)
    values = [df['age_0_5'].sum(), df['age_5_17'].sum(), df['age_18_greater'].sum()]
    colors = [COLORS['age_0_5'], COLORS['age_5_17'], COLORS['age_18']]
    ax1.pie(values, labels=['0-5', '5-17', '18+'], autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Age Distribution', fontweight='bold')
    
    # 2. Top States
    ax2 = fig.add_subplot(3, 3, 2)
    top = df.groupby('state')['total_enrollments'].sum().nlargest(10).sort_values()
    ax2.barh(top.index, top.values, color=plt.cm.Blues(np.linspace(0.4, 0.9, 10)))
    ax2.set_title('Top 10 States', fontweight='bold')
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: format_number(x)))
    
    # 3. Daily Trend
    ax3 = fig.add_subplot(3, 3, 3)
    daily = df.groupby('date')['total_enrollments'].sum()
    ax3.plot(daily.index, daily.values, color=COLORS['primary'], linewidth=0.8)
    ax3.fill_between(daily.index, daily.values, alpha=0.3, color=COLORS['primary'])
    ax3.set_title('Daily Enrollment Trend', fontweight='bold')
    
    # 4. Day of Week
    ax4 = fig.add_subplot(3, 3, 4)
    dow = df.groupby('day_of_week')['total_enrollments'].sum()
    ax4.bar(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], [dow.get(i, 0) for i in range(7)], color='#9b59b6')
    ax4.set_title('By Day of Week', fontweight='bold')
    
    # 5. Monthly
    ax5 = fig.add_subplot(3, 3, 5)
    monthly = df.groupby(df['date'].dt.to_period('M'))['total_enrollments'].sum()
    ax5.bar(range(len(monthly)), monthly.values, color='#e74c3c')
    ax5.set_title('Monthly Trend', fontweight='bold')
    ax5.set_xticks([])
    
    # 6. Quarterly
    ax6 = fig.add_subplot(3, 3, 6)
    q = df.groupby('quarter')['total_enrollments'].sum()
    ax6.bar([f'Q{i}' for i in q.index], q.values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'][:len(q)])
    ax6.set_title('Quarterly', fontweight='bold')
    
    # 7. Age Trends
    ax7 = fig.add_subplot(3, 3, 7)
    ma = df.groupby(df['date'].dt.to_period('M')).agg({'age_0_5': 'sum', 'age_5_17': 'sum', 'age_18_greater': 'sum'})
    ax7.plot(range(len(ma)), ma['age_0_5'], label='0-5', color=COLORS['age_0_5'])
    ax7.plot(range(len(ma)), ma['age_5_17'], label='5-17', color=COLORS['age_5_17'])
    ax7.plot(range(len(ma)), ma['age_18_greater'], label='18+', color=COLORS['age_18'])
    ax7.legend(fontsize=8)
    ax7.set_title('Age Group Trends', fontweight='bold')
    ax7.set_xticks([])
    
    # 8. Cumulative
    ax8 = fig.add_subplot(3, 3, 8)
    cum = daily.sort_index().cumsum()
    ax8.fill_between(cum.index, cum.values, color=COLORS['primary'], alpha=0.5)
    ax8.plot(cum.index, cum.values, color=COLORS['primary'])
    ax8.set_title('Cumulative Growth', fontweight='bold')
    
    # 9. Key Metrics
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    metrics = f"""
╔══════════════════════════════════════╗
║         KEY METRICS SUMMARY          ║
╠══════════════════════════════════════╣
║  Total Records:     {len(df):>15,}  ║
║  Total Enrollments: {df['total_enrollments'].sum():>15,}  ║
║                                      ║
║  States/UTs:        {df['state'].nunique():>15}  ║
║  Districts:         {df['district'].nunique():>15,}  ║
║  Pincodes:          {df['pincode'].nunique():>15,}  ║
║                                      ║
║  Date Range:                         ║
║  {df['date'].min().strftime('%d %b %Y')} - {df['date'].max().strftime('%d %b %Y')}      ║
║                                      ║
║  Age 0-5:           {df['age_0_5'].sum():>15,}  ║
║  Age 5-17:          {df['age_5_17'].sum():>15,}  ║
║  Age 18+:           {df['age_18_greater'].sum():>15,}  ║
╚══════════════════════════════════════╝
    """
    ax9.text(0.5, 0.5, metrics, transform=ax9.transAxes, fontsize=10, verticalalignment='center',
             horizontalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#e0e0e0'))
    
    add_caption(fig, f'Caption: Executive summary dashboard showing key enrollment metrics, trends, and distributions | Generated: {pd.Timestamp.now().strftime("%d %b %Y %H:%M")}', y_pos=0.01)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('images/graph_25_summary_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[OK] images/graph_25_summary_dashboard.png")


# =============================================================================
# MAIN
# =============================================================================

def generate_all_graphs():
    """Generate all enhanced graphs."""
    print("\n" + "=" * 60)
    print("  [GRAPH] GENERATING ENHANCED GRAPHS")
    print("=" * 60 + "\n")
    
    df = load_data()
    print(f"[DATA] {len(df):,} rows × {len(df.columns)} columns\n")
    
    # Generate all graphs
    plot_daily_enrollments(df)
    plot_monthly_enrollments(df)
    plot_weekly_trend(df)
    plot_top_states_bar(df)
    plot_top_districts_bar(df)
    plot_state_enrollment_pie(df)
    plot_age_group_distribution(df)
    plot_age_group_pie(df)
    plot_age_group_trend(df)
    plot_age_group_stacked(df)
    plot_day_of_week(df)
    plot_month_distribution(df)
    plot_quarterly_trend(df)
    plot_heatmap_day_month(df)
    plot_heatmap_state_month(df)
    plot_enrollment_histogram(df)
    plot_boxplot_age_groups(df)
    plot_boxplot_top_states(df)
    plot_correlation_matrix(df)
    plot_scatter_age_groups(df)
    plot_top_pincodes(df)
    plot_state_age_composition(df)
    plot_cumulative_enrollments(df)
    plot_growth_rate(df)
    plot_summary_dashboard(df)
    
    print("\n" + "=" * 60)
    print("[OK] ALL 25 ENHANCED GRAPHS GENERATED!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    generate_all_graphs()
