"""
Advanced Visualizations Module for Aadhaar Enrollment Data
Implements new chart types: Sankey, Treemap, Radar, Pareto, Waterfall, etc.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Style settings
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#e0e0e0',
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.dpi': 150
})

COLORS = {
    'primary': '#1a73e8',
    'secondary': '#34a853',
    'accent': '#ea4335',
    'warning': '#fbbc04',
    'purple': '#9334e6',
    'age_0_5': '#FF6B6B',
    'age_5_17': '#4ECDC4',
    'age_18': '#45B7D1'
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
# NEW VISUALIZATION TYPES
# =============================================================================

def plot_treemap(df, n=15):
    """Create treemap showing state/district hierarchy."""
    print("Generating treemap visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    state_data = df.groupby('state')['total_enrollments'].sum().nlargest(n).sort_values(ascending=False)
    total = state_data.sum()
    
    # Calculate rectangles
    x, y, w, h = 0, 0, 100, 100
    rects = []
    remaining_area = w * h
    remaining_values = state_data.values.sum()
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(state_data)))
    
    for i, (state, value) in enumerate(state_data.items()):
        ratio = value / remaining_values
        area = remaining_area * ratio
        
        if i % 2 == 0:  # Horizontal split
            rect_w = area / h if h > 0 else 0
            rect_h = h
            rect = plt.Rectangle((x, y), rect_w, rect_h, facecolor=colors[i], 
                                   edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            # Label
            if rect_w > 5:
                pct = value / total * 100
                ax.text(x + rect_w/2, y + rect_h/2, f'{state}\n{format_number(value)}\n({pct:.1f}%)',
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            x += rect_w
            remaining_area -= area
        else:  # Vertical split
            rect_w = w - x
            rect_h = area / rect_w if rect_w > 0 else 0
            rect = plt.Rectangle((x, y), rect_w, rect_h, facecolor=colors[i],
                                   edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            if rect_h > 5:
                pct = value / total * 100
                ax.text(x + rect_w/2, y + rect_h/2, f'{state}\n{format_number(value)}\n({pct:.1f}%)',
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            y += rect_h
            h -= rect_h
            remaining_area -= area
        
        remaining_values -= value
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'State Enrollment Treemap (Top {n} States)', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('images/graph_26_treemap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: images/graph_26_treemap.png")


def plot_radar_chart(df, states=None, n=5):
    """Create radar chart comparing states across dimensions."""
    print("Generating radar chart...")
    
    if states is None:
        states = df.groupby('state')['total_enrollments'].sum().nlargest(n).index.tolist()
    
    # Prepare metrics
    state_metrics = []
    for state in states:
        state_df = df[df['state'] == state]
        total = state_df['total_enrollments'].sum()
        metrics = {
            'state': state,
            'Volume': total,
            'Children %': state_df['age_0_5'].sum() / total * 100,
            'Youth %': state_df['age_5_17'].sum() / total * 100,
            'Adults %': state_df['age_18_greater'].sum() / total * 100,
            'Districts': state_df['district'].nunique(),
            'Pincodes': state_df['pincode'].nunique()
        }
        state_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(state_metrics)
    
    # Normalize metrics
    categories = ['Volume', 'Children %', 'Youth %', 'Adults %', 'Districts', 'Pincodes']
    normalized = metrics_df.copy()
    for cat in categories:
        max_val = metrics_df[cat].max()
        normalized[cat] = metrics_df[cat] / max_val * 100
    
    # Create radar
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['warning'], COLORS['purple']]
    
    for i, (_, row) in enumerate(normalized.iterrows()):
        values = [row[cat] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['state'], color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('State Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('images/graph_27_radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: images/graph_27_radar_chart.png")


def plot_pareto_chart(df, n=20):
    """Create Pareto chart for 80/20 analysis."""
    print("Generating Pareto chart...")
    
    state_data = df.groupby('state')['total_enrollments'].sum().sort_values(ascending=False)
    total = state_data.sum()
    
    cumulative_pct = state_data.cumsum() / total * 100
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Bar chart
    x = range(len(state_data))
    bars = ax1.bar(x, state_data.values, color=COLORS['primary'], alpha=0.7, label='Enrollments')
    ax1.set_xlabel('State', fontsize=12)
    ax1.set_ylabel('Total Enrollments', fontsize=12, color=COLORS['primary'])
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(state_data.index, rotation=45, ha='right', fontsize=8)
    
    # Cumulative line
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative_pct.values, 'o-', color=COLORS['accent'], linewidth=2, label='Cumulative %')
    ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='80% line')
    ax2.set_ylabel('Cumulative %', fontsize=12, color=COLORS['accent'])
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax2.set_ylim(0, 105)
    
    # Find 80% point
    idx_80 = np.argmax(cumulative_pct.values >= 80)
    ax1.axvline(x=idx_80, color='gray', linestyle='--', alpha=0.5)
    ax2.annotate(f'{idx_80 + 1} states = 80%', xy=(idx_80, 80), xytext=(idx_80 + 2, 85),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax1.set_title('Pareto Analysis: State Contribution to Total Enrollments', fontsize=14, fontweight='bold')
    fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
    
    plt.tight_layout()
    plt.savefig('images/graph_28_pareto_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: images/graph_28_pareto_chart.png")


def plot_waterfall_chart(df):
    """Create waterfall chart for month-over-month changes."""
    print("Generating waterfall chart...")
    
    df['date'] = pd.to_datetime(df['date'])
    monthly = df.groupby(df['date'].dt.to_period('M'))['total_enrollments'].sum()
    
    changes = monthly.diff().dropna()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = range(len(changes))
    colors = [COLORS['secondary'] if v >= 0 else COLORS['accent'] for v in changes.values]
    
    # Calculate running position
    cumulative = [monthly.iloc[0]]
    for change in changes.values:
        cumulative.append(cumulative[-1] + change)
    
    for i, (change, cum) in enumerate(zip(changes.values, cumulative[1:])):
        bottom = min(cumulative[i], cum)
        height = abs(change)
        ax.bar(i, height, bottom=bottom, color=colors[i], edgecolor='white', linewidth=1)
        
        # Label
        label_pos = cum if change >= 0 else cumulative[i]
        ax.text(i, label_pos + height * 0.1, f'{format_number(abs(change))}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in changes.index], rotation=45, ha='right')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Enrollments', fontsize=12)
    ax.set_title('Month-over-Month Enrollment Changes (Waterfall)', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))
    
    # Legend
    increase = mpatches.Patch(color=COLORS['secondary'], label='Increase')
    decrease = mpatches.Patch(color=COLORS['accent'], label='Decrease')
    ax.legend(handles=[increase, decrease], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('images/graph_29_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: images/graph_29_waterfall.png")


def plot_bubble_chart(df, n=30):
    """Create bubble chart: volume vs growth rate vs age mix."""
    print("Generating bubble chart...")
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate metrics per district
    district_data = df.groupby(['state', 'district']).agg({
        'total_enrollments': 'sum',
        'age_0_5': 'sum',
        'date': ['min', 'max', 'nunique']
    }).reset_index()
    
    district_data.columns = ['state', 'district', 'total', 'children', 'start', 'end', 'days']
    district_data = district_data[district_data['days'] > 7]  # Need history
    
    # Calculate growth rate
    district_data['daily_avg'] = district_data['total'] / district_data['days']
    district_data['children_pct'] = district_data['children'] / district_data['total'] * 100
    
    # Sample for visualization
    sample = district_data.nlargest(n, 'total')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Bubble sizes based on total enrollments
    sizes = (sample['total'] / sample['total'].max() * 1000) + 50
    
    # Colors based on children percentage
    colors = sample['children_pct']
    
    scatter = ax.scatter(sample['daily_avg'], sample['children_pct'],
                        s=sizes, c=colors, cmap='RdYlGn', alpha=0.6, edgecolors='white', linewidth=1)
    
    plt.colorbar(scatter, label='Children % (Color)', ax=ax)
    
    # Label top points
    for _, row in sample.nlargest(5, 'total').iterrows():
        ax.annotate(row['district'], (row['daily_avg'], row['children_pct']),
                   fontsize=8, ha='center', va='bottom')
    
    ax.set_xlabel('Daily Average Enrollments', fontsize=12)
    ax.set_ylabel('Children (0-5) Percentage', fontsize=12)
    ax.set_title('District Analysis: Volume (size) vs Age Mix', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))
    
    plt.tight_layout()
    plt.savefig('images/graph_30_bubble_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: images/graph_30_bubble_chart.png")


def plot_control_chart(df):
    """Create statistical control chart for monitoring."""
    print("Generating control chart...")
    
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby('date')['total_enrollments'].sum().reset_index()
    daily.columns = ['date', 'value']
    daily = daily.sort_values('date')
    
    mean = daily['value'].mean()
    std = daily['value'].std()
    ucl = mean + 3 * std  # Upper Control Limit
    lcl = max(0, mean - 3 * std)  # Lower Control Limit
    uwl = mean + 2 * std  # Upper Warning Limit
    lwl = max(0, mean - 2 * std)  # Lower Warning Limit
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(daily['date'], daily['value'], '-o', markersize=3, color=COLORS['primary'], label='Daily Enrollments')
    
    # Control limits
    ax.axhline(y=mean, color='green', linestyle='-', linewidth=2, label=f'Mean ({format_number(mean)})')
    ax.axhline(y=ucl, color='red', linestyle='--', linewidth=1.5, label=f'UCL ({format_number(ucl)})')
    ax.axhline(y=lcl, color='red', linestyle='--', linewidth=1.5, label=f'LCL ({format_number(lcl)})')
    ax.axhline(y=uwl, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax.axhline(y=lwl, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    
    # Fill zones
    ax.fill_between(daily['date'], lcl, ucl, alpha=0.1, color='green')
    ax.fill_between(daily['date'], uwl, ucl, alpha=0.1, color='orange')
    ax.fill_between(daily['date'], lcl, lwl, alpha=0.1, color='orange')
    
    # Mark out-of-control points
    ooc = daily[(daily['value'] > ucl) | (daily['value'] < lcl)]
    ax.scatter(ooc['date'], ooc['value'], color='red', s=100, zorder=5, label=f'Out of Control ({len(ooc)})')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Daily Enrollments', fontsize=12)
    ax.set_title('Statistical Process Control Chart', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format_number(x)))
    
    plt.tight_layout()
    plt.savefig('images/graph_31_control_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: images/graph_31_control_chart.png")


def plot_sunburst_data(df):
    """Prepare sunburst-style nested bar chart."""
    print("Generating hierarchical breakdown chart...")
    
    # State -> Age Group breakdown
    state_data = df.groupby('state').agg({
        'total_enrollments': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).nlargest(8, 'total_enrollments')
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (state, row) in enumerate(state_data.iterrows()):
        ax = axes[i]
        values = [row['age_0_5'], row['age_5_17'], row['age_18_greater']]
        labels = ['0-5', '5-17', '18+']
        colors = [COLORS['age_0_5'], COLORS['age_5_17'], COLORS['age_18']]
        
        wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                           pctdistance=0.75, startangle=90)
        
        # Add center circle for donut
        centre_circle = plt.Circle((0, 0), 0.5, fc='white')
        ax.add_patch(centre_circle)
        ax.text(0, 0, format_number(row['total_enrollments']), ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        ax.set_title(state[:15], fontsize=10, fontweight='bold')
    
    fig.suptitle('Age Group Composition by Top 8 States', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('images/graph_32_sunburst_style.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: images/graph_32_sunburst_style.png")


def generate_advanced_visualizations(df):
    """Generate all advanced visualizations."""
    print("\n" + "="*50 + "\nGENERATING ADVANCED VISUALIZATIONS\n" + "="*50)
    
    df['date'] = pd.to_datetime(df['date'])
    
    plot_treemap(df)
    plot_radar_chart(df)
    plot_pareto_chart(df)
    plot_waterfall_chart(df)
    plot_bubble_chart(df)
    plot_control_chart(df)
    plot_sunburst_data(df)
    
    print("\n" + "="*50 + "\nALL ADVANCED VISUALIZATIONS COMPLETE\n" + "="*50)


def main():
    """Main function."""
    print("Loading data...")
    try:
        df = pd.read_csv('data/aadhar_enrollment_cleaned.csv')
        df['date'] = pd.to_datetime(df['date'])
    except FileNotFoundError:
        print("Error: Run src/data_processing.py first!")
        return
    
    generate_advanced_visualizations(df)
    print("\n7 new visualizations generated (graphs 26-32)!")


if __name__ == "__main__":
    main()
