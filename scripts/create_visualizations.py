#!/usr/bin/env python3
"""
Create visualizations for ML analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_feature_importance_plot(jp_importance_df, nonjp_importance_df, output_dir):
    """Create feature importance comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Japan
    top_jp = jp_importance_df.head(15)
    axes[0].barh(range(len(top_jp)), top_jp['importance'].values)
    axes[0].set_yticks(range(len(top_jp)))
    axes[0].set_yticklabels(top_jp['feature'].values, fontsize=9)
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_title('Japan - Top 15 Features', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    
    # Non-Japan
    top_nonjp = nonjp_importance_df.head(15)
    axes[1].barh(range(len(top_nonjp)), top_nonjp['importance'].values)
    axes[1].set_yticks(range(len(top_nonjp)))
    axes[1].set_yticklabels(top_nonjp['feature'].values, fontsize=9)
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('Non-Japan - Top 15 Features', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'feature_importance_comparison.png'}")


def create_cvss_analysis_plot(jp_data, nonjp_data, output_dir):
    """Create CVSS score analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # CVSS Score Distribution
    axes[0, 0].hist([jp_data[jp_data['exploited']==1]['cvss_base_score'].fillna(0),
                     jp_data[jp_data['exploited']==0]['cvss_base_score'].fillna(0)],
                    bins=20, alpha=0.7, label=['Exploited', 'Not Exploited'])
    axes[0, 0].set_xlabel('CVSS Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Japan - CVSS Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    
    axes[0, 1].hist([nonjp_data[nonjp_data['exploited']==1]['cvss_base_score'].fillna(0),
                      nonjp_data[nonjp_data['exploited']==0]['cvss_base_score'].fillna(0)],
                     bins=20, alpha=0.7, label=['Exploited', 'Not Exploited'])
    axes[0, 1].set_xlabel('CVSS Score', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Non-Japan - CVSS Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    
    # Severity Distribution
    if 'cvss_base_severity' in jp_data.columns:
        jp_severity = jp_data.groupby(['cvss_base_severity', 'exploited']).size().unstack(fill_value=0)
        jp_severity.plot(kind='bar', ax=axes[1, 0], stacked=False)
        axes[1, 0].set_xlabel('CVSS Severity', fontsize=12)
        axes[1, 0].set_ylabel('Count', fontsize=12)
        axes[1, 0].set_title('Japan - Severity Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend(['Not Exploited', 'Exploited'])
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    if 'cvss_base_severity' in nonjp_data.columns:
        nonjp_severity = nonjp_data.groupby(['cvss_base_severity', 'exploited']).size().unstack(fill_value=0)
        nonjp_severity.plot(kind='bar', ax=axes[1, 1], stacked=False)
        axes[1, 1].set_xlabel('CVSS Severity', fontsize=12)
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Non-Japan - Severity Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].legend(['Not Exploited', 'Exploited'])
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cvss_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'cvss_analysis.png'}")


def create_temporal_trends_plot(jp_data, nonjp_data, output_dir):
    """Create temporal trends visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Japan
    if 'year' in jp_data.columns:
        jp_year = jp_data.groupby('year')['exploited'].agg(['count', 'sum', 'mean'])
        jp_year.columns = ['total', 'exploited', 'rate']
        axes[0].plot(jp_year.index, jp_year['rate'] * 100, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('Exploitation Rate (%)', fontsize=12)
        axes[0].set_title('Japan - Exploitation Rate by Year', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 100])
    
    # Non-Japan
    if 'year' in nonjp_data.columns:
        nonjp_year = nonjp_data.groupby('year')['exploited'].agg(['count', 'sum', 'mean'])
        nonjp_year.columns = ['total', 'exploited', 'rate']
        axes[1].plot(nonjp_year.index, nonjp_year['rate'] * 100, marker='o', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Year', fontsize=12)
        axes[1].set_ylabel('Exploitation Rate (%)', fontsize=12)
        axes[1].set_title('Non-Japan - Exploitation Rate by Year', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'temporal_trends.png'}")


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / 'results'
    
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Load feature importance
    print("\n[1] Loading feature importance data...")
    jp_importance_df = pd.read_csv(results_dir / 'jp_feature_importance.csv')
    nonjp_importance_df = pd.read_csv(results_dir / 'nonjp_feature_importance.csv')
    
    # Load original data
    print("\n[2] Loading CVE data...")
    jp_file = project_root / 'results' / 'jp_2021_2025_three_categories.csv'
    nonjp_file = project_root / 'results' / 'nonjp_2021_2025_three_categories_sample_6000.csv'
    cve_features_file = project_root / 'results' / 'cve_features_2021_2025.csv'
    
    jp_df = pd.read_csv(jp_file)
    nonjp_df = pd.read_csv(nonjp_file)
    cve_features_df = pd.read_csv(cve_features_file, low_memory=False)
    
    jp_merged = jp_df.merge(cve_features_df, on='cve_id', how='left')
    nonjp_merged = nonjp_df.merge(cve_features_df, on='cve_id', how='left')
    
    jp_merged['exploited'] = ((jp_merged['in_kev'] == True) | (jp_merged['has_indicators'] == True)).astype(int)
    nonjp_merged['exploited'] = ((nonjp_merged['in_kev'] == True) | (nonjp_merged['has_indicators'] == True)).astype(int)
    
    # Create visualizations
    print("\n[3] Creating visualizations...")
    
    create_feature_importance_plot(jp_importance_df, nonjp_importance_df, results_dir)
    create_cvss_analysis_plot(jp_merged, nonjp_merged, results_dir)
    create_temporal_trends_plot(jp_merged, nonjp_merged, results_dir)
    
    print("\n" + "=" * 80)
    print("✅ All visualizations created!")
    print("=" * 80)


if __name__ == '__main__':
    main()

