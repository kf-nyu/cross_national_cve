#!/usr/bin/env python3
"""
Further Investigation of ML Results
- Feature importance analysis
- Error analysis (false positives/negatives)
- Comparison between Japan and Non-Japan patterns
- Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
import sys

# Increase CSV field size limit
import csv
csv.field_size_limit(10000000)

warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Import feature engineering functions from main script
sys.path.insert(0, str(Path(__file__).parent))
from cross_national_ml_analysis import (
    create_features, improved_english_tokenizer, improved_japanese_tokenizer,
    calculate_vendor_popularity, CYBERSECURITY_TERMS, EXTENDED_ENGLISH_STOPWORDS
)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def get_feature_importance(model, feature_names, model_name="Model"):
    """Extract feature importance from trained model."""
    importances = {}
    
    if hasattr(model, 'feature_importances_'):
        # Random Forest, XGBoost
        importances_dict = dict(zip(feature_names, model.feature_importances_))
        importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
    elif hasattr(model, 'coef_'):
        # Logistic Regression
        coef = model.coef_[0]
        importances_dict = dict(zip(feature_names, np.abs(coef)))
        importances = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)
    
    return importances


def analyze_errors(model, X_test, y_test, test_data, model_name="Model"):
    """Analyze false positives and false negatives."""
    y_pred = model.predict(X_test)
    
    # Get indices of errors
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]  # False Positives
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]  # False Negatives
    
    fp_data = test_data.iloc[fp_indices] if len(fp_indices) > 0 else pd.DataFrame()
    fn_data = test_data.iloc[fn_indices] if len(fn_indices) > 0 else pd.DataFrame()
    
    return fp_data, fn_data


def analyze_by_dimension(data, dimension_col, exploited_col='exploited'):
    """Analyze exploitation rates by different dimensions."""
    analysis = {}
    
    if dimension_col in data.columns:
        grouped = data.groupby(dimension_col)[exploited_col].agg(['count', 'sum', 'mean'])
        grouped.columns = ['total', 'exploited', 'exploitation_rate']
        grouped = grouped.sort_values('exploitation_rate', ascending=False)
        analysis[dimension_col] = grouped
    
    return analysis


def compare_feature_importance(jp_importances, nonjp_importances, top_n=30):
    """Compare feature importance between Japan and Non-Japan models."""
    # Convert to dictionaries for easier lookup
    jp_dict = {f: imp for f, imp in jp_importances}
    nonjp_dict = {f: imp for f, imp in nonjp_importances}
    
    # Get all unique features
    all_features = set(jp_dict.keys()) | set(nonjp_dict.keys())
    
    # Create comparison dataframe
    comparison = []
    for feature in all_features:
        jp_imp = jp_dict.get(feature, 0)
        nonjp_imp = nonjp_dict.get(feature, 0)
        diff = jp_imp - nonjp_imp
        ratio = jp_imp / nonjp_imp if nonjp_imp > 0 else (jp_imp / 0.0001 if jp_imp > 0 else 1)
        
        # Categorize feature
        category = categorize_feature(feature)
        
        comparison.append({
            'feature': feature,
            'jp_importance': jp_imp,
            'nonjp_importance': nonjp_imp,
            'difference': diff,
            'ratio': ratio,
            'category': category
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Sort by absolute difference
    comparison_df['abs_difference'] = comparison_df['difference'].abs()
    comparison_df = comparison_df.sort_values('abs_difference', ascending=False)
    
    return comparison_df


def categorize_feature(feature_name):
    """Categorize a feature by its name."""
    feature_lower = feature_name.lower()
    
    if 'cvss' in feature_lower:
        if 'attack' in feature_lower or 'complexity' in feature_lower or 'privileges' in feature_lower or 'user_interaction' in feature_lower:
            return 'CVSS_Exploitability'
        elif 'confidentiality' in feature_lower or 'integrity' in feature_lower or 'availability' in feature_lower:
            return 'CVSS_Impact'
        elif 'severity' in feature_lower:
            return 'CVSS_Severity'
        else:
            return 'CVSS_Score'
    elif 'vendor' in feature_lower:
        return 'Vendor'
    elif 'product' in feature_lower:
        return 'Product'
    elif 'cwe' in feature_lower:
        return 'CWE'
    elif 'reference' in feature_lower:
        return 'Reference'
    elif 'year' in feature_lower:
        return 'Temporal'
    elif 'exploit' in feature_lower:
        return 'Exploit_Indicator'
    else:
        return 'Other'


def create_comparison_visualizations(comparison_df, jp_importances, nonjp_importances, results_dir):
    """Create visualizations comparing Japan vs Non-Japan feature importance."""
    
    # 1. Top features comparison - side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    top_n = 20
    jp_top = jp_importances[:top_n]
    nonjp_top = nonjp_importances[:top_n]
    
    # Japan
    jp_features = [f[0] for f in jp_top]
    jp_imps = [f[1] for f in jp_top]
    axes[0].barh(range(len(jp_features)), jp_imps, color='#FF6B6B')
    axes[0].set_yticks(range(len(jp_features)))
    axes[0].set_yticklabels([f[:40] + '...' if len(f) > 40 else f for f in jp_features], fontsize=8)
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_title('Japan - Top 20 Features', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Non-Japan
    nonjp_features = [f[0] for f in nonjp_top]
    nonjp_imps = [f[1] for f in nonjp_top]
    axes[1].barh(range(len(nonjp_features)), nonjp_imps, color='#4ECDC4')
    axes[1].set_yticks(range(len(nonjp_features)))
    axes[1].set_yticklabels([f[:40] + '...' if len(f) > 40 else f for f in nonjp_features], fontsize=8)
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_title('Non-Japan - Top 20 Features', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Difference plot - features with largest differences
    top_diff = comparison_df.head(30)
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = ['#FF6B6B' if d > 0 else '#4ECDC4' for d in top_diff['difference']]
    bars = ax.barh(range(len(top_diff)), top_diff['difference'], color=colors)
    ax.set_yticks(range(len(top_diff)))
    ax.set_yticklabels([f[:50] + '...' if len(f) > 50 else f for f in top_diff['feature']], fontsize=8)
    ax.set_xlabel('Importance Difference (Japan - Non-Japan)', fontsize=12)
    ax.set_title('Top 30 Features with Largest JP vs Non-JP Differences', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='More Important for Japan'),
        Patch(facecolor='#4ECDC4', label='More Important for Non-Japan')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_difference_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Category comparison
    category_comparison = comparison_df.groupby('category').agg({
        'jp_importance': 'sum',
        'nonjp_importance': 'sum',
        'difference': 'sum'
    }).sort_values('difference', key=abs, ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(category_comparison))
    width = 0.35
    
    ax.bar(x - width/2, category_comparison['jp_importance'], width, label='Japan', color='#FF6B6B')
    ax.bar(x + width/2, category_comparison['nonjp_importance'], width, label='Non-Japan', color='#4ECDC4')
    
    ax.set_xlabel('Feature Category', fontsize=12)
    ax.set_ylabel('Total Importance', fontsize=12)
    ax.set_title('Feature Importance by Category: Japan vs Non-Japan', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(category_comparison.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'category_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Created visualization: feature_importance_comparison.png")
    print("  ✅ Created visualization: feature_difference_comparison.png")
    print("  ✅ Created visualization: category_importance_comparison.png")


def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("FURTHER INVESTIGATION OF ML RESULTS")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading data...")
    jp_file = project_root / 'results' / 'jp_2021_2025_three_categories.csv'
    nonjp_file = project_root / 'results' / 'nonjp_2021_2025_three_categories_sample_6000.csv'
    cve_features_file = project_root / 'results' / 'cve_features_2021_2025.csv'
    
    jp_df = pd.read_csv(jp_file)
    nonjp_df = pd.read_csv(nonjp_file)
    cve_features_df = pd.read_csv(cve_features_file, low_memory=False)
    
    # Merge with features
    jp_merged = jp_df.merge(cve_features_df, on='cve_id', how='left')
    nonjp_merged = nonjp_df.merge(cve_features_df, on='cve_id', how='left')
    
    # Create target variable
    jp_merged['exploited'] = ((jp_merged['in_kev'] == True) | (jp_merged['has_indicators'] == True)).astype(int)
    nonjp_merged['exploited'] = ((nonjp_merged['in_kev'] == True) | (nonjp_merged['has_indicators'] == True)).astype(int)
    
    # Feature engineering
    print("\n[2] Feature Engineering...")
    X_jp, tfidf_jp, scaler_jp = create_features(jp_merged, is_japanese=True)
    y_jp = jp_merged['exploited'].values
    
    X_nonjp, tfidf_nonjp, scaler_nonjp = create_features(nonjp_merged, is_japanese=False)
    y_nonjp = nonjp_merged['exploited'].values
    
    # Train-test split
    X_jp_train, X_jp_test, y_jp_train, y_jp_test = train_test_split(
        X_jp, y_jp, test_size=0.2, random_state=42, stratify=y_jp
    )
    jp_test_indices = X_jp_test.index
    jp_test_data = jp_merged.iloc[jp_test_indices].reset_index(drop=True)
    
    X_nonjp_train, X_nonjp_test, y_nonjp_train, y_nonjp_test = train_test_split(
        X_nonjp, y_nonjp, test_size=0.2, random_state=42, stratify=y_nonjp
    )
    nonjp_test_indices = X_nonjp_test.index
    nonjp_test_data = nonjp_merged.iloc[nonjp_test_indices].reset_index(drop=True)
    
    # Train best models
    print("\n[3] Training Best Models...")
    
    # Japan - Random Forest (best ROC-AUC)
    print("  Training Japan Random Forest...")
    smote_jp = SMOTE(random_state=42)
    X_jp_train_smote, y_jp_train_smote = smote_jp.fit_resample(X_jp_train, y_jp_train)
    rf_jp = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, random_state=42)
    rf_jp.fit(X_jp_train_smote, y_jp_train_smote)
    
    # Non-Japan - XGBoost (best overall)
    print("  Training Non-Japan XGBoost...")
    smote_nonjp = SMOTE(random_state=42)
    X_nonjp_train_smote, y_nonjp_train_smote = smote_nonjp.fit_resample(X_nonjp_train, y_nonjp_train)
    xgb_nonjp = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
    xgb_nonjp.fit(X_nonjp_train_smote, y_nonjp_train_smote)
    
    # Feature Importance Analysis
    print("\n[4] Feature Importance Analysis...")
    
    jp_importances = get_feature_importance(rf_jp, X_jp.columns, "Japan Random Forest")
    nonjp_importances = get_feature_importance(xgb_nonjp, X_nonjp.columns, "Non-Japan XGBoost")
    
    print("\n" + "=" * 80)
    print("TOP 20 FEATURES - JAPAN")
    print("=" * 80)
    for i, (feature, importance) in enumerate(jp_importances[:20], 1):
        print(f"{i:2d}. {feature:<50} {importance:.6f}")
    
    print("\n" + "=" * 80)
    print("TOP 20 FEATURES - NON-JAPAN")
    print("=" * 80)
    for i, (feature, importance) in enumerate(nonjp_importances[:20], 1):
        print(f"{i:2d}. {feature:<50} {importance:.6f}")
    
    # Comparative Analysis - Highlighting Differences
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS: JAPAN vs NON-JAPAN DIFFERENCES")
    print("=" * 80)
    
    comparison_df = compare_feature_importance(jp_importances, nonjp_importances)
    
    # Features more important for Japan
    jp_favored = comparison_df[comparison_df['difference'] > 0].head(15)
    print("\n🔴 TOP 15 FEATURES MORE IMPORTANT FOR JAPAN:")
    print("-" * 80)
    for i, row in enumerate(jp_favored.itertuples(), 1):
        print(f"{i:2d}. {row.feature:<50} JP:{row.jp_importance:.6f} NonJP:{row.nonjp_importance:.6f} (Diff: +{row.difference:.6f})")
    
    # Features more important for Non-Japan
    nonjp_favored = comparison_df[comparison_df['difference'] < 0].head(15)
    print("\n🔵 TOP 15 FEATURES MORE IMPORTANT FOR NON-JAPAN:")
    print("-" * 80)
    for i, row in enumerate(nonjp_favored.itertuples(), 1):
        print(f"{i:2d}. {row.feature:<50} JP:{row.jp_importance:.6f} NonJP:{row.nonjp_importance:.6f} (Diff: {row.difference:.6f})")
    
    # Category-level differences
    print("\n" + "=" * 80)
    print("CATEGORY-LEVEL DIFFERENCES")
    print("=" * 80)
    category_summary = comparison_df.groupby('category').agg({
        'jp_importance': 'sum',
        'nonjp_importance': 'sum',
        'difference': 'sum'
    }).sort_values('difference', key=lambda x: x.abs(), ascending=False)
    
    print(f"\n{'Category':<25} {'Japan Total':<15} {'Non-JP Total':<15} {'Difference':<15}")
    print("-" * 70)
    for cat, row in category_summary.iterrows():
        diff_str = f"{row['difference']:+.6f}"
        print(f"{cat:<25} {row['jp_importance']:<15.6f} {row['nonjp_importance']:<15.6f} {diff_str:<15}")
    
    # Unique features (only in one model)
    jp_features_set = set([f[0] for f in jp_importances[:50]])
    nonjp_features_set = set([f[0] for f in nonjp_importances[:50]])
    
    jp_unique = jp_features_set - nonjp_features_set
    nonjp_unique = nonjp_features_set - jp_features_set
    
    print("\n" + "=" * 80)
    print("UNIQUE FEATURES (Top 50)")
    print("=" * 80)
    print(f"\n🔴 Features ONLY in Japan Top 50: {len(jp_unique)}")
    if jp_unique:
        for feat in list(jp_unique)[:10]:
            jp_imp = next((imp for f, imp in jp_importances if f == feat), 0)
            print(f"  - {feat:<50} (Importance: {jp_imp:.6f})")
    
    print(f"\n🔵 Features ONLY in Non-Japan Top 50: {len(nonjp_unique)}")
    if nonjp_unique:
        for feat in list(nonjp_unique)[:10]:
            nonjp_imp = next((imp for f, imp in nonjp_importances if f == feat), 0)
            print(f"  - {feat:<50} (Importance: {nonjp_imp:.6f})")
    
    # Error Analysis
    print("\n[5] Error Analysis...")
    
    jp_fp, jp_fn = analyze_errors(rf_jp, X_jp_test, y_jp_test, jp_test_data, "Japan Random Forest")
    nonjp_fp, nonjp_fn = analyze_errors(xgb_nonjp, X_nonjp_test, y_nonjp_test, nonjp_test_data, "Non-Japan XGBoost")
    
    print(f"\nJapan Random Forest Errors:")
    print(f"  False Positives: {len(jp_fp)}")
    print(f"  False Negatives: {len(jp_fn)}")
    
    print(f"\nNon-Japan XGBoost Errors:")
    print(f"  False Positives: {len(nonjp_fp)}")
    print(f"  False Negatives: {len(nonjp_fn)}")
    
    # Analyze errors by dimensions
    print("\n[6] Error Analysis by Dimensions...")
    
    if len(jp_fp) > 0:
        print("\nJapan False Positives Analysis:")
        print(f"  Average CVSS Score: {jp_fp['cvss_base_score'].fillna(0).mean():.2f}")
        print(f"  Top Vendors: {jp_fp['vendor'].value_counts().head(5).to_dict()}")
    
    if len(jp_fn) > 0:
        print("\nJapan False Negatives Analysis:")
        print(f"  Average CVSS Score: {jp_fn['cvss_base_score'].fillna(0).mean():.2f}")
        print(f"  Top Vendors: {jp_fn['vendor'].value_counts().head(5).to_dict()}")
    
    if len(nonjp_fp) > 0:
        print("\nNon-Japan False Positives Analysis:")
        print(f"  Average CVSS Score: {nonjp_fp['cvss_base_score'].fillna(0).mean():.2f}")
        print(f"  Top Vendors: {nonjp_fp['vendor'].value_counts().head(5).to_dict()}")
    
    if len(nonjp_fn) > 0:
        print("\nNon-Japan False Negatives Analysis:")
        print(f"  Average CVSS Score: {nonjp_fn['cvss_base_score'].fillna(0).mean():.2f}")
        print(f"  Top Vendors: {nonjp_fn['vendor'].value_counts().head(5).to_dict()}")
    
    # Analysis by dimensions
    print("\n[7] Exploitation Patterns by Dimensions...")
    
    print("\nJapan - CVSS Severity Distribution:")
    if 'cvss_base_severity' in jp_merged.columns:
        severity_analysis = analyze_by_dimension(jp_merged, 'cvss_base_severity')
        if 'cvss_base_severity' in severity_analysis:
            print(severity_analysis['cvss_base_severity'])
    
    print("\nNon-Japan - CVSS Severity Distribution:")
    if 'cvss_base_severity' in nonjp_merged.columns:
        severity_analysis = analyze_by_dimension(nonjp_merged, 'cvss_base_severity')
        if 'cvss_base_severity' in severity_analysis:
            print(severity_analysis['cvss_base_severity'])
    
    # Vendor analysis
    print("\n[8] Top Vendors Analysis...")
    
    print("\nJapan - Top 10 Vendors by Exploitation Rate:")
    jp_vendor_analysis = analyze_by_dimension(jp_merged, 'vendor')
    if 'vendor' in jp_vendor_analysis:
        print(jp_vendor_analysis['vendor'].head(10))
    
    print("\nNon-Japan - Top 10 Vendors by Exploitation Rate:")
    nonjp_vendor_analysis = analyze_by_dimension(nonjp_merged, 'vendor')
    if 'vendor' in nonjp_vendor_analysis:
        print(nonjp_vendor_analysis['vendor'].head(10))
    
    # Year analysis
    print("\n[9] Temporal Trends...")
    
    print("\nJapan - Exploitation Rate by Year:")
    jp_year_analysis = analyze_by_dimension(jp_merged, 'year')
    if 'year' in jp_year_analysis:
        print(jp_year_analysis['year'])
    
    print("\nNon-Japan - Exploitation Rate by Year:")
    nonjp_year_analysis = analyze_by_dimension(nonjp_merged, 'year')
    if 'year' in nonjp_year_analysis:
        print(nonjp_year_analysis['year'])
    
    # Create visualizations
    print("\n[10] Creating Comparison Visualizations...")
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    create_comparison_visualizations(comparison_df, jp_importances, nonjp_importances, results_dir)
    
    # Save results
    print("\n[11] Saving Investigation Results...")
    
    # Save feature importance
    jp_importance_df = pd.DataFrame(jp_importances[:50], columns=['feature', 'importance'])
    jp_importance_df.to_csv(results_dir / 'jp_feature_importance.csv', index=False)
    
    nonjp_importance_df = pd.DataFrame(nonjp_importances[:50], columns=['feature', 'importance'])
    nonjp_importance_df.to_csv(results_dir / 'nonjp_feature_importance.csv', index=False)
    
    # Save comparison dataframe
    comparison_df.to_csv(results_dir / 'feature_importance_comparison.csv', index=False)
    print(f"  ✅ Saved: feature_importance_comparison.csv")
    
    # Save error analysis
    if len(jp_fp) > 0:
        jp_fp.to_csv(results_dir / 'jp_false_positives.csv', index=False)
    if len(jp_fn) > 0:
        jp_fn.to_csv(results_dir / 'jp_false_negatives.csv', index=False)
    if len(nonjp_fp) > 0:
        nonjp_fp.to_csv(results_dir / 'nonjp_false_positives.csv', index=False)
    if len(nonjp_fn) > 0:
        nonjp_fn.to_csv(results_dir / 'nonjp_false_negatives.csv', index=False)
    
    # Save summary JSON
    summary = {
        'jp_feature_importance_top10': [{'feature': f, 'importance': float(i)} for f, i in jp_importances[:10]],
        'nonjp_feature_importance_top10': [{'feature': f, 'importance': float(i)} for f, i in nonjp_importances[:10]],
        'jp_errors': {'false_positives': len(jp_fp), 'false_negatives': len(jp_fn)},
        'nonjp_errors': {'false_positives': len(nonjp_fp), 'false_negatives': len(nonjp_fn)},
        'comparison': {
            'top_jp_favored_features': [
                {'feature': row.feature, 'jp_importance': float(row.jp_importance), 
                 'nonjp_importance': float(row.nonjp_importance), 'difference': float(row.difference)}
                for row in comparison_df[comparison_df['difference'] > 0].head(10).itertuples()
            ],
            'top_nonjp_favored_features': [
                {'feature': row.feature, 'jp_importance': float(row.jp_importance), 
                 'nonjp_importance': float(row.nonjp_importance), 'difference': float(row.difference)}
                for row in comparison_df[comparison_df['difference'] < 0].head(10).itertuples()
            ],
            'category_differences': {
                cat: {
                    'jp_total': float(row['jp_importance']),
                    'nonjp_total': float(row['nonjp_importance']),
                    'difference': float(row['difference'])
                }
                for cat, row in category_summary.iterrows()
            }
        }
    }
    
    with open(results_dir / 'investigation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Investigation results saved to:")
    print(f"   - {results_dir / 'jp_feature_importance.csv'}")
    print(f"   - {results_dir / 'nonjp_feature_importance.csv'}")
    print(f"   - {results_dir / 'investigation_summary.json'}")
    if len(jp_fp) > 0 or len(jp_fn) > 0 or len(nonjp_fp) > 0 or len(nonjp_fn) > 0:
        print(f"   - Error analysis CSV files")
    
    print("\n" + "=" * 80)
    print("✅ Investigation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

