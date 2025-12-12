#!/usr/bin/env python3
"""
Example: Adding Candidate Features to the Model

This script demonstrates how to add candidate features from candidate_features.md
to the feature engineering pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


def calculate_vendor_popularity(df):
    """Calculate vendor popularity based on frequency."""
    vendor_counts = df['vendor'].value_counts()
    vendor_popularity = df['vendor'].map(vendor_counts)
    return vendor_popularity.fillna(0)


def calculate_product_popularity(df):
    """Calculate product popularity based on frequency."""
    product_counts = df['product'].value_counts()
    product_popularity = df['product'].map(product_counts)
    return product_popularity.fillna(0)


def create_cvss_impact_score(df):
    """Create combined CVSS impact score from individual impacts."""
    impact_map = {'NONE': 0, 'LOW': 1, 'HIGH': 2}
    
    conf_impact = df['cvss_confidentiality_impact'].map(impact_map).fillna(0)
    integ_impact = df['cvss_integrity_impact'].map(impact_map).fillna(0)
    avail_impact = df['cvss_availability_impact'].map(impact_map).fillna(0)
    
    # Sum the three impacts (max 6)
    impact_score = conf_impact + integ_impact + avail_impact
    return impact_score


def create_cvss_exploitability_score(df):
    """Create combined CVSS exploitability score."""
    # Attack Vector: NETWORK=3, ADJACENT_NETWORK=2, LOCAL=1, PHYSICAL=0
    av_map = {'NETWORK': 3, 'ADJACENT_NETWORK': 2, 'LOCAL': 1, 'PHYSICAL': 0}
    av_score = df['cvss_attack_vector'].map(av_map).fillna(0)
    
    # Attack Complexity: LOW=1, HIGH=0
    ac_score = (df['cvss_attack_complexity'] == 'LOW').astype(int).fillna(0)
    
    # Privileges Required: NONE=1, LOW=0.5, HIGH=0
    pr_map = {'NONE': 1, 'LOW': 0.5, 'HIGH': 0}
    pr_score = df['cvss_privileges_required'].map(pr_map).fillna(0)
    
    # User Interaction: NONE=1, REQUIRED=0
    ui_score = (df['cvss_user_interaction'] == 'NONE').astype(int).fillna(0)
    
    # Simple weighted combination (higher = more exploitable)
    exploitability_score = (av_score * 0.3 + ac_score * 0.3 + pr_score * 0.2 + ui_score * 0.2)
    return exploitability_score


def create_features_with_candidates(data, include_phase1=True, include_phase2=True, 
                                    include_phase3=False, include_derived=True):
    """
    Create feature matrix with candidate features added.
    
    Parameters:
    -----------
    include_phase1 : bool
        Include Phase 1 features: reference_count, has_metrics, year
    include_phase2 : bool
        Include Phase 2 features: CVSS details (attack_vector, complexity, etc.)
    include_phase3 : bool
        Include Phase 3 features: cwe_ids (top N), reference_tags
    include_derived : bool
        Include derived features: product_popularity, cvss_impact_score, etc.
    """
    data = data.reset_index(drop=True)
    feature_dfs = []
    
    # ===== EXISTING FEATURES =====
    
    # Categorical features (one-hot encoding)
    categorical_features = ['vendor', 'product', 'cvss_base_severity']
    
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            dummies.index = data.index
            feature_dfs.append(dummies)
    
    # Numerical features
    vendor_popularity = calculate_vendor_popularity(data)
    
    # Handle CVSS scores
    cvss_scores = pd.to_numeric(data.get('cvss_base_score', 0), errors='coerce')
    has_cvss_score = cvss_scores.notna().astype(int)
    cvss_median = cvss_scores.median()
    cvss_imputed = cvss_scores.fillna(cvss_median)
    
    numerical_features = {
        'cvss_score': cvss_imputed,
        'has_cvss_score': has_cvss_score,
        'vendor_popularity': vendor_popularity
    }
    
    # ===== PHASE 1: QUICK WINS =====
    if include_phase1:
        print("  Adding Phase 1 features: reference_count, has_metrics, year")
        
        # reference_count
        if 'reference_count' in data.columns:
            numerical_features['reference_count'] = pd.to_numeric(
                data['reference_count'], errors='coerce'
            ).fillna(0)
        
        # has_metrics
        if 'has_metrics' in data.columns:
            numerical_features['has_metrics'] = pd.to_numeric(
                data['has_metrics'], errors='coerce'
            ).fillna(0).astype(int)
        
        # year (as categorical)
        if 'year' in data.columns:
            year_dummies = pd.get_dummies(data['year'], prefix='year', drop_first=True)
            year_dummies.index = data.index
            feature_dfs.append(year_dummies)
    
    # ===== PHASE 2: CVSS DETAILS =====
    if include_phase2:
        print("  Adding Phase 2 features: CVSS attack details")
        
        cvss_categorical = [
            'cvss_attack_vector',
            'cvss_attack_complexity',
            'cvss_privileges_required',
            'cvss_user_interaction'
        ]
        
        for col in cvss_categorical:
            if col in data.columns:
                # Fill missing with 'UNKNOWN'
                data[col] = data[col].fillna('UNKNOWN')
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                dummies.index = data.index
                feature_dfs.append(dummies)
        
        # CVSS impact features
        cvss_impact_features = [
            'cvss_confidentiality_impact',
            'cvss_integrity_impact',
            'cvss_availability_impact'
        ]
        
        for col in cvss_impact_features:
            if col in data.columns:
                data[col] = data[col].fillna('UNKNOWN')
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                dummies.index = data.index
                feature_dfs.append(dummies)
    
    # ===== PHASE 3: ADVANCED FEATURES =====
    if include_phase3:
        print("  Adding Phase 3 features: CWE IDs, reference tags")
        
        # CWE IDs - extract top N most common
        if 'cwe_ids' in data.columns:
            # Get all CWE IDs
            all_cwes = []
            for cwe_str in data['cwe_ids'].dropna():
                if isinstance(cwe_str, str):
                    cwes = [c.strip() for c in cwe_str.split('|') if c.strip()]
                    all_cwes.extend(cwes)
            
            # Get top 20 most common CWEs
            if all_cwes:
                from collections import Counter
                top_cwes = [cwe for cwe, count in Counter(all_cwes).most_common(20)]
                
                # Create binary features for top CWEs
                for cwe in top_cwes:
                    has_cwe = data['cwe_ids'].str.contains(cwe, na=False, regex=False).astype(int)
                    numerical_features[f'has_cwe_{cwe}'] = has_cwe
        
        # Reference tags - check for exploit-related tags
        if 'reference_tags' in data.columns:
            # Check if any reference tag contains "exploit"
            has_exploit_tag = data['reference_tags'].str.contains(
                'exploit', case=False, na=False, regex=False
            ).astype(int)
            numerical_features['has_exploit_reference'] = has_exploit_tag
    
    # ===== DERIVED FEATURES =====
    if include_derived:
        print("  Adding derived features: product_popularity, CVSS scores")
        
        # Product popularity
        product_popularity = calculate_product_popularity(data)
        numerical_features['product_popularity'] = product_popularity
        
        # CVSS impact score
        if include_phase2:
            cvss_impact_score = create_cvss_impact_score(data)
            numerical_features['cvss_impact_score'] = cvss_impact_score
            
            # CVSS exploitability score
            cvss_exploitability_score = create_cvss_exploitability_score(data)
            numerical_features['cvss_exploitability_score'] = cvss_exploitability_score
        
        # Days since published (if date_published available)
        if 'date_published' in data.columns:
            try:
                # Parse dates
                dates = pd.to_datetime(data['date_published'], errors='coerce')
                # Use a reference date (e.g., 2025-01-01 or max date)
                reference_date = dates.max() if dates.notna().any() else pd.Timestamp('2025-01-01')
                days_since = (reference_date - dates).dt.days.fillna(0)
                numerical_features['days_since_published'] = days_since
            except:
                pass
    
    # ===== CREATE NUMERICAL DATAFRAME =====
    numerical_df = pd.DataFrame(numerical_features, index=data.index)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_df)
    numerical_df_scaled = pd.DataFrame(
        numerical_scaled, 
        columns=numerical_df.columns, 
        index=data.index
    )
    feature_dfs.append(numerical_df_scaled)
    
    # ===== COMBINE ALL FEATURES =====
    X = pd.concat(feature_dfs, axis=1) if feature_dfs else numerical_df_scaled
    
    # Clean column names
    X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
                 for col in X.columns]
    
    # Handle missing values
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    return X, scaler


def main():
    """Demonstrate adding candidate features."""
    project_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("ADDING CANDIDATE FEATURES - DEMONSTRATION")
    print("=" * 80)
    
    # Load data
    print("\n[1] Loading data...")
    jp_file = project_root / 'results' / 'jp_2021_2025_three_categories.csv'
    cve_features_file = project_root / 'results' / 'cve_features_2021_2025.csv'
    
    jp_df = pd.read_csv(jp_file)
    cve_features_df = pd.read_csv(cve_features_file, low_memory=False)
    jp_merged = jp_df.merge(cve_features_df, on='cve_id', how='left')
    
    print(f"  Loaded {len(jp_merged):,} Japan CVEs")
    
    # Baseline features (current)
    print("\n[2] Creating baseline features (current)...")
    X_baseline, _ = create_features_with_candidates(
        jp_merged,
        include_phase1=False,
        include_phase2=False,
        include_phase3=False,
        include_derived=False
    )
    print(f"  Baseline features: {X_baseline.shape[1]} features")
    
    # With Phase 1 features
    print("\n[3] Adding Phase 1 features...")
    X_phase1, _ = create_features_with_candidates(
        jp_merged,
        include_phase1=True,
        include_phase2=False,
        include_phase3=False,
        include_derived=False
    )
    print(f"  Phase 1 features: {X_phase1.shape[1]} features (+{X_phase1.shape[1] - X_baseline.shape[1]})")
    
    # With Phase 1 + Phase 2 features
    print("\n[4] Adding Phase 2 features...")
    X_phase2, _ = create_features_with_candidates(
        jp_merged,
        include_phase1=True,
        include_phase2=True,
        include_phase3=False,
        include_derived=False
    )
    print(f"  Phase 1+2 features: {X_phase2.shape[1]} features (+{X_phase2.shape[1] - X_phase1.shape[1]})")
    
    # With all features
    print("\n[5] Adding Phase 3 and derived features...")
    X_all, scaler = create_features_with_candidates(
        jp_merged,
        include_phase1=True,
        include_phase2=True,
        include_phase3=True,
        include_derived=True
    )
    print(f"  All features: {X_all.shape[1]} features (+{X_all.shape[1] - X_phase2.shape[1]})")
    
    # Summary
    print("\n" + "=" * 80)
    print("FEATURE COUNT SUMMARY")
    print("=" * 80)
    print(f"Baseline (current):        {X_baseline.shape[1]:5d} features")
    print(f"+ Phase 1 (quick wins):    {X_phase1.shape[1]:5d} features (+{X_phase1.shape[1] - X_baseline.shape[1]:4d})")
    print(f"+ Phase 2 (CVSS details): {X_phase2.shape[1]:5d} features (+{X_phase2.shape[1] - X_phase1.shape[1]:4d})")
    print(f"+ Phase 3 + Derived:      {X_all.shape[1]:5d} features (+{X_all.shape[1] - X_phase2.shape[1]:4d})")
    
    print("\n" + "=" * 80)
    print("SAMPLE NEW FEATURES:")
    print("=" * 80)
    
    # Show some new features
    new_features = [col for col in X_all.columns if col not in X_baseline.columns]
    print(f"\nTotal new features: {len(new_features)}")
    print(f"\nFirst 20 new features:")
    for feat in new_features[:20]:
        print(f"  - {feat}")
    
    print("\n" + "=" * 80)
    print("✅ Demonstration complete!")
    print("\nNext steps:")
    print("1. Integrate these features into cross_national_ml_analysis.py")
    print("2. Train models with new features")
    print("3. Compare performance with baseline")
    print("4. Analyze feature importance to see which new features matter most")
    print("=" * 80)


if __name__ == '__main__':
    main()

