#!/usr/bin/env python3
"""
Example: Feature creation WITHOUT TF-IDF
Shows how to do feature importance analysis using only structured features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def create_features_no_tfidf(data):
    """
    Create feature matrix WITHOUT TF-IDF - only structured features.
    """
    data = data.reset_index(drop=True)
    
    # Categorical features (one-hot encoding)
    categorical_features = ['vendor', 'product', 'cvss_base_severity']
    feature_df = pd.DataFrame(index=data.index)
    
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            feature_df = pd.concat([feature_df, dummies], axis=1)
    
    # Numerical features
    vendor_counts = data['vendor'].value_counts()
    vendor_popularity = data['vendor'].map(vendor_counts).fillna(0)
    
    numerical_data = pd.DataFrame({
        'cvss_score': pd.to_numeric(data.get('cvss_base_score', 0), errors='coerce').fillna(0),
        'vendor_popularity': vendor_popularity,
        'reference_count': pd.to_numeric(data.get('reference_count', 0), errors='coerce').fillna(0),
        'has_metrics': pd.to_numeric(data.get('has_metrics', 0), errors='coerce').fillna(0),
        'has_affected': pd.to_numeric(data.get('has_affected', 0), errors='coerce').fillna(0),
        'has_references': pd.to_numeric(data.get('has_references', 0), errors='coerce').fillna(0),
    }, index=data.index)
    
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_data)
    numerical_df = pd.DataFrame(numerical_scaled, columns=numerical_data.columns, index=data.index)
    
    # Combine all features
    X = pd.concat([feature_df, numerical_df], axis=1)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    return X, scaler


def main():
    """Demonstrate feature creation without TF-IDF."""
    project_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("FEATURE CREATION WITHOUT TF-IDF - DEMONSTRATION")
    print("=" * 80)
    
    # Load data
    jp_file = project_root / 'results' / 'jp_2021_2025_three_categories.csv'
    cve_features_file = project_root / 'results' / 'cve_features_2021_2025.csv'
    
    jp_df = pd.read_csv(jp_file)
    cve_features_df = pd.read_csv(cve_features_file, low_memory=False)
    jp_merged = jp_df.merge(cve_features_df, on='cve_id', how='left')
    
    # Create features WITHOUT TF-IDF
    print("\nCreating features WITHOUT TF-IDF...")
    X_no_tfidf, scaler = create_features_no_tfidf(jp_merged)
    
    print(f"\nFeature matrix shape (no TF-IDF): {X_no_tfidf.shape}")
    print(f"Number of features: {X_no_tfidf.shape[1]}")
    print(f"\nFeature types:")
    print(f"  - Categorical (one-hot): {len([c for c in X_no_tfidf.columns if any(x in c for x in ['vendor_', 'product_', 'cvss_base_severity_'])])}")
    print(f"  - Numerical: {len([c for c in X_no_tfidf.columns if c in ['cvss_score', 'vendor_popularity', 'reference_count', 'has_metrics', 'has_affected', 'has_references']])}")
    
    print(f"\nSample features:")
    print(X_no_tfidf.columns[:20].tolist())
    
    print("\n" + "=" * 80)
    print("✅ Demonstration complete!")
    print("\nNote: Feature importance analysis works the same way with these features.")
    print("You can train models and get feature importance without TF-IDF.")
    print("=" * 80)


if __name__ == '__main__':
    main()

