#!/usr/bin/env python3
"""
Compare Tuning Impact Script
Trains Default vs Tuned models for Japan/Non-Japan and outputs comparison.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import csv

csv.field_size_limit(10000000)
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# Feature Engineering Logic (Duplicated from tune_model.py for standalone run)
# -------------------------------------------------------------------------

def calculate_vendor_popularity(df):
    vendor_counts = df['vendor'].value_counts()
    vendor_popularity = df['vendor'].map(vendor_counts)
    return vendor_popularity.fillna(0)

def create_features(data, is_japanese=False, scaler_obj=None):
    data = data.reset_index(drop=True)
    feature_dfs = []
    
    categorical_features = ['vendor', 'product', 'cvss_base_severity']
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            dummies.index = data.index
            feature_dfs.append(dummies)
    
    if 'year' in data.columns:
        year_dummies = pd.get_dummies(data['year'], prefix='year', drop_first=True)
        year_dummies.index = data.index
        feature_dfs.append(year_dummies)
    
    cvss_categorical = [
        'cvss_attack_vector', 'cvss_attack_complexity', 'cvss_privileges_required',
        'cvss_user_interaction', 'cvss_confidentiality_impact', 'cvss_integrity_impact',
        'cvss_availability_impact'
    ]
    for col in cvss_categorical:
        if col in data.columns:
            data[col] = data[col].fillna('UNKNOWN')
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            dummies.index = data.index
            feature_dfs.append(dummies)
    
    if 'cwe_ids' in data.columns:
        all_cwes = []
        for cwe_str in data['cwe_ids'].dropna():
            if isinstance(cwe_str, str):
                cwes = [c.strip() for c in cwe_str.split('|') if c.strip()]
                all_cwes.extend(cwes)
        if all_cwes:
            top_cwes = [cwe for cwe, count in Counter(all_cwes).most_common(20)]
            cwe_features = {}
            for cwe in top_cwes:
                cwe_clean = cwe.replace('-', '_').replace(' ', '_')
                has_cwe = data['cwe_ids'].str.contains(cwe, na=False, regex=False).astype(int)
                cwe_features[f'has_cwe_{cwe_clean}'] = has_cwe
            if cwe_features:
                cwe_df = pd.DataFrame(cwe_features, index=data.index)
                feature_dfs.append(cwe_df)
    
    if 'reference_tags' in data.columns:
        has_exploit_tag = data['reference_tags'].str.contains('exploit', case=False, na=False, regex=False).astype(int)
        exploit_tag_df = pd.DataFrame({'has_exploit_reference': has_exploit_tag}, index=data.index)
        feature_dfs.append(exploit_tag_df)
    
    vendor_popularity = calculate_vendor_popularity(data)
    cvss_scores = pd.to_numeric(data.get('cvss_base_score', 0), errors='coerce')
    has_cvss_score = cvss_scores.notna().astype(int)
    cvss_median = cvss_scores.median()
    if 'year' in data.columns:
        year_medians = cvss_scores.groupby(data['year']).median()
        cvss_imputed = cvss_scores.copy()
        for year in data['year'].unique():
            year_mask = (data['year'] == year) & cvss_scores.isna()
            if year in year_medians.index and pd.notna(year_medians[year]):
                cvss_imputed[year_mask] = year_medians[year]
            else:
                cvss_imputed[year_mask] = cvss_median
        cvss_imputed = cvss_imputed.fillna(cvss_median)
    else:
        cvss_imputed = cvss_scores.fillna(cvss_median)
    
    numerical_features = {
        'cvss_score': cvss_imputed,
        'has_cvss_score': has_cvss_score,
        'vendor_popularity': vendor_popularity
    }
    
    if 'reference_count' in data.columns:
        numerical_features['reference_count'] = pd.to_numeric(data['reference_count'], errors='coerce').fillna(0)
    if 'has_metrics' in data.columns:
        numerical_features['has_metrics'] = pd.to_numeric(data['has_metrics'], errors='coerce').fillna(0).astype(int)
    
    numerical_data = pd.DataFrame(numerical_features, index=data.index)
    
    if scaler_obj is None:
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(numerical_data)
    else:
        scaler = scaler_obj
        numerical_scaled = scaler.transform(numerical_data)
    
    numerical_df = pd.DataFrame(numerical_scaled, columns=numerical_data.columns, index=data.index)
    feature_dfs.append(numerical_df)
    
    X = pd.concat(feature_dfs, axis=1) if feature_dfs else numerical_df
    X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X.columns]
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    return X, scaler

def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    return {
        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0)
    }

def print_comparison(model_name, region, metrics_default, metrics_tuned):
    print(f"\nModel: {model_name} | Region: {region}")
    print(f"{'-'*85}")
    print(f"{'Metric':<15} | {'Default':<15} | {'Tuned':<15} | {'Improvement':<15}")
    print(f"{'-'*85}")
    for metric in ['ROC-AUC', 'F1-Score', 'Accuracy', 'Precision', 'Recall']:
        def_val = metrics_default[metric]
        tuned_val = metrics_tuned[metric]
        diff = tuned_val - def_val
        print(f"{metric:<15} | {def_val:.4f}          | {tuned_val:.4f}          | {diff:+.4f}")
    print(f"{'-'*85}")

# -------------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------------

def main():
    project_root = Path(__file__).parent.parent
    jp_file = project_root / 'results' / 'jp_2021_2025_three_categories.csv'
    nonjp_file = project_root / 'results' / 'nonjp_2021_2025_three_categories_sample_6000.csv'
    cve_features_file = project_root / 'results' / 'cve_features_2021_2025.csv'
    
    print("Loading data...")
    df_jp = pd.read_csv(jp_file)
    df_nonjp = pd.read_csv(nonjp_file)
    cve_features_df = pd.read_csv(cve_features_file, low_memory=False)
    
    # Process Japan
    print("Processing Japan Data...")
    merged_jp = df_jp.merge(cve_features_df, on='cve_id', how='left')
    merged_jp['exploited'] = ((merged_jp['in_kev'] == True) | (merged_jp['has_indicators'] == True)).astype(int)
    X_jp, _ = create_features(merged_jp, is_japanese=True)
    y_jp = merged_jp['exploited'].values
    X_jp_train, X_jp_test, y_jp_train, y_jp_test = train_test_split(X_jp, y_jp, test_size=0.2, random_state=42, stratify=y_jp)
    smote_jp = SMOTE(random_state=42)
    X_jp_train_smote, y_jp_train_smote = smote_jp.fit_resample(X_jp_train, y_jp_train)
    
    # Process Non-Japan
    print("Processing Non-Japan Data...")
    merged_nonjp = df_nonjp.merge(cve_features_df, on='cve_id', how='left')
    merged_nonjp['exploited'] = ((merged_nonjp['in_kev'] == True) | (merged_nonjp['has_indicators'] == True)).astype(int)
    X_nonjp, _ = create_features(merged_nonjp, is_japanese=False)
    y_nonjp = merged_nonjp['exploited'].values
    X_nonjp_train, X_nonjp_test, y_nonjp_train, y_nonjp_test = train_test_split(X_nonjp, y_nonjp, test_size=0.2, random_state=42, stratify=y_nonjp)
    smote_nonjp = SMOTE(random_state=42)
    X_nonjp_train_smote, y_nonjp_train_smote = smote_nonjp.fit_resample(X_nonjp_train, y_nonjp_train)
    
    regions = [
        ('Japan', X_jp_train_smote, y_jp_train_smote, X_jp_test, y_jp_test),
        ('Non-Japan', X_nonjp_train_smote, y_nonjp_train_smote, X_nonjp_test, y_nonjp_test)
    ]
    
    # -------------------------------------------------------------------------
    # Models to Compare
    # -------------------------------------------------------------------------
    
    # Random Forest
    for region_name, X_train, y_train, X_test, y_test in regions:
        # Default
        rf_def = RandomForestClassifier(random_state=42)
        rf_def.fit(X_train, y_train)
        metrics_def = get_metrics(rf_def, X_test, y_test)
        
        # Tuned (Same params for both regions based on tuning results: n_estimators=200, max_depth=None, min_samples_split=2)
        rf_tuned = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
        rf_tuned.fit(X_train, y_train)
        metrics_tuned = get_metrics(rf_tuned, X_test, y_test)
        
        print_comparison("Random Forest", region_name, metrics_def, metrics_tuned)

    # Logistic Regression
    for region_name, X_train, y_train, X_test, y_test in regions:
        # Default
        lr_def = LogisticRegression(random_state=42, max_iter=1000)
        lr_def.fit(X_train, y_train)
        metrics_def = get_metrics(lr_def, X_test, y_test)
        
        # Tuned (C=10.0, penalty='l2', solver='liblinear')
        lr_tuned = LogisticRegression(C=10.0, penalty='l2', solver='liblinear', random_state=42, max_iter=1000)
        lr_tuned.fit(X_train, y_train)
        metrics_tuned = get_metrics(lr_tuned, X_test, y_test)
        
        print_comparison("Logistic Regression", region_name, metrics_def, metrics_tuned)

    # Gradient Boosting
    for region_name, X_train, y_train, X_test, y_test in regions:
        # Default (sklearn defaults: learning_rate=0.1, n_estimators=100, max_depth=3)
        gb_def = GradientBoostingClassifier(random_state=42)
        gb_def.fit(X_train, y_train)
        metrics_def = get_metrics(gb_def, X_test, y_test)
        
        # Tuned (Region-specific)
        if region_name == 'Japan':
            # Japan Best: learning_rate=0.2, max_depth=7, n_estimators=300, subsample=0.8
            gb_tuned = GradientBoostingClassifier(n_estimators=300, max_depth=7, learning_rate=0.2, subsample=0.8, random_state=42)
        else:
            # Non-Japan Best: learning_rate=0.2, max_depth=9, n_estimators=300, subsample=0.8
            gb_tuned = GradientBoostingClassifier(n_estimators=300, max_depth=9, learning_rate=0.2, subsample=0.8, random_state=42)
            
        gb_tuned.fit(X_train, y_train)
        metrics_tuned = get_metrics(gb_tuned, X_test, y_test)
        
        print_comparison("Gradient Boosting", region_name, metrics_def, metrics_tuned)

if __name__ == "__main__":
    main()
