#!/usr/bin/env python3
"""
Stacking Ensemble Training Script
Builds a StackingClassifier using optimized RF, LR, and GB base learners.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings

# Machine Learning
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import csv

csv.field_size_limit(10000000)
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# Feature Engineering Logic (Standardized)
# -------------------------------------------------------------------------

def calculate_vendor_popularity(df):
    vendor_counts = df['vendor'].value_counts()
    vendor_popularity = df['vendor'].map(vendor_counts)
    return vendor_popularity.fillna(0)

def create_features(data, is_japanese=False, scaler_obj=None):
    data = data.reset_index(drop=True)
    feature_dfs = []
    
    # Categorical features
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

def get_metrics(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else 0
    }
    
    print(f"\n{model_name} Results:")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:<15} : {v:.4f}")
    
    return metrics

# -------------------------------------------------------------------------
# Main Stacking Logic
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
    
    configs = [
        {
            'region': 'Japan',
            'df': df_jp,
            'rf_params': {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'random_state': 42},
            'lr_params': {'C': 10.0, 'penalty': 'l2', 'solver': 'liblinear', 'random_state': 42, 'max_iter': 1000},
            'gb_params': {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.8, 'random_state': 42},
            'xgb_params': {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 1.0, 'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'},
            'nn_params': {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': (100, 50), 'learning_rate_init': 0.001, 'random_state': 42, 'max_iter': 1000}
        },
        {
            'region': 'Non-Japan',
            'df': df_nonjp,
            'rf_params': {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'random_state': 42},
            'lr_params': {'C': 10.0, 'penalty': 'l2', 'solver': 'liblinear', 'random_state': 42, 'max_iter': 1000},
            'gb_params': {'learning_rate': 0.2, 'max_depth': 9, 'n_estimators': 300, 'subsample': 0.8, 'random_state': 42},
            'xgb_params': {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.8, 'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'logloss'},
            'nn_params': {'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.001, 'random_state': 42, 'max_iter': 1000}
        }
    ]
    
    for config in configs:
        region = config['region']
        print(f"\n{'='*60}")
        print(f"Processing Region: {region}")
        print(f"{'='*60}")
        
        # Prepare Data
        merged = config['df'].merge(cve_features_df, on='cve_id', how='left')
        merged['exploited'] = ((merged['in_kev'] == True) | (merged['has_indicators'] == True)).astype(int)
        
        X, _ = create_features(merged, is_japanese=(region == 'Japan'))
        y = merged['exploited'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Define Base Learners (Now 5 Models)
        estimators = [
            ('rf', RandomForestClassifier(**config['rf_params'])),
            ('lr', LogisticRegression(**config['lr_params'])),
            ('gb', GradientBoostingClassifier(**config['gb_params'])),
            ('xgb', XGBClassifier(**config['xgb_params'])),
            ('nn', MLPClassifier(**config['nn_params']))
        ]
        
        # Define Stacking Classifier
        # cv=5 ensures robust meta-feature generation
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            n_jobs=-1
        )
        
        # Train
        print("Training Stacking Ensemble (this may take time)...")
        stacking_clf.fit(X_train_smote, y_train_smote)
        
        # Evaluate Stacking
        metrics_stack = get_metrics(stacking_clf, X_test, y_test, f"Stacking Ensemble ({region})")
        
        # Evaluate Best Single Model (Gradient Boosting typically best) for comparison
        print("\nComparing vs Best Single Model (Gradient Boosting)...")
        gb_model = GradientBoostingClassifier(**config['gb_params'])
        gb_model.fit(X_train_smote, y_train_smote)
        metrics_gb = get_metrics(gb_model, X_test, y_test, f"Gradient Boosting ({region})")
        
        # Print Diff
        print("\nImprovement with Stacking:")
        print(f"ROC-AUC: {metrics_stack['ROC-AUC'] - metrics_gb['ROC-AUC']:+.4f}")
        print(f"F1-Score: {metrics_stack['F1-Score'] - metrics_gb['F1-Score']:+.4f}")

if __name__ == "__main__":
    main()
