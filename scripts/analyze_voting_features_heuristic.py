#!/usr/bin/env python3
"""
Voting Classifier Feature Analysis (Heuristic)
Calculates Composite Feature Importance by averaging normalized importance/weights 
from underlying base learners (RF, GB, XGB, LR).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

def get_normalized_importance(model, feature_names, model_type):
    if model_type in ['rf', 'gb', 'xgb']:
        importances = model.feature_importances_
    elif model_type == 'lr':
        importances = np.abs(model.coef_[0])
    
    # MinMax Scale to 0-1 for fair comparison across model types
    scaler = MinMaxScaler()
    importances_scaled = scaler.fit_transform(importances.reshape(-1, 1)).flatten()
    return pd.Series(importances_scaled, index=feature_names)

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
        print(f"Analyzing Region: {region} (Heuristic)")
        print(f"{'='*60}")
        
        # Prepare Data
        merged = config['df'].merge(cve_features_df, on='cve_id', how='left')
        merged['exploited'] = ((merged['in_kev'] == True) | (merged['has_indicators'] == True)).astype(int)
        
        X, _ = create_features(merged, is_japanese=(region == 'Japan'))
        y = merged['exploited'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Train Individual Models
        print("Training Random Forest...")
        rf = RandomForestClassifier(**config['rf_params'])
        rf.fit(X_train_smote, y_train_smote)
        
        print("Training Logistic Regression...")
        lr = LogisticRegression(**config['lr_params'])
        lr.fit(X_train_smote, y_train_smote)
        
        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(**config['gb_params'])
        gb.fit(X_train_smote, y_train_smote)
        
        print("Training XGBoost...")
        xgb = XGBClassifier(**config['xgb_params'])
        xgb.fit(X_train_smote, y_train_smote)
        
        # Ignore NN for feature importance as it's a black box without P.I.
        # We will average the available interpretable models.
        
        print("Calculating Composite Importance...")
        # Get Normalized Scores
        rf_imp = get_normalized_importance(rf, X.columns, 'rf')
        lr_imp = get_normalized_importance(lr, X.columns, 'lr')
        gb_imp = get_normalized_importance(gb, X.columns, 'gb')
        xgb_imp = get_normalized_importance(xgb, X.columns, 'xgb')
        
        # Average
        composite_importance = (rf_imp + lr_imp + gb_imp + xgb_imp) / 4.0
        
        sorted_feats = composite_importance.sort_values(ascending=False).head(20)
        
        print(f"\nTop 20 Features ({region}):")
        print("-" * 50)
        print(f"{'Feature':<40} {'Composite Score':<10}")
        print("-" * 50)
        
        for feat, score in sorted_feats.items():
            print(f"{feat:<40} {score:.4f}")

if __name__ == "__main__":
    main()
