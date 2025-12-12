#!/usr/bin/env python3
"""
Single Model Hyperparameter Tuning Script
Replicates logic from Cross_National_CVE_Analysis.ipynb exactly.
"""
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
import warnings

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# Increase CSV field size limit
import csv
csv.field_size_limit(10000000)

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# Feature Engineering Logic (Copied from Notebook)
# -------------------------------------------------------------------------

def calculate_vendor_popularity(df):
    """Calculate vendor popularity based on frequency."""
    vendor_counts = df['vendor'].value_counts()
    vendor_popularity = df['vendor'].map(vendor_counts)
    return vendor_popularity.fillna(0)

def create_features(data, is_japanese=False, scaler_obj=None):
    """
    Create feature matrix from vulnerability data.
    Uses categorical and numerical features including Phase 1-3 candidate features.
    """
    data = data.reset_index(drop=True)
    feature_dfs = []
    
    # Categorical features (one-hot encoding)
    categorical_features = ['vendor', 'product', 'cvss_base_severity']
    
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            dummies.index = data.index
            feature_dfs.append(dummies)
    
    # Phase 1: year (categorical, one-hot encoded)
    if 'year' in data.columns:
        year_dummies = pd.get_dummies(data['year'], prefix='year', drop_first=True)
        year_dummies.index = data.index
        feature_dfs.append(year_dummies)
    
    # Phase 2: CVSS details
    cvss_categorical = [
        'cvss_attack_vector',
        'cvss_attack_complexity',
        'cvss_privileges_required',
        'cvss_user_interaction',
        'cvss_confidentiality_impact',
        'cvss_integrity_impact',
        'cvss_availability_impact'
    ]
    
    for col in cvss_categorical:
        if col in data.columns:
            data[col] = data[col].fillna('UNKNOWN')
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            dummies.index = data.index
            feature_dfs.append(dummies)
    
    # Phase 3: CWE IDs - extract top 20 most common
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
    
    # Reference tags - check for exploit-related tags
    if 'reference_tags' in data.columns:
        has_exploit_tag = data['reference_tags'].str.contains(
            'exploit', case=False, na=False, regex=False
        ).astype(int)
        exploit_tag_df = pd.DataFrame({'has_exploit_reference': has_exploit_tag}, index=data.index)
        feature_dfs.append(exploit_tag_df)
    
    # Numerical features
    vendor_popularity = calculate_vendor_popularity(data)
    
    # Handle CVSS scores - better imputation for missing values
    cvss_scores = pd.to_numeric(data.get('cvss_base_score', 0), errors='coerce')
    has_cvss_score = cvss_scores.notna().astype(int)
    
    # Impute missing CVSS scores with median
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
    
    # Prepare numerical features
    numerical_features = {
        'cvss_score': cvss_imputed,
        'has_cvss_score': has_cvss_score,
        'vendor_popularity': vendor_popularity
    }
    
    # Phase 1: reference_count, has_metrics
    if 'reference_count' in data.columns:
        numerical_features['reference_count'] = pd.to_numeric(
            data['reference_count'], errors='coerce'
        ).fillna(0)
    
    if 'has_metrics' in data.columns:
        numerical_features['has_metrics'] = pd.to_numeric(
            data['has_metrics'], errors='coerce'
        ).fillna(0).astype(int)
    
    numerical_data = pd.DataFrame(numerical_features, index=data.index)
    
    if scaler_obj is None:
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(numerical_data)
    else:
        scaler = scaler_obj
        numerical_scaled = scaler.transform(numerical_data)
    
    numerical_df = pd.DataFrame(numerical_scaled, columns=numerical_data.columns, index=data.index)
    feature_dfs.append(numerical_df)
    
    # Combine all features
    X = pd.concat(feature_dfs, axis=1) if feature_dfs else numerical_df
    
    # Clean column names
    X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
                 for col in X.columns]
    
    # Handle missing values
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    return X, scaler

def print_top_results(grid, n=5):
    """Print top n hyperparameter combinations from GridSearchCV with multiple metrics."""
    print(f"\nTop {n} Hyperparameter Combinations (Sorted by ROC-AUC):")
    print("-" * 140)
    print(f"{'Rank':<5} | {'ROC-AUC':<10} | {'F1':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'Parameters'}")
    print("-" * 140)
    
    results = pd.DataFrame(grid.cv_results_)
    # Sort by rank of the refit metric (roc_auc)
    results = results.sort_values('rank_test_roc_auc')
    
    for i, row in results.head(n).iterrows():
        params_str = str(row['params'])
        print(f"{row['rank_test_roc_auc']:<5} | "
              f"{row['mean_test_roc_auc']:.4f}     | "
              f"{row['mean_test_f1']:.4f}     | "
              f"{row['mean_test_accuracy']:.4f}     | "
              f"{row['mean_test_precision']:.4f}     | "
              f"{row['mean_test_recall']:.4f}     | "
              f"{params_str}")
    print("-" * 140)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1] if cm.shape[1] > 1 else 0, 
                                                   cm[1,0] if cm.shape[0] > 1 else 0, 
                                                   cm[1,1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'confusion_matrix': cm,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }
    
    print(f"\n{'='*60}")
    print(f"{model_name} Performance")
    print(f"{'='*60}")
    print(f"Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"Actual Negative   {tn:4d}     {fp:4d}")
    print(f"       Positive   {fn:4d}     {tp:4d}")
    print(f"\nMetrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
    return metrics

# -------------------------------------------------------------------------
# Main Logic
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Tune a single model for a specific region.')
    parser.add_argument('--model', type=str, required=True, choices=['rf', 'lr', 'gb', 'xgb', 'nn'],
                        help='Model to tune: rf, lr, gb, xgb, nn (Neural Network)')
    parser.add_argument('--region', type=str, required=True, choices=['Japan', 'Non-Japan'],
                        help='Region to analyze: Japan or Non-Japan')
    
    args = parser.parse_args()
    
    print(f"Starting tuning for Model: {args.model.upper()}, Region: {args.region}")
    
    # 1. Load Data
    project_root = Path(__file__).parent.parent
    jp_file = project_root / 'results' / 'jp_2021_2025_three_categories.csv'
    nonjp_file = project_root / 'results' / 'nonjp_2021_2025_three_categories_sample_6000.csv'
    cve_features_file = project_root / 'results' / 'cve_features_2021_2025.csv'
    
    print("Loading data...")
    if args.region == 'Japan':
        df = pd.read_csv(jp_file)
    else:
        df = pd.read_csv(nonjp_file)
        
    cve_features_df = pd.read_csv(cve_features_file, low_memory=False)
    merged = df.merge(cve_features_df, on='cve_id', how='left')
    
    # Target variable
    merged['exploited'] = ((merged['in_kev'] == True) | (merged['has_indicators'] == True)).astype(int)
    
    # 2. Feature Engineering
    print("Feature engineering...")
    X, _ = create_features(merged, is_japanese=(args.region == 'Japan'))
    y = merged['exploited'].values
    
    # 3. Train-Test Split (Standardized random_state=42)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. SMOTE (Standardized random_state=42)
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # 5. Tune Model
    results = {}
    
    if args.model == 'rf':
        print("Tuning Random Forest...")
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestClassifier(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=5, 
                          scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                          refit='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X_train_smote, y_train_smote)
        
        print_top_results(grid)
        
        print("\nBest Parameters:", grid.best_params_)
        evaluate_model(grid.best_estimator_, X_test, y_test, f"{args.region} Random Forest")
        
    elif args.model == 'lr':
        print("Tuning Logistic Regression...")
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
        grid = GridSearchCV(model, param_grid, cv=5, 
                          scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                          refit='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X_train_smote, y_train_smote)
        
        print_top_results(grid)
        
        print("\nBest Parameters:", grid.best_params_)
        evaluate_model(grid.best_estimator_, X_test, y_test, f"{args.region} Logistic Regression")
        
    elif args.model == 'gb':
        print("Tuning Gradient Boosting...")
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        model = GradientBoostingClassifier(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=5, 
                          scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                          refit='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X_train_smote, y_train_smote)
        
        print_top_results(grid)
        
        print("\nBest Parameters:", grid.best_params_)
        evaluate_model(grid.best_estimator_, X_test, y_test, f"{args.region} Gradient Boosting")

    elif args.model == 'xgb':
        print("Tuning XGBoost...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        grid = GridSearchCV(model, param_grid, cv=5, 
                          scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                          refit='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X_train_smote, y_train_smote)
        
        print_top_results(grid)
        
        print("\nBest Parameters:", grid.best_params_)
        evaluate_model(grid.best_estimator_, X_test, y_test, f"{args.region} XGBoost")
        
    elif args.model == 'nn':
        print("Tuning Neural Network...")
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
        # max_iter=1000 to ensure convergence
        model = MLPClassifier(random_state=42, max_iter=1000)
        grid = GridSearchCV(model, param_grid, cv=5, 
                          scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                          refit='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X_train_smote, y_train_smote)
        
        print_top_results(grid)
        
        print("\nBest Parameters:", grid.best_params_)
        evaluate_model(grid.best_estimator_, X_test, y_test, f"{args.region} Neural Network")

if __name__ == "__main__":
    main()
