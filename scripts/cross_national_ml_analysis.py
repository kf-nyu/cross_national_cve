#!/usr/bin/env python3
"""
Cross-National Vulnerability Prediction ML Analysis
Adapted from Cross_National_Vulnerability_Prediction.ipynb

Trains ML models separately for Japan and Non-Japan CVEs to identify
nation-specific patterns in vulnerability exploitation prediction.

Exploited = in_kev OR has_indicators

Features:
- Base: vendor, product, cvss_base_severity, cvss_score, vendor_popularity
- Phase 1: reference_count, has_metrics, year
- Phase 2: CVSS details (attack_vector, complexity, privileges, user_interaction, impacts)
- Phase 3: CWE IDs (top 20), reference_tags (exploit indicators)

NOTE: This version does NOT use TF-IDF features - only structured features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import re
from typing import List
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Try to import janome for Japanese text processing
try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False
    print("⚠️  Warning: janome not available. Japanese tokenization will use fallback method.")


# Domain-specific technical terms for cybersecurity
CYBERSECURITY_TERMS = {
    'exploit', 'vulnerability', 'injection', 'buffer', 'overflow', 'xss', 'csrf',
    'sql', 'command', 'execution', 'privilege', 'escalation', 'authentication',
    'authorization', 'bypass', 'traversal', 'path', 'remote', 'code', 'arbitrary',
    'cross-site', 'scripting', 'deserialization', 'rce', 'lfi', 'rfi', 'ssrf',
    'xxe', 'idor', 'cve', 'cwe', 'cvss', 'patch', 'mitigation',
    'disclosure', 'exposure', 'leak', 'unauthorized', 'access', 'denial', 'service',
    'dos', 'ddos', 'malware', 'ransomware', 'phishing', 'spoofing', 'tampering'
}

# Extended stopwords for English
EXTENDED_ENGLISH_STOPWORDS = {
    'allow', 'allows', 'allowed', 'provide', 'provides', 'provided', 'input',
    'output', 'data', 'information', 'system', 'systems', 'user', 'users',
    'file', 'files', 'can', 'could', 'may', 'might', 'would', 'should',
    'this', 'that', 'these', 'those', 'the', 'a', 'an', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'shall', 'must', 'need', 'needs', 'use', 'uses', 'used', 'using'
}


def extract_technical_phrases(text: str) -> List[str]:
    """Extract technical phrases from vulnerability descriptions."""
    if not text or not isinstance(text, str):
        return []
    text_lower = text.lower()
    patterns = [
        r'\b(sql|nosql|command|code|os|ldap|xml|xpath|template)\s+injection\b',
        r'\b(cross[- ]site|stored|reflected|dom)\s+(xss|scripting)\b',
        r'\b(buffer|stack|heap|integer)\s+overflow\b',
        r'\b(path|directory)\s+traversal\b',
        r'\b(remote|arbitrary)\s+code\s+execution\b',
        r'\b(privilege|permission)\s+escalation\b',
        r'\b(authentication|authorization)\s+bypass\b',
        r'\b(server[- ]side|client[- ]side)\s+request\s+forgery\b',
        r'\b(xml|xxe)\s+external\s+entity\b',
        r'\b(denial|distributed)\s+of\s+service\b',
        r'\b(deserialization|unserialize)\s+vulnerability\b',
    ]
    phrases = []
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            if isinstance(matches[0], tuple):
                phrases.extend([' '.join(m) for m in matches])
            else:
                phrases.extend(matches)
    return phrases


def improved_english_tokenizer(text: str) -> List[str]:
    """Improved tokenizer for English - extracts technical terms and phrases."""
    if not text or not isinstance(text, str):
        return []
    phrases = extract_technical_phrases(text)
    text_lower = text.lower()
    text_lower = re.sub(r'https?://\S+', '', text_lower)
    text_lower = re.sub(r'\S+@\S+', '', text_lower)
    text_lower = re.sub(r'cve-\d{4}-\d+', '', text_lower, flags=re.IGNORECASE)
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    filtered_words = []
    for word in words:
        if word in CYBERSECURITY_TERMS:
            filtered_words.append(word)
        elif word not in EXTENDED_ENGLISH_STOPWORDS and len(word) >= 4:
            filtered_words.append(word)
    return phrases + filtered_words


def improved_japanese_tokenizer(text: str) -> List[str]:
    """Improved tokenizer for Japanese - extracts meaningful words, filters particles."""
    if not text or not isinstance(text, str):
        return []
    
    if not JANOME_AVAILABLE:
        # Fallback: extract Japanese words and English technical terms
        japanese_words = re.findall(r'[ひらがなカタカナ漢字]{2,}', text)
        japanese_words = [w for w in japanese_words if w not in ['の', 'に', 'は', 'が', 'を', 'で', 'と']]
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        return japanese_words[:15] + [w.lower() for w in english_words if w.lower() in CYBERSECURITY_TERMS][:10]
    
    try:
        janome_tok = JanomeTokenizer()
        tokens = []
        
        # Extract technical terms
        technical_patterns = [
            r'脆弱性', r'セキュリティ', r'攻撃', r'実行', r'権限', r'認証',
            r'認可', r'インジェクション', r'クロスサイト', r'スクリプティング',
            r'バッファ', r'オーバーフロー', r'オーバーフロー', r'リモート', r'コード', r'任意',
            r'SQL', r'XSS', r'CSRF', r'RCE', r'DoS', r'DDoS',
            r'ハッカー', r'マルウェア', r'ランサムウェア', r'フィッシング',
            r'バイパス', r'エスカレーション'
        ]
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            tokens.extend(matches)
        
        # Tokenize with Janome
        for m in janome_tok.tokenize(text):
            part = m.part_of_speech.split(',')[0]
            base = m.base_form if m.base_form != '*' else m.surface
            surface = m.surface
            
            if part in ('名詞', '動詞', '形容詞', '副詞', '固有名詞'):
                if base not in ['の', 'に', 'は', 'が', 'を', 'で', 'と', 'から', 'まで', 
                               'です', 'ます', 'である', 'する', 'なる', 'ある', 'いる',
                               'こと', 'もの', 'ため', 'よう', 'とき', 'ところ']:
                    if len(base) >= 2:
                        tokens.append(base)
                    elif len(surface) >= 2 and surface != base:
                        tokens.append(surface)
        
        # Extract English technical terms
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        for word in english_words:
            word_lower = word.lower()
            if word_lower in CYBERSECURITY_TERMS or (word_lower not in EXTENDED_ENGLISH_STOPWORDS and len(word_lower) >= 4):
                tokens.append(word_lower)
        
        tokens = [t for t in tokens if t and len(t.strip()) > 0]
        tokens = list(set(tokens))
        
        if len(tokens) == 0:
            japanese_words = re.findall(r'[ひらがなカタカナ漢字]{2,}', text)
            japanese_words = [w for w in japanese_words if w not in ['の', 'に', 'は', 'が', 'を', 'で', 'と']]
            tokens = japanese_words[:20]
        
        return tokens
    except Exception as e:
        print(f"Warning in Japanese tokenizer: {e}")
        japanese_words = re.findall(r'[ひらがなカタカナ漢字]{2,}', text)
        japanese_words = [w for w in japanese_words if w not in ['の', 'に', 'は', 'が', 'を', 'で', 'と']]
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        return japanese_words[:15] + [w.lower() for w in english_words if len(w) >= 4][:10]


def calculate_vendor_popularity(df):
    """Calculate vendor popularity based on frequency."""
    vendor_counts = df['vendor'].value_counts()
    vendor_popularity = df['vendor'].map(vendor_counts)
    return vendor_popularity.fillna(0)


def create_features(data, is_japanese=False, tfidf_vectorizer=None, scaler_obj=None):
    """
    Create feature matrix from vulnerability data (NO TF-IDF).
    Uses categorical and numerical features including Phase 1-3 candidate features.
    
    Phase 1: reference_count, has_metrics, year
    Phase 2: CVSS details (attack_vector, complexity, privileges, user_interaction, impacts)
    Phase 3: CWE IDs (top 20), reference_tags (exploit indicators)
    """
    data = data.reset_index(drop=True)
    feature_dfs = []
    
    # ===== EXISTING CATEGORICAL FEATURES =====
    categorical_features = ['vendor', 'product', 'cvss_base_severity']
    
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].fillna('Unknown')
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            dummies.index = data.index
            feature_dfs.append(dummies)
    
    # ===== PHASE 1: QUICK WINS =====
    # year (categorical, one-hot encoded)
    if 'year' in data.columns:
        year_dummies = pd.get_dummies(data['year'], prefix='year', drop_first=True)
        year_dummies.index = data.index
        feature_dfs.append(year_dummies)
    
    # ===== PHASE 2: CVSS DETAILS =====
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
            # Fill missing with 'UNKNOWN'
            data[col] = data[col].fillna('UNKNOWN')
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            dummies.index = data.index
            feature_dfs.append(dummies)
    
    # ===== PHASE 3: ADVANCED FEATURES =====
    # CWE IDs - extract top 20 most common
    if 'cwe_ids' in data.columns:
        from collections import Counter
        all_cwes = []
        for cwe_str in data['cwe_ids'].dropna():
            if isinstance(cwe_str, str):
                cwes = [c.strip() for c in cwe_str.split('|') if c.strip()]
                all_cwes.extend(cwes)
        
        if all_cwes:
            top_cwes = [cwe for cwe, count in Counter(all_cwes).most_common(20)]
            cwe_features = {}
            for cwe in top_cwes:
                # Clean CWE ID for column name (e.g., "CWE-79" -> "CWE_79")
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
    
    # ===== NUMERICAL FEATURES =====
    # Calculate vendor popularity
    vendor_popularity = calculate_vendor_popularity(data)
    
    # Handle CVSS scores - better imputation for missing values
    cvss_scores = pd.to_numeric(data.get('cvss_base_score', 0), errors='coerce')
    has_cvss_score = cvss_scores.notna().astype(int)
    
    # Impute missing CVSS scores with median (better than 0)
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
    
    # Combine all features (categorical + numerical, NO TF-IDF)
    X = pd.concat(feature_dfs, axis=1) if feature_dfs else numerical_df
    
    # Clean column names
    X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
                 for col in X.columns]
    
    # Handle missing values
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    return X, None, scaler  # Return None for tfidf since we're not using it


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate confusion matrix
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
    print(f"Precision: {metrics['precision']:.4f} (TP/(TP+FP) = {tp}/({tp}+{fp}))")
    print(f"Recall:    {metrics['recall']:.4f} (TP/(TP+FN) = {tp}/({tp}+{fn}))")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def train_models(X_train, X_test, y_train, y_test, country_name="Dataset"):
    """Train and evaluate multiple models."""
    print(f"\n{'='*80}")
    print(f"Training Models for {country_name}")
    print(f"{'='*80}")
    
    results = {}
    
    # Apply SMOTE to handle class imbalance
    print(f"\nApplying SMOTE to training data...")
    print(f"  Before SMOTE: {X_train.shape[0]} samples, {y_train.mean():.2%} exploited")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {X_train_smote.shape[0]} samples, {y_train_smote.mean():.2%} exploited")
    
    # 1. Logistic Regression
    print("\n[1] Training Logistic Regression...")
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    }
    lr = LogisticRegression(random_state=42)
    grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_lr.fit(X_train_smote, y_train_smote)
    results['Logistic Regression'] = evaluate_model(grid_lr.best_estimator_, X_test, y_test, "Logistic Regression")
    
    # 2. Random Forest
    print("\n[2] Training Random Forest...")
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_rf.fit(X_train_smote, y_train_smote)
    results['Random Forest'] = evaluate_model(grid_rf.best_estimator_, X_test, y_test, "Random Forest")
    
    # 3. XGBoost
    print("\n[3] Training XGBoost...")
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_xgb.fit(X_train_smote, y_train_smote)
    results['XGBoost'] = evaluate_model(grid_xgb.best_estimator_, X_test, y_test, "XGBoost")
    
    return results


def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Load data
    print("=" * 80)
    print("CROSS-NATIONAL VULNERABILITY PREDICTION ML ANALYSIS")
    print("Exploited = in_kev OR has_indicators")
    print("=" * 80)
    
    print("\n[1] Loading data...")
    
    # Load jp and nonjp CSV files
    jp_file = project_root / 'results' / 'jp_2021_2025_three_categories.csv'
    nonjp_file = project_root / 'results' / 'nonjp_2021_2025_three_categories_sample_6000.csv'
    cve_features_file = project_root / 'results' / 'cve_features_2021_2025.csv'
    
    # Load classification files
    jp_df = pd.read_csv(jp_file)
    nonjp_df = pd.read_csv(nonjp_file)
    
    print(f"  - Japan CVEs: {len(jp_df):,}")
    print(f"  - Non-Japan CVEs: {len(nonjp_df):,}")
    
    # Load full CVE features
    print("\n[2] Loading CVE features...")
    cve_features_df = pd.read_csv(cve_features_file, low_memory=False)
    print(f"  - Total CVE features: {len(cve_features_df):,}")
    
    # Merge with features
    jp_merged = jp_df.merge(cve_features_df, on='cve_id', how='left')
    nonjp_merged = nonjp_df.merge(cve_features_df, on='cve_id', how='left')
    
    # Create target variable: exploited (1) if in_kev OR has_indicators, else 0
    jp_merged['exploited'] = ((jp_merged['in_kev'] == True) | (jp_merged['has_indicators'] == True)).astype(int)
    nonjp_merged['exploited'] = ((nonjp_merged['in_kev'] == True) | (nonjp_merged['has_indicators'] == True)).astype(int)
    
    print(f"\n[3] Dataset Statistics:")
    print(f"\nJapan Dataset:")
    print(f"  - Total: {len(jp_merged):,}")
    print(f"  - Exploited (in_kev OR has_indicators): {jp_merged['exploited'].sum():,} ({jp_merged['exploited'].mean():.2%})")
    print(f"  - Not Exploited: {(~jp_merged['exploited'].astype(bool)).sum():,} ({(~jp_merged['exploited'].astype(bool)).mean():.2%})")
    
    print(f"\nNon-Japan Dataset:")
    print(f"  - Total: {len(nonjp_merged):,}")
    print(f"  - Exploited (in_kev OR has_indicators): {nonjp_merged['exploited'].sum():,} ({nonjp_merged['exploited'].mean():.2%})")
    print(f"  - Not Exploited: {(~nonjp_merged['exploited'].astype(bool)).sum():,} ({(~nonjp_merged['exploited'].astype(bool)).mean():.2%})")
    
    # Feature engineering (NO TF-IDF - only categorical and numerical features)
    print(f"\n[4] Feature Engineering (NO TF-IDF)...")
    
    print("\n  Creating features for Japan dataset...")
    X_jp, tfidf_jp, scaler_jp = create_features(jp_merged, is_japanese=True)
    y_jp = jp_merged['exploited'].values
    
    print(f"  Japan feature matrix shape: {X_jp.shape}")
    
    print("\n  Creating features for Non-Japan dataset...")
    X_nonjp, tfidf_nonjp, scaler_nonjp = create_features(nonjp_merged, is_japanese=False)
    y_nonjp = nonjp_merged['exploited'].values
    
    print(f"  Non-Japan feature matrix shape: {X_nonjp.shape}")
    
    # Train-test split
    print(f"\n[5] Train-Test Split...")
    X_jp_train, X_jp_test, y_jp_train, y_jp_test = train_test_split(
        X_jp, y_jp, test_size=0.2, random_state=42, stratify=y_jp
    )
    
    X_nonjp_train, X_nonjp_test, y_nonjp_train, y_nonjp_test = train_test_split(
        X_nonjp, y_nonjp, test_size=0.2, random_state=42, stratify=y_nonjp
    )
    
    print(f"  Japan - Train: {X_jp_train.shape[0]}, Test: {X_jp_test.shape[0]}")
    print(f"  Non-Japan - Train: {X_nonjp_train.shape[0]}, Test: {X_nonjp_test.shape[0]}")
    
    # Train models
    print(f"\n[6] Model Training...")
    
    jp_results = train_models(X_jp_train, X_jp_test, y_jp_train, y_jp_test, "Japan")
    nonjp_results = train_models(X_nonjp_train, X_nonjp_test, y_nonjp_train, y_nonjp_test, "Non-Japan")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Model':<25} {'Metric':<15} {'Japan':<15} {'Non-Japan':<15}")
    print("-" * 80)
    
    for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
        print(f"\n{model_name}:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            jp_val = jp_results[model_name].get(metric, 0)
            nonjp_val = nonjp_results[model_name].get(metric, 0)
            if jp_val is not None and nonjp_val is not None:
                print(f"  {metric.capitalize():<23} {jp_val:<15.4f} {nonjp_val:<15.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")


if __name__ == '__main__':
    main()

