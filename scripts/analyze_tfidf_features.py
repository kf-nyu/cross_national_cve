#!/usr/bin/env python3
"""
Analyze TF-IDF features to identify what words/phrases they represent
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

# Increase CSV field size limit
import csv
csv.field_size_limit(10000000)

warnings.filterwarnings('ignore')

# Import feature engineering functions
sys.path.insert(0, str(Path(__file__).parent))
from cross_national_ml_analysis import (
    create_features, improved_english_tokenizer, improved_japanese_tokenizer
)

def get_tfidf_vocabulary(tfidf_vectorizer):
    """Get vocabulary mapping from TF-IDF vectorizer."""
    if hasattr(tfidf_vectorizer, 'vocabulary_'):
        # Reverse the vocabulary dictionary to get index -> word mapping
        vocab = {v: k for k, v in tfidf_vectorizer.vocabulary_.items()}
        return vocab
    elif hasattr(tfidf_vectorizer, 'get_feature_names_out'):
        # Newer sklearn versions
        feature_names = tfidf_vectorizer.get_feature_names_out()
        return {i: name for i, name in enumerate(feature_names)}
    elif hasattr(tfidf_vectorizer, 'get_feature_names'):
        # Older sklearn versions
        feature_names = tfidf_vectorizer.get_feature_names()
        return {i: name for i, name in enumerate(feature_names)}
    return {}

def main():
    project_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("TF-IDF FEATURE ANALYSIS")
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
    
    jp_merged['exploited'] = ((jp_merged['in_kev'] == True) | (jp_merged['has_indicators'] == True)).astype(int)
    nonjp_merged['exploited'] = ((nonjp_merged['in_kev'] == True) | (nonjp_merged['has_indicators'] == True)).astype(int)
    
    # Create features to get TF-IDF vectorizers
    print("\n[2] Creating features to extract TF-IDF vocabulary...")
    X_jp, tfidf_jp, scaler_jp = create_features(jp_merged, is_japanese=True)
    X_nonjp, tfidf_nonjp, scaler_nonjp = create_features(nonjp_merged, is_japanese=False)
    
    # Get vocabularies
    print("\n[3] Extracting TF-IDF vocabularies...")
    jp_vocab = get_tfidf_vocabulary(tfidf_jp)
    nonjp_vocab = get_tfidf_vocabulary(tfidf_nonjp)
    
    # Load feature importance
    print("\n[4] Loading feature importance data...")
    jp_importance_df = pd.read_csv(project_root / 'results' / 'jp_feature_importance.csv')
    nonjp_importance_df = pd.read_csv(project_root / 'results' / 'nonjp_feature_importance.csv')
    
    # Extract TF-IDF features from top 20
    print("\n" + "=" * 80)
    print("TOP 20 FEATURES - JAPAN (with TF-IDF meanings)")
    print("=" * 80)
    
    jp_top20 = jp_importance_df.head(20)
    for idx, row in jp_top20.iterrows():
        feature = row['feature']
        importance = row['importance']
        
        if feature.startswith('tfidf_'):
            # Extract index
            tfidf_idx = int(feature.split('_')[1])
            word = jp_vocab.get(tfidf_idx, 'UNKNOWN')
            print(f"{idx+1:2d}. {feature:<15} {importance:.6f} → '{word}'")
        else:
            print(f"{idx+1:2d}. {feature:<15} {importance:.6f}")
    
    print("\n" + "=" * 80)
    print("TOP 20 FEATURES - NON-JAPAN (with TF-IDF meanings)")
    print("=" * 80)
    
    nonjp_top20 = nonjp_importance_df.head(20)
    for idx, row in nonjp_top20.iterrows():
        feature = row['feature']
        importance = row['importance']
        
        if feature.startswith('tfidf_'):
            # Extract index
            tfidf_idx = int(feature.split('_')[1])
            word = nonjp_vocab.get(tfidf_idx, 'UNKNOWN')
            print(f"{idx+1:2d}. {feature:<15} {importance:.6f} → '{word}'")
        else:
            print(f"{idx+1:2d}. {feature:<15} {importance:.6f}")
    
    # Create detailed TF-IDF mapping
    print("\n[5] Creating TF-IDF feature mapping files...")
    
    # Japan TF-IDF mapping
    jp_tfidf_features = jp_top20[jp_top20['feature'].str.startswith('tfidf_')]
    jp_tfidf_mapping = []
    for _, row in jp_tfidf_features.iterrows():
        feature = row['feature']
        tfidf_idx = int(feature.split('_')[1])
        word = jp_vocab.get(tfidf_idx, 'UNKNOWN')
        jp_tfidf_mapping.append({
            'feature': feature,
            'index': tfidf_idx,
            'word_phrase': word,
            'importance': row['importance']
        })
    
    jp_tfidf_df = pd.DataFrame(jp_tfidf_mapping)
    jp_tfidf_df.to_csv(project_root / 'results' / 'jp_tfidf_top20_mapping.csv', index=False)
    
    # Non-Japan TF-IDF mapping
    nonjp_tfidf_features = nonjp_top20[nonjp_top20['feature'].str.startswith('tfidf_')]
    nonjp_tfidf_mapping = []
    for _, row in nonjp_tfidf_features.iterrows():
        feature = row['feature']
        tfidf_idx = int(feature.split('_')[1])
        word = nonjp_vocab.get(tfidf_idx, 'UNKNOWN')
        nonjp_tfidf_mapping.append({
            'feature': feature,
            'index': tfidf_idx,
            'word_phrase': word,
            'importance': row['importance']
        })
    
    nonjp_tfidf_df = pd.DataFrame(nonjp_tfidf_mapping)
    nonjp_tfidf_df.to_csv(project_root / 'results' / 'nonjp_tfidf_top20_mapping.csv', index=False)
    
    print(f"\n✅ TF-IDF mappings saved:")
    print(f"   - {project_root / 'results' / 'jp_tfidf_top20_mapping.csv'}")
    print(f"   - {project_root / 'results' / 'nonjp_tfidf_top20_mapping.csv'}")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

