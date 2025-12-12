# Methodology and Approach: Cross-National CVE Exploitation Prediction Analysis

## Table of Contents

1. [Overview](#overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Engineering](#feature-engineering)
4. [Model Selection and Design](#model-selection-and-design)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Evaluation Strategy](#evaluation-strategy)
7. [Design Decisions and Justifications](#design-decisions-and-justifications)

---

## Overview

This document explains the methodology, design decisions, preprocessing steps, and parameter choices for the cross-national CVE exploitation prediction analysis. The analysis compares vulnerability exploitation patterns between Japan-related and Non-Japan-related CVEs using advanced machine learning techniques.

**Key Design Principle**: Separate models for each region to capture region-specific patterns, as preliminary analysis revealed significant differences in feature importance and exploitation rates between Japan and Non-Japan CVEs.

---

## Data Preprocessing

### 1. Data Filtering

**Decision**: Only process CVEs with `state = 'PUBLISHED'` from years 2021-2025.

**Justification**:

- Published CVEs represent confirmed vulnerabilities with complete information
- 2021-2025 timeframe provides recent, relevant data while maintaining sufficient sample size
- Excludes draft or rejected CVEs that may have incomplete or unreliable data

### 2. Regional Categorization

**Decision**: Identify Japan-related CVEs using keyword matching (JVN, JPCERT, etc.) in descriptions, reference URLs, and vendor/product names.

**Justification**:

- JVN (Japan Vulnerability Notes) and JPCERT are authoritative sources for Japan-specific vulnerability information
- Keyword-based approach captures both direct Japan references and indirect associations
- Provides clear binary classification (Japan vs Non-Japan) for separate model training

### 3. Target Variable Definition

**Decision**: `exploited = in_kev OR has_indicators`

**Justification**:

- **in_kev**: Known Exploited Vulnerabilities catalog provides ground truth for actual exploitation
- **has_indicators**: Captures CVEs with exploit indicators (keywords, high CVSS scores) that suggest exploitation likelihood
- Combined definition balances precision (KEV) with recall (indicators) to capture broader exploitation patterns
- Aligns with cybersecurity best practices for vulnerability prioritization

### 4. Sampling Strategy

**Decision**:

- Japan: Use all available CVEs (1,509 total)
- Non-Japan: Random sample of 6,000 CVEs (from ~150,000+ available)

**Justification**:

- **Japan**: Small dataset size (1,509) requires using all available data to maximize training samples
- **Non-Japan**: Large dataset allows random sampling to:
  - Reduce computational cost while maintaining statistical representativeness
  - Balance dataset sizes for fair comparison (6,000 vs 1,509 is reasonable ratio)
  - Enable faster iteration and experimentation
- **Random seed (42)**: Ensures reproducibility of sampling results

### 5. Train-Test Split

**Decision**: `test_size=0.2, random_state=42, stratify=y`

**Justification**:

- **20% test set**: Standard practice providing sufficient test data (~300 samples for Japan, ~1,200 for Non-Japan) while maximizing training data
- **Stratification**: Ensures both train and test sets maintain similar class distribution (exploited vs non-exploited ratio)
  - Critical for imbalanced datasets to prevent test set from having drastically different class proportions
- **Random seed (42)**: Ensures reproducible splits across runs

---

## Feature Engineering

### 1. Feature Selection: Phased Approach

**Design Decision**: Three-phase feature selection strategy

**Phase 1: Core Features (Always Included)**

- **Categorical**: `vendor`, `product`, `cvss_base_severity`
- **Numerical**: `cvss_score`, `has_cvss_score`, `vendor_popularity`

**Justification**:

- These features are fundamental to vulnerability assessment in cybersecurity
- High data availability and domain relevance
- Provide baseline predictive power

**Phase 2: CVSS Details**

- `cvss_attack_vector`, `cvss_attack_complexity`, `cvss_privileges_required`, `cvss_user_interaction`
- `cvss_confidentiality_impact`, `cvss_integrity_impact`, `cvss_availability_impact`

**Justification**:

- CVSS components provide granular exploitability and impact information
- Directly relate to exploitation likelihood (network-accessible vulnerabilities are more exploitable)
- Missing values handled as 'UNKNOWN' category to preserve information about data completeness

**Phase 3: Advanced Features**

- Top 20 most common CWE IDs (frequency-based selection)
- `has_exploit_reference` (binary indicator from reference tags)

**Justification**:

- **CWE frequency-based**: Top 20 captures most common vulnerability patterns while avoiding sparsity
- **Exploit references**: Direct signal of exploitation activity from metadata
- Domain-specific features complement CVSS metrics

### 2. Categorical Encoding

**Decision**: One-hot encoding with `drop_first=True`

**Justification**:

- **One-hot encoding**: Converts categorical variables to binary features suitable for ML algorithms
- **drop_first=True**: Removes one category per feature to avoid multicollinearity (dummy variable trap)
- Preserves all category information while maintaining numerical compatibility

**Missing Value Handling**: Fill with 'Unknown' or 'UNKNOWN' before encoding

**Justification**:

- Missing values are informative (indicates data completeness)
- Creating explicit 'Unknown' category preserves this information rather than losing it

### 3. Numerical Preprocessing

**Decision**: Standardization using `StandardScaler` (zero mean, unit variance)

**Justification**:

- **Standardization**: Essential for algorithms sensitive to feature scale (e.g., Logistic Regression, Neural Networks)
- Ensures all features contribute equally regardless of original scale
- CVSS scores (0-10) and vendor popularity (counts) have different scales that need normalization

**CVSS Score Imputation**:

- **Decision**: Median imputation (or year-specific median if available)
- **Justification**:
  - Median is robust to outliers compared to mean
  - Year-specific median accounts for temporal trends in CVSS scoring
  - Better than zero-imputation which would bias toward low-severity

**Vendor Popularity Calculation**:

- **Decision**: Frequency count per vendor, then standardized
- **Justification**:
  - Captures vendor market presence and attractiveness to attackers
  - Popular vendors are more likely targets for exploitation
  - Standardization makes it comparable across regions

### 4. Feature Categories

**Decision**: Automatic categorization system for interpretability

**Categories**:

- CVSS_Score, CVSS_Severity, CVSS_Exploitability, CVSS_Impact
- Vendor, Product (separated)
- CWE Patterns
- Reference, Exploit_Indicator
- Temporal, Other

**Justification**:

- Enables category-level analysis and comparison
- Helps understand which types of features matter most
- Separates Vendor and Product for granular analysis

---

## Model Selection and Design

### 1. Model Types: Seven Algorithms

**Decision**: Train 7 different model types for comprehensive comparison

**Models Selected**:

1. **Random Forest** - Ensemble of decision trees
2. **XGBoost** - Gradient boosting framework
3. **Logistic Regression** - Linear classifier
4. **Neural Network** - Multi-layer perceptron
5. **Gradient Boosting** - Sequential ensemble method
6. **Voting Classifier** - Soft voting ensemble
7. **Stacking Classifier** - Meta-learner ensemble

**Justification**:

- **Diversity**: Different algorithms capture different patterns (linear vs non-linear, ensemble vs single)
- **Robustness**: Multiple models enable ensemble methods (Voting, Stacking) which often outperform individual models
- **Interpretability**: Different models provide different insights (feature importance, coefficients, etc.)
- **Best Practice**: Comprehensive model comparison is standard in ML research

### 2. Separate Models for Each Region

**Decision**: Train separate models for Japan and Non-Japan datasets

**Justification**:

- Preliminary analysis showed different feature importance patterns between regions
- Different optimal algorithms for each region (validated by results: Voting best for both, but different second-best)
- Region-specific patterns require tailored approaches
- Enables fair comparison of regional differences

### 3. Class Imbalance Handling: SMOTE

**Decision**: Apply SMOTE (Synthetic Minority Oversampling Technique) to training data

**Justification**:

- **Problem**: Exploited CVEs are minority class (~20% for Japan, ~25% for Non-Japan)
- **SMOTE**: Creates synthetic samples of minority class rather than simple oversampling
  - Avoids overfitting from exact duplicates
  - Generates realistic synthetic examples in feature space
- **Random seed (42)**: Ensures reproducible SMOTE results
- **Applied only to training data**: Prevents data leakage into test set

**Alternative Considered**: Class weights

- **Why SMOTE chosen**: SMOTE creates actual samples, improving model's ability to learn minority class patterns
- Class weights only adjust loss function but don't add training examples

---

## Hyperparameter Optimization

### 1. Optimization Strategies by Model

#### **Grid Search** (Random Forest, Logistic Regression, Gradient Boosting)

**Decision**: Exhaustive search over parameter grid

**Justification**:

- **Small parameter spaces**: These models have relatively few hyperparameters
- **Deterministic**: Guarantees finding best combination within grid
- **Computational feasibility**: Grid sizes are manageable (e.g., 3×3×2 = 18 combinations for Random Forest)

**Parameter Grids**:

**Random Forest**:

```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
```

- **n_estimators**: 50-200 provides range from fast training to high accuracy
- **max_depth**: 10-20 limits overfitting; None allows full depth for comparison
- **min_samples_split**: 2 (default) vs 5 (more conservative) tests regularization

**Logistic Regression**:

```python
{
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
```

- **C**: Inverse regularization strength - wide range (0.01 to 10.0) tests from strong to weak regularization
- **penalty**: L1 (sparse) vs L2 (smooth) - different regularization behaviors
- **solver**: 'liblinear' supports both L1 and L2 penalties
- **max_iter=1000**: Prevents convergence issues with complex datasets

**Gradient Boosting**:

```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}
```

- **n_estimators**: Standard range for ensemble size
- **max_depth**: Shallow trees (3-7) prevent overfitting in boosting
- **learning_rate**: Low (0.01) for careful learning vs higher (0.1) for faster convergence

#### **Random Search** (XGBoost)

**Decision**: Random search with 100 iterations

**Justification**:

- **Large parameter space**: XGBoost has many hyperparameters
- **Efficiency**: Random search often finds good solutions faster than grid search
- **100 iterations**: Provides good coverage while remaining computationally feasible
- **Random seed (42)**: Ensures reproducibility

**Parameter Distribution**:

```python
{
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 7)
}
```

- **Wide ranges**: Captures diverse hyperparameter combinations
- **Continuous distributions**: For learning_rate, subsample, colsample_bytree - allows fine-grained search
- **Integer distributions**: For n_estimators, max_depth, min_child_weight - discrete parameters

**XGBoost Apple Silicon Optimization**:

```python
tree_method='hist', device='cpu', n_jobs=-1, predictor='cpu_predictor'
```

- **Justification**: XGBoost doesn't support Metal GPU, but these settings optimize CPU performance on Apple Silicon

#### **Bayesian Optimization** (Neural Networks)

**Decision**: Optuna with TPE (Tree-structured Parzen Estimator) sampler, 77 trials

**Justification**:

- **Complex hyperparameter space**: Neural networks have many interdependent hyperparameters
- **Bayesian optimization**: Efficiently explores parameter space by learning from previous trials
- **77 trials**: Balance between thoroughness and computational cost (reduced from 100 based on user feedback)
- **TPE sampler**: State-of-the-art Bayesian optimization algorithm

**Hyperparameter Ranges**:

```python
hidden_layer_sizes: [(50,), (100,), (200,), (50,50), (100,50), (200,100), (100,100)]
alpha: loguniform(1e-5, 1e-1)  # L2 regularization
learning_rate_init: loguniform(1e-5, 1e-1)
batch_size: [32, 64, 128, 256]
epochs: 50-200 (TensorFlow) or max_iter: 200-1000 (scikit-learn)
```

**Justification**:

- **hidden_layer_sizes**: Tests single and multi-layer architectures
- **Log-uniform distributions**: Appropriate for regularization and learning rate (orders of magnitude matter)
- **batch_size**: Standard power-of-2 values for efficient GPU/CPU processing
- **epochs range**: 50-200 balances training time with convergence

**Early Stopping**:

- **Decision**: `patience=10, restore_best_weights=True`
- **Justification**: Prevents overfitting and reduces training time

**TensorFlow/Keras vs scikit-learn**:

- **Decision**: Use TensorFlow/Keras with Metal GPU acceleration if available, fallback to scikit-learn
- **Justification**:
  - GPU acceleration significantly speeds up neural network training
  - Apple Silicon Metal backend provides transparent acceleration
  - Fallback ensures compatibility across systems

### 2. Cross-Validation Strategy

**Decision**: Stratified 5-fold cross-validation (`cv=5`)

**Justification**:

- **5 folds**: Standard practice balancing bias-variance tradeoff
  - More folds = less bias but higher variance and computational cost
  - Fewer folds = more bias but lower variance
- **Stratified**: Maintains class distribution in each fold
  - Critical for imbalanced datasets
  - Ensures each fold has representative proportion of exploited vs non-exploited CVEs
- **Used for**: Hyperparameter optimization and model selection

### 3. Scoring Metric

**Decision**: `scoring='roc_auc'` (ROC-AUC score)

**Justification**:

- **Robustness to Imbalance**: In this dataset, exploited CVEs are a significant minority. A model predicting "Not Exploited" for *everything* would achieve high accuracy (e.g., >98%) but would be useless for detection. ROC-AUC is insensitive to class skew and evaluates how well the model separates the classes, rather than just majority correctness.
- **Prioritization (Ranking) Focus**: For vulnerability management, the absolute probability is often less critical than the *relative ranking*. Security teams need to know which vulnerabilities to patch *first*. ROC-AUC effectively measures the probability that a randomly chosen exploited CVE is ranked higher than a randomly chosen safe one.
- **Threshold Independence**: We do not want to hardcode a specific decision threshold (e.g., 0.5) during model training. ROC-AUC evaluates performance across *all* possible thresholds, allowing organizations to later tune the sensitivity/specificity trade-off based on their remediation capacity (e.g., "fix top 100 risks").
- **Alternative considered**: Accuracy
  - **Why Rejected**: Accuracy is misleading for imbalanced data. It masks poor performance on the minority class (the exploited CVEs we care about).
- **ROC-AUC interpretation**:
  - 0.5 = random guessing
  - 1.0 = perfect classifier
  - >0.8 = good performance
  - >0.9 = excellent performance

---

## Evaluation Strategy

### 1. Holdout Test Set

**Decision**: Evaluate all models on separate 20% holdout test set

**Justification**:

- **Unseen data**: Provides unbiased estimate of generalization performance
- **No data leakage**: Test set never used in training or hyperparameter optimization
- **Fair comparison**: All models evaluated on same test set

### 2. Evaluation Metrics

**Metrics Calculated**:

- Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

**Justification**:

- **Comprehensive assessment**: Different metrics capture different aspects of performance
- **ROC-AUC**: Primary metric (used for optimization)
- **Precision/Recall**: Important for understanding false positive/negative tradeoffs
- **F1-Score**: Harmonic mean balances precision and recall
- **Confusion Matrix**: Provides detailed breakdown of prediction errors

### 3. Feature Importance Analysis

**Decision**: Analyze feature importance for best, second-best, and remaining models

**Justification**:

- **Interpretability**: Understands what features drive predictions
- **Model comparison**: Identifies consistent vs model-specific important features
- **Regional comparison**: Reveals differences between Japan and Non-Japan patterns

**Methods**:

- **Tree-based models**: Use `feature_importances_` attribute
- **Linear models**: Use absolute value of coefficients (normalized)
- **Neural Networks**: Permutation importance (requires test data)
- **Ensemble models**: Aggregate importances from base estimators

### 4. Bias Detection and Calibration Analysis

**Decision**: Analyze False Positive Rate (FPR), False Negative Rate (FNR), and calibration plots

**Justification**:

- **Fairness assessment**: FPR/FNR reveal systematic biases in predictions
- **Calibration**: Evaluates reliability of probability estimates
  - Well-calibrated models: predicted probabilities match actual outcomes
  - Poor calibration: probabilities don't reflect true likelihood
- **Cross-regional comparison**: Identifies fairness differences between models

**Calibration Method**:

- **Binning**: Divide predictions into 10 bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
- **Comparison**: Mean predicted probability vs actual fraction of positives per bin
- **Perfect calibration**: Points align with diagonal line (y=x)

---

## Design Decisions and Justifications

### 1. Why Separate Models for Each Region?

**Decision**: Train separate models rather than single global model

**Justification**:

- **Different patterns**: Preliminary analysis showed different feature importance rankings
- **Different optimal algorithms**: Results show different second-best models (Stacking for Japan, Gradient Boosting for Non-Japan)
- **Data quality differences**: Japan has higher data completeness (99.66% vendor, 100% product vs 88.70%, 96.01%)
- **Fair comparison**: Enables direct comparison of regional differences

### 2. Why Ensemble Methods?

**Decision**: Include Voting and Stacking classifiers

**Justification**:

- **Results validate**: Voting achieved best performance for both regions
- **Complementary strengths**: Different base models capture different patterns
- **Robustness**: Ensemble methods reduce variance and improve generalization
- **Best practice**: Standard approach in ML competitions and research

### 3. Why SMOTE Instead of Class Weights?

**Decision**: Use SMOTE for class imbalance

**Justification**:

- **Creates samples**: SMOTE generates synthetic training examples, improving model's ability to learn minority class
- **Class weights**: Only adjust loss function but don't add training data
- **Empirical**: SMOTE generally performs better for imbalanced classification

### 4. Why Random Seed 42?

**Decision**: Use `random_state=42` throughout

**Justification**:

- **Reproducibility**: Ensures consistent results across runs
- **Standard practice**: 42 is common convention in ML research
- **Debugging**: Makes it easier to reproduce and debug issues

### 5. Why 77 Trials for Neural Networks?

**Decision**: Reduced from 100 to 77 trials

**Justification**:

- **User feedback**: Original 100 trials took ~50 minutes, user requested reduction
- **Diminishing returns**: Bayesian optimization often finds good solutions in fewer trials
- **Computational efficiency**: 77 trials balances thoroughness with time constraints

### 6. Why No Feature Selection?

**Decision**: Include all available features without explicit filtering

**Justification**:

- **Tree-based models**: Handle high-dimensional data well (Random Forest, XGBoost)
- **Ensemble methods**: Can benefit from diverse feature sets
- **Interpretability**: Feature importance analysis reveals which features matter
- **Future work**: Could add feature selection as improvement

### 7. Why Separate Vendor and Product Categories?

**Decision**: Separate Vendor and Product in feature importance analysis (instead of combined "Vendor/Product")

**Justification**:

- **Granular analysis**: Enables understanding of vendor-level vs product-level patterns
- **Different insights**: Vendors and products may have different importance rankings
- **Better interpretability**: More specific categories provide clearer insights

---

## Reproducibility

### Random Seeds

- **All random operations**: `random_state=42`
- **Sampling**: `random_state=42`
- **Train-test split**: `random_state=42`
- **SMOTE**: `random_state=42`
- **Optuna**: `seed=42` in TPESampler
- **Neural Networks**: `tf.random.set_seed(42)` and `random_state=42`

### Version Tracking

- XGBoost version logged
- Optuna version logged
- TensorFlow/Keras versions logged
- Python version logged

### Parameter Documentation

- All hyperparameter grids/ranges documented in code
- Optimization strategies clearly specified
- Evaluation metrics explicitly defined

---

## Summary

This methodology employs a comprehensive, well-justified approach to cross-national CVE exploitation prediction:

1. **Systematic preprocessing** with clear justifications for each step
2. **Phased feature engineering** incorporating domain knowledge
3. **Multiple model types** with appropriate optimization strategies
4. **Rigorous evaluation** using multiple metrics and bias detection
5. **Reproducible design** with consistent random seeds and documentation

All design decisions are based on:

- **Domain knowledge**: Cybersecurity best practices
- **ML best practices**: Standard techniques for imbalanced classification
- **Empirical validation**: Results validate the approach (ensemble methods perform best)
- **Practical constraints**: Balancing thoroughness with computational efficiency
