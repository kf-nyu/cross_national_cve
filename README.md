# Cross-National CVE Analysis Project

## Directory Structure

```text
cross_national_cve/
├── data/              # CVE data files
│   └── cvelistV5-main/
│       └── cves/      # CVE JSON files organized by year
├── scripts/           # Analysis scripts
├── results/           # Analysis results and outputs
└── venv/              # Python virtual environment
```

## Setup & Data Acquisition

To reproduce the analysis, you need to download the full CVE list (Version 5).

### 1. Download the Data (Optional for Reproduction)

**Note:** The processed KEV dataset `data/unified_vulnerability_dataset_real_kev.csv` is **included in this repository**, so you can run the analysis scripts immediately.

If you wish to regenerate the dataset from scratch using the raw CVE JSONs, follow these steps:

1. Visit the [CVE.org Downloads page](https://www.cve.org/downloads) or go directly to the [CVE List V5 GitHub Repository](https://github.com/CVEProject/cvelistV5).
2. Click on the green **code** button and select **Download ZIP**, or download the source code zip from a release.
    - *Direct download link:* [cvelistV5-main.zip](https://github.com/CVEProject/cvelistV5/archive/refs/heads/main.zip)

### 2. Unzip and Organize

1. Create the `data` directory if it doesn't exist:

    ```bash
    mkdir -p data
    ```

2. Move the downloaded zip file into the `data` directory.
3. Unzip the file:

    ```bash
    cd data
    unzip cvelistV5-main.zip
    ```

    *Note: If your unzipped folder has a different name (e.g., `cvelistV5-master`), please rename it to `cvelistV5-main` or update the path in your scripts.*

### 3. Verify Structure

Ensure your directory looks like this:

```text
data/
└── cvelistV5-main/
    ├── cves/
    │   ├── 1999/
    │   ├── 2000/
    │   └── ...
    └── ...
```

## Data Structure

The CVE data is organized by year:

- `cves/1999/`, `cves/2000/`, ... `cves/2025/`
- Each year contains subdirectories like `0xxx/`, `1xxx/`, `10xxx/`, etc.
- Each CVE is stored as a JSON file: `CVE-YYYY-NNNN.json`

For a detailed definition of all 49 features extracted from these files, see [features_list.md](features_list.md).

## Usage

All analysis scripts should now reference:

```python
from pathlib import Path

# Assuming you are in the project root
project_root = Path.cwd()
cve_data_dir = project_root / 'data' / 'cvelistV5-main' / 'cves'
```

## Troubleshooting: XGBoost on macOS

If you are running this project on a Mac (especially M1/M2/M3 chips) and encounter an error like:

`XGBoostLibraryNotFound: libomp.dylib not found`

This is a common issue because XGBoost requires the OpenMP library, which is not installed by default on macOS.

**Solution:**

1. We have included a diagnostic script. Run this first to check your environment:

    ```bash
    python3 check_xgboost_setup.py
    ```

2. If it reports `libomp` is missing, you can fix it automatically by running:

    ```bash
    chmod +x fix_xgboost_libomp.sh
    sudo ./fix_xgboost_libomp.sh
    ```

    *(Or manually install via `brew install libomp`)*

3. After fixing, run `python3 check_xgboost_setup.py` again to confirm everything is working.

## Key Analysis Scripts

| Script | Purpose |
| :--- | :--- |
| `scripts/extract_cve_to_csv.py` | **Data Generation**: Extracts features from raw CVE JSON files (in `data/`) and creates the master CSV dataset. Crucial for reproduction if you don't use the pre-packaged CSV. |
| `scripts/tune_model.py` | **Hyperparameter Tuning**: Runs Grid Search CV on Random Forest, Gradient Boosting, XGBoost, etc., to find optimal parameters for Japan vs Non-Japan datasets. |
| `scripts/train_voting.py` | **Voting Ensemble**: Trains the finalized "Soft Voting" ensemble using the optimized 5-model configuration (RF, LR, GB, XGB, NN). |
| `scripts/train_stacking.py` | **Stacking Ensemble**: Trains the Stacking Classifier with Logistic Regression meta-learner. |
| `scripts/final_ranking.py` | **Reporting**: Generates the final ranked tables comparing all models (base + ensembles) by ROC-AUC and other metrics. |
| `scripts/analyze_voting_features_heuristic.py` | **Feature Analysis**: Calculates feature importance for the Voting Ensemble using a composite score of underlying model weights. |
