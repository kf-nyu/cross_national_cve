# XGBoost Setup Instructions

## Problem
Jupyter is using Python 3.11 (x86_64/Intel), but XGBoost needs libomp which is installed for ARM64 (Apple Silicon).

## Solution: Use Python 3.13 (Recommended)

Python 3.13 (ARM64) already has XGBoost working. Configure Jupyter to use it:

### Option 1: Install Jupyter with Python 3.13

```bash
# Install Jupyter with Python 3.13
python3 -m pip install jupyter ipykernel

# Register Python 3.13 as a Jupyter kernel
python3 -m ipykernel install --user --name python313 --display-name "Python 3.13"

# Start Jupyter
jupyter notebook
# Then select "Python 3.13" as the kernel
```

### Option 2: Install libomp for x86_64 (if you must use Python 3.11)

If you need to use Python 3.11, install libomp for x86_64:

```bash
# Install Homebrew for Intel (if not already installed)
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install libomp for x86_64
arch -x86_64 brew install libomp

# Create symlink
sudo mkdir -p /usr/local/opt/libomp/lib
sudo ln -sf /usr/local/opt/libomp/lib/libomp.dylib /usr/local/opt/libomp/lib/libomp.dylib
```

### Option 3: Quick Fix Script

Run the fix script (requires sudo):

```bash
sudo ./fix_xgboost_libomp.sh
```

**Note**: This will only work if Python 3.11 and libomp are the same architecture.

## Verify Setup

After setup, verify XGBoost works:

```bash
# For Python 3.13
python3 -c "import xgboost as xgb; print('✅ XGBoost works!')"

# For Python 3.11
python3.11 -c "import xgboost as xgb; print('✅ XGBoost works!')"
```

## Recommended Approach

**Use Python 3.13** - it's already working and is the native architecture for Apple Silicon Macs.







