#!/usr/bin/env python3
"""
Check XGBoost setup and provide instructions if needed.
Run this before starting Jupyter to ensure XGBoost will work.
"""

import sys
import os
from pathlib import Path

print("=" * 80)
print("XGBoost Setup Checker")
print("=" * 80)

# Check Python version
print(f"\nPython version: {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")

# Check if libomp exists
libomp_paths = [
    '/opt/homebrew/opt/libomp/lib/libomp.dylib',
    '/usr/local/opt/libomp/lib/libomp.dylib'
]

libomp_found = False
for path in libomp_paths:
    if os.path.exists(path):
        print(f"\n✅ Found libomp at: {path}")
        libomp_found = True
        break

if not libomp_found:
    print("\n❌ libomp.dylib not found!")
    print("   Install with: brew install libomp")
    sys.exit(1)

# Try to import XGBoost
print("\n" + "-" * 80)
print("Testing XGBoost import...")
print("-" * 80)

try:
    import xgboost as xgb
    print(f"✅ XGBoost imported successfully!")
    print(f"   Version: {xgb.__version__}")
    print(f"   Location: {xgb.__file__}")
    
    # Test creating a simple model
    print("\nTesting XGBoost functionality...")
    import numpy as np
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    print(f"✅ XGBoost model created and tested successfully!")
    print(f"   Predictions shape: {predictions.shape}")
    
    print("\n" + "=" * 80)
    print("✅ All checks passed! XGBoost is ready to use.")
    print("=" * 80)
    print("\nYou can now start Jupyter and run the notebook.")
    
except ImportError as e:
    print(f"\n❌ Failed to import XGBoost!")
    print(f"   Error: {e}")
    print("\n🔧 Solutions:")
    print("   1. Install libomp: brew install libomp")
    print("   2. Create symlink:")
    print("      sudo mkdir -p /usr/local/opt/libomp/lib")
    print("      sudo ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /usr/local/opt/libomp/lib/libomp.dylib")
    print("   3. Reinstall XGBoost: pip install --upgrade --force-reinstall xgboost")
    print("   4. Restart terminal/Jupyter kernel")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ XGBoost import succeeded but functionality test failed!")
    print(f"   Error: {e}")
    print("\n🔧 Try reinstalling XGBoost:")
    print("   pip install --upgrade --force-reinstall xgboost")
    sys.exit(1)







