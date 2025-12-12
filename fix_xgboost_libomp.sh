#!/bin/bash
# Fix XGBoost libomp path issue for Python 3.11
# Run this script with sudo: sudo ./fix_xgboost_libomp.sh

echo "Fixing XGBoost libomp path issue..."
echo ""

# Create directory if it doesn't exist
mkdir -p /usr/local/opt/libomp/lib

# Create symlink from Homebrew libomp to where XGBoost expects it
if [ -f "/opt/homebrew/opt/libomp/lib/libomp.dylib" ]; then
    ln -sf /opt/homebrew/opt/libomp/lib/libomp.dylib /usr/local/opt/libomp/lib/libomp.dylib
    echo "✅ Created symlink: /usr/local/opt/libomp/lib/libomp.dylib -> /opt/homebrew/opt/libomp/lib/libomp.dylib"
elif [ -f "/usr/local/opt/libomp/lib/libomp.dylib" ]; then
    echo "✅ libomp already exists at /usr/local/opt/libomp/lib/libomp.dylib"
else
    echo "❌ Error: libomp.dylib not found. Please install with: brew install libomp"
    exit 1
fi

# Verify the symlink
if [ -L "/usr/local/opt/libomp/lib/libomp.dylib" ] || [ -f "/usr/local/opt/libomp/lib/libomp.dylib" ]; then
    echo ""
    echo "✅ Symlink created successfully!"
    echo ""
    echo "Testing XGBoost import..."
    python3 -c "import xgboost as xgb; print('✅ XGBoost works for Python 3!')" 2>&1
    echo ""
    echo "🎉 Done! You can now restart your Jupyter kernel."
else
    echo "❌ Failed to create symlink"
    exit 1
fi







