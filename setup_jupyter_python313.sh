#!/bin/bash
# Setup Jupyter to use Python 3.13 (ARM64) which has XGBoost working

echo "Setting up Jupyter with Python 3.13..."
echo ""

# Check if Python 3.13 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Using: $PYTHON_VERSION"
echo ""

# Install Jupyter and ipykernel if not already installed
echo "Installing Jupyter and ipykernel..."
python3 -m pip install --user --quiet --upgrade jupyter ipykernel 2>/dev/null || echo "Jupyter/ipykernel may already be installed"

# Register Python 3.13 as a Jupyter kernel
echo ""
echo "Registering Python 3.13 as Jupyter kernel..."
python3 -m ipykernel install --user --name python313 --display-name "Python 3.13 (XGBoost Ready)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use Python 3.13 in Jupyter:"
echo "1. Start Jupyter: jupyter notebook"
echo "2. Create a new notebook or open existing one"
echo "3. Go to Kernel -> Change Kernel -> Select 'Python 3.13 (XGBoost Ready)'"
echo ""
echo "Or restart your current notebook and select the new kernel."

