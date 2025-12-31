#!/bin/bash
# Activate virtual environment for Hebrew Idiom Detection project
# Usage: source activate_env.sh

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing analysis packages..."
    pip install pandas numpy scipy scikit-learn matplotlib seaborn tabulate --quiet
    echo "✅ Virtual environment created and packages installed"
else
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    echo "Python: $(which python)"
    echo "Packages: pandas=$(python -c 'import pandas; print(pandas.__version__)'), scipy=$(python -c 'import scipy; print(scipy.__version__)')"
fi
