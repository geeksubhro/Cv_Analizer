#!/bin/bash
echo "🔧 Setting up virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "📦 Installing requirements..."
pip install -r requirements.txt

echo "✅ Setup complete. Run with: python main.py"
