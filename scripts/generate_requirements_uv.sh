#!/bin/bash
# Bash script to generate fresh requirements.txt and conda.yaml using uv (faster alternative to pip)
# uv is a fast Python package installer written in Rust by Astral: https://github.com/astral-sh/uv

echo "🚀 Setting up Python environment with uv (fast package installer)..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "⚠️  uv is not installed. Installing uv..."
    
    # Install uv using the official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    echo "✅ uv installed successfully!"
    echo ""
fi

# Create virtual environment using uv (much faster than python -m venv)
echo "🔨 Creating virtual environment with uv..."
uv venv .venv

# Activate virtual environment
source .venv/bin/activate

echo "📦 Installing core packages with uv (10-100x faster than pip)..."

# Azure ML packages
uv pip install azure-ai-ml azure-identity azureml-mlflow

# ML Training packages
uv pip install transformers peft accelerate datasets torch trl

# Optional: Install bitsandbytes for GPU quantization
# Note: Skip on CPU-only machines or if not using quantization
# uv pip install bitsandbytes

# Evaluation packages
uv pip install rouge-score nltk evaluate

# Utilities
uv pip install mlflow pandas numpy pyyaml tqdm

echo "💾 Generating requirements.txt..."
uv pip freeze > environment/requirements.txt

echo "📝 Generating conda.yaml..."
cat > environment/conda.yaml << EOF
name: azureml-lora-finetune
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - -r requirements.txt
EOF

echo "✅ Done! Files generated:"
echo "   - environment/requirements.txt"
echo "   - environment/conda.yaml"
echo ""
echo "📊 Installed versions:"
uv pip list | grep -E "(azure|transformers|torch|peft|trl|datasets)"

echo ""
echo "⚡ Performance Note:"
echo "   uv is 10-100x faster than pip for package installation"
echo "   Learn more: https://github.com/astral-sh/uv"
