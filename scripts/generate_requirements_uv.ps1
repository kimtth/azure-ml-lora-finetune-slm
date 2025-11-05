# PowerShell script to generate fresh requirements.txt and conda.yaml using uv (faster alternative to pip)
# uv is a fast Python package installer written in Rust by Astral: https://github.com/astral-sh/uv
#
# - Current approach (uv pip install): Azure ML expects requirements.txt (not pyproject.toml)
# uv pip install transformers peft torch
# uv pip freeze > requirements.txt
# - Modern approach (uv add)
# uv add transformers peft torch

# Modern approach (uv add)
uv add transformers peft torch
# Creates/updates pyproject.toml and uv.lock automatically

# Modern approach (uv add)
uv add transformers peft torch
# Creates/updates pyproject.toml and uv.lock automatically

Write-Host "🚀 Setting up Python environment with uv (fast package installer)..." -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️  uv is not installed. Installing uv..." -ForegroundColor Yellow
    
    # Install uv using PowerShell command
    # For Windows, use the official installer
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    
    Write-Host "✅ uv installed successfully!" -ForegroundColor Green
    Write-Host ""
}

# Create virtual environment using uv (much faster than python -m venv)
Write-Host "🔨 Creating virtual environment with uv..." -ForegroundColor Yellow
uv venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

Write-Host "📦 Installing core packages with uv (10-100x faster than pip)..." -ForegroundColor Yellow

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

Write-Host "💾 Generating requirements.txt..." -ForegroundColor Green
uv pip freeze > environment\requirements.txt

Write-Host "📝 Generating conda.yaml..." -ForegroundColor Green
@"
name: azureml-lora-finetune
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - -r requirements.txt
"@ | Out-File -FilePath environment\conda.yaml -Encoding UTF8

Write-Host "✅ Done! Files generated:" -ForegroundColor Green
Write-Host "   - environment\requirements.txt"
Write-Host "   - environment\conda.yaml"
Write-Host ""
Write-Host "📊 Installed versions:" -ForegroundColor Cyan
uv pip list | Select-String -Pattern "(azure|transformers|torch|peft|trl|datasets)"

Write-Host ""
Write-Host "⚡ Performance Note:" -ForegroundColor Magenta
Write-Host "   uv is 10-100x faster than pip for package installation"
Write-Host "   Learn more: https://github.com/astral-sh/uv"
