# PowerShell script to generate fresh requirements.txt and conda.yaml

Write-Host "🔧 Setting up Python environment for Azure ML LoRA Fine-tuning..." -ForegroundColor Cyan

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

Write-Host "📦 Installing core packages..." -ForegroundColor Yellow

# Azure ML packages
pip install azure-ai-ml azure-identity azureml-mlflow

# ML Training packages
pip install transformers peft accelerate datasets torch trl

# Optional: Install bitsandbytes for GPU quantization
# Note: Skip on CPU-only machines or if not using quantization
# pip install bitsandbytes

# Evaluation packages
pip install rouge-score nltk evaluate

# Utilities
pip install mlflow pandas numpy pyyaml tqdm

Write-Host "💾 Generating requirements.txt..." -ForegroundColor Green
pip freeze > environment\requirements.txt

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
pip list | Select-String -Pattern "(azure|transformers|torch|peft|trl|datasets)"
