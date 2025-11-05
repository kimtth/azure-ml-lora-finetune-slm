"""
Setup script to verify environment and prepare for training.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.9+."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    required_packages = [
        "azure-ai-ml",
        "azure-identity",
        "transformers",
        "peft",
        "torch",
        "datasets",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            missing.append(package)
    
    if missing:
        print("\nTo install missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_data_files():
    """Check if data files exist."""
    print("\nChecking data files...")
    data_dir = Path("data")
    
    required_files = ["train.jsonl", "validation.jsonl"]
    all_exist = True
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            # Count lines
            with open(file_path) as f:
                num_lines = sum(1 for _ in f)
            print(f"✓ {file_name} ({num_lines} examples)")
        else:
            print(f"❌ {file_name} not found")
            all_exist = False
    
    return all_exist


def check_env_file():
    """Check if .env file exists."""
    print("\nChecking environment configuration...")
    env_file = Path(".env")
    
    if env_file.exists():
        print("✓ .env file found")
        return True
    else:
        print("⚠ .env file not found")
        print("  Copy .env.template to .env and fill in your Azure details")
        return False


def validate_data_format():
    """Validate data format."""
    print("\nValidating data format...")
    
    try:
        from src.utils import validate_jsonl_format
        
        train_valid = validate_jsonl_format("data/train.jsonl")
        val_valid = validate_jsonl_format("data/validation.jsonl")
        
        return train_valid and val_valid
    except Exception as e:
        print(f"❌ Error validating data: {e}")
        return False


def print_next_steps():
    """Print next steps."""
    print("\n" + "=" * 80)
    print("Setup Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Configure .env with your Azure ML workspace details")
    print("  2. Ensure GPU compute cluster exists in your workspace")
    print("  3. Submit training job:")
    print("     cd jobs")
    print("     python submit_training_job.py")
    print("\nFor more details, see QUICKSTART.md")
    print("=" * 80)


def main():
    """Main setup function."""
    print("=" * 80)
    print("Azure ML LoRA Fine-tuning - Setup Verification")
    print("=" * 80)
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data files", check_data_files),
        ("Environment config", check_env_file),
        ("Data format", validate_data_format),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Setup Summary")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{name:.<40} {status}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n✓ All checks passed!")
        print_next_steps()
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("Run this script again after fixing the issues.")


if __name__ == "__main__":
    main()
