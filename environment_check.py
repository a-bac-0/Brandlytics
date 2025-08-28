# environment_check.py
"""
Quick environment check for YOLO brand detection project
Run this first to see what we're working with
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    print(f"🐍 Python Version: {sys.version}")
    
def check_gpu():
    try:
        import torch
        print(f"🚀 PyTorch Version: {torch.__version__}")
        print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU Device: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   CPU training only")
    except ImportError:
        print("❌ PyTorch not installed")

def check_required_packages():
    packages = {
        'ultralytics': 'YOLOv8 library',
        'opencv-python': 'OpenCV for image processing', 
        'pillow': 'Image handling',
        'matplotlib': 'Visualization',
        'pandas': 'Data handling',
        'numpy': 'Numerical computing'
    }
    
    print("\n📦 Package Status:")
    installed = []
    missing = []
    
    for package, description in packages.items():
        if importlib.util.find_spec(package.replace('-', '_')) is not None:
            try:
                module = importlib.import_module(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package}: {version} - {description}")
                installed.append(package)
            except:
                print(f"⚠️  {package}: installed but version unknown - {description}")
                installed.append(package)
        else:
            print(f"❌ {package}: not installed - {description}")
            missing.append(package)
    
    return installed, missing

def create_install_script(missing_packages):
    if missing_packages:
        print(f"\n📥 Installation needed for: {', '.join(missing_packages)}")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        print(f"Run: {install_cmd}")
        
        # Create install script
        with open('install_requirements.sh', 'w') as f:
            f.write(f"#!/bin/bash\n")
            f.write(f"echo 'Installing required packages...'\n")
            f.write(f"{install_cmd}\n")
            f.write(f"echo 'Installation complete!'\n")
        print("Created install_requirements.sh script")
    else:
        print("✅ All packages installed!")

def check_directory_structure():
    import os
    print("\n📁 Current Directory Structure:")
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and common irrelevant ones
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show only first 5 files per directory
            if not file.startswith('.'):
                print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

if __name__ == "__main__":
    print("🔍 Environment Check for YOLO Brand Detection Project")
    print("=" * 60)
    
    check_python_version()
    check_gpu()
    installed, missing = check_required_packages()
    create_install_script(missing)
    check_directory_structure()
    
    print("\n" + "=" * 60)
    print("✅ Environment check complete!")
    print("\nNext steps:")
    if missing:
        print("1. Install missing packages using the command above")
        print("2. Re-run this script to verify installation")
        print("3. Share the results with your team")
    else:
        print("1. Your environment is ready!")
        print("2. Share your directory structure")  
        print("3. We can proceed with dataset preparation")