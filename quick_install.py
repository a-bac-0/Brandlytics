# quick_install.py
"""
Quick installation script - installs packages one by one with progress
"""

import subprocess
import sys
import time

def install_package(package_name, description=""):
    """Install a single package with real-time output"""
    print(f"\n🔧 Installing {package_name}...")
    if description:
        print(f"   Purpose: {description}")
    
    try:
        # Use pip with --progress-bar and --verbose for better feedback
        cmd = [sys.executable, "-m", "pip", "install", package_name, "--progress-bar", "on"]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        # Print output in real time
        for line in process.stdout:
            print(f"   {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print(f"✅ {package_name} installed successfully!")
            return True
        else:
            print(f"❌ Failed to install {package_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing {package_name}: {e}")
        return False

def test_import(package_name, import_name=None):
    """Test if a package can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} working correctly")
        return True
    except ImportError:
        print(f"❌ {package_name} import failed")
        return False

def main():
    print("⚡ Quick Installation for YOLO Brand Detection")
    print("=" * 55)
    
    # List of packages in order of importance
    packages = [
        ("torch", "PyTorch for deep learning - this may take a few minutes"),
        ("torchvision", "PyTorch vision utilities"),
        ("ultralytics", "YOLOv8 library - the main model we'll use"),
        ("opencv-python", "Computer vision library"),
        ("pillow", "Image processing"),
        ("albumentations", "Data augmentation"),
        ("tqdm", "Progress bars")
    ]
    
    print("📦 Installing essential packages...")
    print("Note: PyTorch might take 2-3 minutes, others are quick")
    
    successful = []
    failed = []
    
    for package, description in packages:
        print(f"\n{'='*20} Package {len(successful)+len(failed)+1}/{len(packages)} {'='*20}")
        
        if install_package(package, description):
            successful.append(package)
        else:
            failed.append(package)
            
            # Ask if user wants to continue
            response = input(f"\n⚠️ {package} failed. Continue with next package? (y/n): ").lower()
            if response != 'y':
                break
    
    # Test imports
    print(f"\n🧪 Testing installations...")
    test_mapping = {
        "torch": "torch",
        "torchvision": "torchvision", 
        "ultralytics": "ultralytics",
        "opencv-python": "cv2",
        "pillow": "PIL",
        "albumentations": "albumentations",
        "tqdm": "tqdm"
    }
    
    working = []
    for package in successful:
        if package in test_mapping:
            if test_import(package, test_mapping[package]):
                working.append(package)
    
    # Summary
    print(f"\n" + "="*55)
    print(f"📊 Installation Summary:")
    print(f"   ✅ Successful: {len(successful)} packages")
    print(f"   ❌ Failed: {len(failed)} packages")
    print(f"   🧪 Working: {len(working)} packages")
    
    if working:
        print(f"\n🎉 Ready to use: {', '.join(working)}")
    
    if failed:
        print(f"\n⚠️ Failed packages: {', '.join(failed)}")
        print("You can try installing these manually later")
    
    # Next steps
    if "ultralytics" in working and "opencv-python" in working:
        print(f"\n🚀 Great news! You have the core packages needed:")
        print("   ✅ YOLO model (ultralytics)")
        print("   ✅ Image processing (opencv)")
        print("\nNext steps:")
        print("1. Run: python dataset_checker.py")
        print("2. We can start training with your existing images!")
    else:
        print(f"\n📋 Still need to install core packages manually:")
        if "ultralytics" not in working:
            print("   pip install ultralytics")
        if "opencv-python" not in working:
            print("   pip install opencv-python")

if __name__ == "__main__":
    main()