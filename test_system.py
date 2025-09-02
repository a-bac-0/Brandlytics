#!/usr/bin/env python3
"""
Test script for the unified video analysis system.
Quick verification that everything is working correctly.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.services.video_analysis import VideoAnalysisService
from app.config.video_analysis_config import VideoAnalysisConfig, load_preset_config


def test_config_system():
    """Test configuration system."""
    print("🔧 Testing configuration system...")
    
    # Test default config
    config = VideoAnalysisConfig()
    processing_config = config.get_processing_config()
    assert processing_config['confidence_threshold'] == 0.5
    print("✅ Default configuration loaded")
    
    # Test presets
    fast_config = load_preset_config('fast')
    fast_processing = fast_config.get_processing_config()
    assert fast_processing['confidence_threshold'] == 0.7
    print("✅ Preset configuration loaded")


def test_service_initialization():
    """Test video analysis service initialization."""
    print("🎬 Testing service initialization...")
    
    try:
        service = VideoAnalysisService()
        assert service.model is not None
        print("✅ VideoAnalysisService initialized successfully")
        print(f"✅ Model loaded from: {service.model_path}")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")
        return False
    
    return True


def test_video_info_extraction():
    """Test video info extraction without processing."""
    print("📹 Testing video info extraction...")
    
    # Look for any video files in common locations
    video_paths = []
    
    # Check for test videos
    possible_locations = [
        "data/raw/videos/",
        ".",
        "tests/"
    ]
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for location in possible_locations:
        location_path = Path(location)
        if location_path.exists():
            for ext in video_extensions:
                video_files = list(location_path.glob(f"*{ext}"))
                video_paths.extend(video_files)
    
    if not video_paths:
        print("⚠️  No video files found for testing")
        return True
    
    # Test with first available video
    video_path = str(video_paths[0])
    print(f"📹 Testing with: {video_path}")
    
    try:
        service = VideoAnalysisService()
        video_info = service.get_video_info(video_path)
        
        assert 'duration_seconds' in video_info
        assert 'fps' in video_info
        assert 'total_frames' in video_info
        
        print("✅ Video info extraction successful")
        print(f"   Duration: {video_info['duration_seconds']:.1f}s")
        print(f"   FPS: {video_info['fps']:.1f}")
        print(f"   Frames: {video_info['total_frames']}")
        
    except Exception as e:
        print(f"❌ Video info extraction failed: {e}")
        return False
    
    return True


def run_all_tests():
    """Run all tests."""
    print("🧪 Running Brandlytics Video Analysis System Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration System", test_config_system),
        ("Service Initialization", test_service_initialization),
        ("Video Info Extraction", test_video_info_extraction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            if result is not False:
                passed += 1
                print(f"✅ {test_name} PASSED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 Quick start commands:")
        print("   python video_analyzer.py analyze your_video.mp4")
        print("   python video_analyzer.py batch /path/to/videos/")
    else:
        print("⚠️  Some tests failed. Please check the configuration.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
