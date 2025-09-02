#!/usr/bin/env python3
"""
Brandlytics Video Analysis - Usage Examples
Demonstrates how to use the unified video analysis system.
"""

import json
from pathlib import Path

# Import the unified system
from app.services.video_analysis import analyze_video, analyze_video_folder, VideoAnalysisService
from app.config.video_analysis_config import VideoAnalysisConfig, load_preset_config


def example_1_simple_analysis():
    """Example 1: Simple video analysis with default settings."""
    print("📹 Example 1: Simple Video Analysis")
    print("-" * 40)
    
    # Analyze a single video with default settings
    results = analyze_video(
        video_path="data/raw/videos/input/Coca-Cola.mp4",
        video_name="coca_cola_example",
        save_to_database=False  # Skip database for example
    )
    
    # Print summary
    summary = results['summary']
    print(f"✅ Analysis complete!")
    print(f"   Total detections: {summary['total_detections']}")
    print(f"   Unique brands: {len(summary['unique_brands'])}")
    print(f"   Brands found: {', '.join(summary['unique_brands'])}")
    print(f"   Average confidence: {summary['avg_confidence']:.3f}")


def example_2_advanced_analysis():
    """Example 2: Advanced analysis with custom settings."""
    print("\n🔧 Example 2: Advanced Analysis with Custom Settings")
    print("-" * 50)
    
    # Create service with custom settings
    service = VideoAnalysisService()
    
    results = service.process_video(
        video_path="data/raw/videos/input/brands.mp4",
        video_name="brands_advanced_example",
        save_to_database=False,
        max_frames=100,  # Limit to 100 frames for speed
        confidence_threshold=0.7  # Higher confidence threshold
    )
    
    # Print detailed results
    processing = results['processing_info']
    print(f"✅ Advanced analysis complete!")
    print(f"   Processed frames: {processing['processed_frames']}")
    print(f"   Coverage: {processing['coverage_percentage']:.1f}%")
    print(f"   Processing time: {processing['total_processing_time']:.2f}s")
    print(f"   Detection rate: {processing['detections_per_second']:.2f} det/s")


def example_3_configuration_presets():
    """Example 3: Using configuration presets."""
    print("\n⚙️ Example 3: Configuration Presets")
    print("-" * 35)
    
    # Load different presets
    presets = ['fast', 'balanced', 'thorough']
    
    for preset_name in presets:
        config = load_preset_config(preset_name)
        processing_config = config.get_processing_config()
        
        print(f"{preset_name.title()} preset:")
        print(f"  Confidence threshold: {processing_config['confidence_threshold']}")
        print(f"  Max frames: {processing_config['max_frames']}")
        
        # Show sampling strategy for medium videos
        strategy = config.get_sampling_strategy(180)  # 3-minute video
        print(f"  Sampling interval: {strategy['sampling_interval']}")


def example_4_batch_processing():
    """Example 4: Batch processing demonstration."""
    print("\n📁 Example 4: Batch Processing")
    print("-" * 30)
    
    # Process all videos in the input folder
    results = analyze_video_folder(
        folder_path="data/raw/videos/input",
        save_to_database=False
    )
    
    print(f"✅ Batch processing complete!")
    print(f"   Videos processed: {len(results)}")
    
    # Show summary for each video
    for video_name, result in results.items():
        if 'error' not in result:
            total_detections = result['summary']['total_detections']
            brands = len(result['summary']['unique_brands'])
            print(f"   {video_name}: {total_detections} detections, {brands} brands")


def example_5_comparison_analysis():
    """Example 5: Video comparison and ranking."""
    print("\n🏆 Example 5: Video Comparison")
    print("-" * 30)
    
    service = VideoAnalysisService()
    
    # Compare multiple videos
    video_paths = [
        "data/raw/videos/input/brands.mp4",
        "data/raw/videos/input/Coca-Cola.mp4",
        "data/raw/videos/input/0901.mp4"
    ]
    
    comparison_results = {}
    
    for video_path in video_paths:
        video_name = Path(video_path).stem + "_comparison_example"
        
        # Process with limited frames for speed
        result = service.process_video(
            video_path=video_path,
            video_name=video_name,
            save_to_database=False,
            max_frames=50  # Quick comparison
        )
        
        comparison_results[video_path] = result
    
    # Rank videos by detection count
    rankings = []
    for video_path, result in comparison_results.items():
        video_name = Path(video_path).name
        detections = result['summary']['total_detections']
        brands = len(result['summary']['unique_brands'])
        
        rankings.append({
            'video': video_name,
            'detections': detections,
            'brands': brands
        })
    
    # Sort by detections (descending)
    rankings.sort(key=lambda x: x['detections'], reverse=True)
    
    print("🏆 Video Rankings:")
    for i, ranking in enumerate(rankings, 1):
        print(f"   {i}. {ranking['video']}: {ranking['detections']} detections, {ranking['brands']} brands")


def main():
    """Run all examples."""
    print("🚀 Brandlytics Video Analysis - Usage Examples")
    print("=" * 60)
    
    try:
        # Check if we have video files
        video_folder = Path("data/raw/videos/input")
        if not video_folder.exists():
            print("❌ Video folder not found. Please ensure videos are in data/raw/videos/input/")
            return
        
        video_files = list(video_folder.glob("*.mp4"))
        if not video_files:
            print("❌ No video files found. Please add some videos to data/raw/videos/input/")
            return
        
        print(f"📹 Found {len(video_files)} video files for examples")
        
        # Run examples
        example_1_simple_analysis()
        example_2_advanced_analysis()
        example_3_configuration_presets()
        example_4_batch_processing()
        example_5_comparison_analysis()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n🎯 Next steps:")
        print("   • Try the CLI: python video_analyzer.py --help")
        print("   • Customize config: edit video_analysis_config.yaml")
        print("   • Integrate with your app: import VideoAnalysisService")
        
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()
