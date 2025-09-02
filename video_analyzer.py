#!/usr/bin/env python3
"""
Brandlytics Video Analysis CLI
Professional command-line interface for video brand detection.

Usage:
    python video_analyzer.py analyze video.mp4
    python video_analyzer.py batch /path/to/videos/
    python video_analyzer.py compare video1.mp4 video2.mp4 video3.mp4
    python video_analyzer.py report --video-name "brands_analysis"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add app to path for imports
sys.path.append(str(Path(__file__).parent))

from app.services.video_analysis import VideoAnalysisService, analyze_video, analyze_video_folder
from app.database.operations import get_video_analytics
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoAnalyzer:
    """Professional video analysis command-line interface."""
    
    def __init__(self):
        self.service = VideoAnalysisService()
    
    def analyze_single_video(self, video_path: str, video_name: str = None, 
                           max_frames: int = None, no_database: bool = False) -> Dict:
        """Analyze a single video file."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if video_name is None:
            video_name = Path(video_path).stem + "_analysis"
        
        print(f"🎬 Starting analysis of: {video_path}")
        print(f"📋 Video identifier: {video_name}")
        
        results = self.service.process_video(
            video_path=video_path,
            video_name=video_name,
            save_to_database=not no_database,
            max_frames=max_frames
        )
        
        self._print_analysis_summary(results)
        return results
    
    def analyze_video_batch(self, folder_path: str, no_database: bool = False) -> Dict:
        """Analyze all videos in a folder."""
        if not Path(folder_path).exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        print(f"📁 Starting batch analysis of folder: {folder_path}")
        
        results = analyze_video_folder(folder_path, save_to_database=not no_database)
        
        print(f"\n📊 BATCH ANALYSIS SUMMARY")
        print("=" * 50)
        
        total_videos = len(results)
        successful = sum(1 for r in results.values() if 'error' not in r)
        failed = total_videos - successful
        
        print(f"Total videos processed: {total_videos}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if successful > 0:
            total_detections = sum(
                r.get('summary', {}).get('total_detections', 0) 
                for r in results.values() if 'error' not in r
            )
            all_brands = set()
            for r in results.values():
                if 'error' not in r:
                    all_brands.update(r.get('summary', {}).get('unique_brands', []))
            
            print(f"Total detections across all videos: {total_detections}")
            print(f"Unique brands detected: {len(all_brands)}")
            print(f"Brands: {', '.join(sorted(all_brands))}")
        
        return results
    
    def compare_videos(self, video_paths: List[str], no_database: bool = False) -> Dict:
        """Compare multiple videos and generate ranking."""
        print(f"🏆 Starting comparison of {len(video_paths)} videos")
        
        results = {}
        for video_path in video_paths:
            if not Path(video_path).exists():
                print(f"⚠️  Warning: Video not found: {video_path}")
                continue
            
            video_name = Path(video_path).stem + "_comparison"
            print(f"\n🎬 Analyzing: {Path(video_path).name}")
            
            results[video_path] = self.service.process_video(
                video_path=video_path,
                video_name=video_name,
                save_to_database=not no_database
            )
        
        # Generate comparison report
        comparison = self._generate_comparison_report(results)
        self._print_comparison_results(comparison)
        
        return comparison
    
    def generate_report(self, video_name: str = None, output_file: str = None) -> Dict:
        """Generate detailed report from database."""
        try:
            if video_name:
                print(f"📊 Generating report for: {video_name}")
                data = get_video_analytics(video_name)
            else:
                print("📊 Generating comprehensive report for all videos")
                # This would need additional database function
                data = {"message": "Use --video-name to specify a video"}
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                print(f"💾 Report saved to: {output_file}")
            
            return data
            
        except Exception as e:
            print(f"❌ Failed to generate report: {e}")
            return {"error": str(e)}
    
    def _print_analysis_summary(self, results: Dict) -> None:
        """Print formatted analysis summary."""
        summary = results.get('summary', {})
        processing = results.get('processing_info', {})
        video_info = results.get('video_info', {})
        brand_analysis = results.get('brand_analysis', {})
        
        print(f"\n📊 ANALYSIS RESULTS")
        print("=" * 50)
        
        # Video info
        print(f"Video: {processing.get('video_name', 'N/A')}")
        print(f"Duration: {video_info.get('duration_seconds', 0):.1f} seconds")
        print(f"File size: {video_info.get('file_size_mb', 0):.1f} MB")
        print(f"FPS: {video_info.get('fps', 0):.1f}")
        
        # Processing info
        print(f"\nProcessing:")
        print(f"  Frames processed: {processing.get('processed_frames', 0):,}")
        print(f"  Coverage: {processing.get('coverage_percentage', 0):.1f}%")
        print(f"  Processing time: {processing.get('total_processing_time', 0):.2f} seconds")
        
        # Detection results
        print(f"\nDetection Results:")
        print(f"  Total detections: {summary.get('total_detections', 0)}")
        print(f"  Unique brands: {summary.get('brands_count', 0)}")
        print(f"  Average confidence: {summary.get('avg_confidence', 0):.3f}")
        print(f"  Saved to database: {summary.get('detections_saved_to_db', 0)}")
        
        # Brand breakdown
        if brand_analysis:
            print(f"\nBrand Breakdown:")
            for brand, stats in sorted(brand_analysis.items(), 
                                     key=lambda x: x[1]['count'], reverse=True):
                print(f"  {brand}: {stats['count']} detections "
                      f"(avg conf: {stats['avg_confidence']:.3f})")
        
        print("=" * 50)
    
    def _generate_comparison_report(self, results: Dict) -> Dict:
        """Generate comparison report from multiple video results."""
        comparison_data = []
        
        for video_path, result in results.items():
            if 'error' in result:
                continue
            
            video_name = Path(video_path).name
            summary = result.get('summary', {})
            processing = result.get('processing_info', {})
            video_info = result.get('video_info', {})
            
            comparison_data.append({
                'video_name': video_name,
                'video_path': video_path,
                'total_detections': summary.get('total_detections', 0),
                'unique_brands': summary.get('brands_count', 0),
                'avg_confidence': summary.get('avg_confidence', 0),
                'duration_seconds': video_info.get('duration_seconds', 0),
                'file_size_mb': video_info.get('file_size_mb', 0),
                'processing_time': processing.get('total_processing_time', 0),
                'detection_rate': summary.get('total_detections', 0) / video_info.get('duration_seconds', 1),
                'brands': summary.get('unique_brands', [])
            })
        
        # Sort by total detections (descending)
        comparison_data.sort(key=lambda x: x['total_detections'], reverse=True)
        
        return {
            'comparison_date': datetime.now().isoformat(),
            'total_videos': len(comparison_data),
            'rankings': comparison_data,
            'winner': comparison_data[0] if comparison_data else None,
            'summary_stats': self._calculate_summary_stats(comparison_data)
        }
    
    def _calculate_summary_stats(self, comparison_data: List[Dict]) -> Dict:
        """Calculate summary statistics across all videos."""
        if not comparison_data:
            return {}
        
        total_detections = sum(v['total_detections'] for v in comparison_data)
        all_brands = set()
        for v in comparison_data:
            all_brands.update(v['brands'])
        
        return {
            'total_detections_all_videos': total_detections,
            'total_unique_brands': len(all_brands),
            'avg_detections_per_video': total_detections / len(comparison_data),
            'all_brands_detected': sorted(list(all_brands))
        }
    
    def _print_comparison_results(self, comparison: Dict) -> None:
        """Print formatted comparison results."""
        rankings = comparison.get('rankings', [])
        summary_stats = comparison.get('summary_stats', {})
        
        print(f"\n🏆 VIDEO COMPARISON RESULTS")
        print("=" * 70)
        
        if rankings:
            print(f"Winner: {rankings[0]['video_name']} "
                  f"({rankings[0]['total_detections']} detections)")
            
            print(f"\nRankings:")
            for i, video in enumerate(rankings, 1):
                print(f"{i:2d}. {video['video_name']:<20} "
                      f"{video['total_detections']:>4d} detections "
                      f"({video['unique_brands']:>2d} brands, "
                      f"avg conf: {video['avg_confidence']:.3f})")
            
            print(f"\nSummary Statistics:")
            print(f"  Total detections across all videos: {summary_stats.get('total_detections_all_videos', 0)}")
            print(f"  Total unique brands: {summary_stats.get('total_unique_brands', 0)}")
            print(f"  Average detections per video: {summary_stats.get('avg_detections_per_video', 0):.1f}")
            
            if summary_stats.get('all_brands_detected'):
                print(f"  All brands detected: {', '.join(summary_stats['all_brands_detected'])}")
        
        print("=" * 70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Brandlytics Video Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze video.mp4
  %(prog)s analyze video.mp4 --video-name "my_analysis" --max-frames 1000
  %(prog)s batch /path/to/videos/
  %(prog)s compare video1.mp4 video2.mp4 video3.mp4
  %(prog)s report --video-name "brands_analysis" --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single video')
    analyze_parser.add_argument('video_path', help='Path to video file')
    analyze_parser.add_argument('--video-name', help='Custom video identifier')
    analyze_parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    analyze_parser.add_argument('--no-database', action='store_true', 
                               help='Skip saving to database')
    analyze_parser.add_argument('--output', help='Save results to JSON file')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze all videos in folder')
    batch_parser.add_argument('folder_path', help='Path to folder containing videos')
    batch_parser.add_argument('--no-database', action='store_true', 
                             help='Skip saving to database')
    batch_parser.add_argument('--output', help='Save results to JSON file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple videos')
    compare_parser.add_argument('video_paths', nargs='+', help='Paths to video files')
    compare_parser.add_argument('--no-database', action='store_true', 
                               help='Skip saving to database')
    compare_parser.add_argument('--output', help='Save comparison to JSON file')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate report from database')
    report_parser.add_argument('--video-name', help='Specific video to report on')
    report_parser.add_argument('--output', help='Save report to JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    analyzer = VideoAnalyzer()
    
    try:
        if args.command == 'analyze':
            results = analyzer.analyze_single_video(
                args.video_path, args.video_name, args.max_frames, args.no_database
            )
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Results saved to: {args.output}")
        
        elif args.command == 'batch':
            results = analyzer.analyze_video_batch(args.folder_path, args.no_database)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Results saved to: {args.output}")
        
        elif args.command == 'compare':
            results = analyzer.compare_videos(args.video_paths, args.no_database)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"💾 Comparison saved to: {args.output}")
        
        elif args.command == 'report':
            results = analyzer.generate_report(args.video_name, args.output)
            
            if not args.output:
                print(json.dumps(results, indent=2, default=str))
    
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
