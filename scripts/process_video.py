#!/usr/bin/env python3
"""
Process video for brand detection
"""
import argparse
import yaml
import json
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.detector import BrandDetector
from src.utils.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description='Process video for brand detection')
    parser.add_argument('--input', type=Path, required=True, help='Input video path')
    parser.add_argument('--output', type=Path, help='Output directory')
    parser.add_argument('--model', type=Path, default=Path('models/trained/best.pt'), help='Model path')
    parser.add_argument('--config', type=Path, default=Path('config/config.yaml'), help='Config file')
    parser.add_argument('--save-video', action='store_true', help='Save annotated video')
    parser.add_argument('--save-report', action='store_true', help='Save analysis report')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        output_video = args.output / f"annotated_{args.input.name}" if args.save_video else None
    else:
        output_video = None
    
    try:
        # Initialize detector and processor
        logger.info(f"Loading model from {args.model}")
        detector = BrandDetector(str(args.model), config)
        processor = VideoProcessor(detector, config)
        
        # Process video
        logger.info(f"Processing video: {args.input}")
        results = processor.process_video(
            str(args.input), 
            str(output_video) if output_video else None
        )
        
        # Print summary
        logger.info("Processing completed!")
        logger.info(f"Video duration: {results['video_info']['duration']:.1f}s")
        logger.info(f"Processed frames: {results['video_info']['processed_frames']}")
        logger.info(f"Processing time: {results['video_info']['processing_time']:.1f}s")
        
        if results['brand_detections']:
            logger.info("\nBrand Analysis:")
            for brand_name, brand_data in results['brand_detections'].items():
                logger.info(f"  {brand_name}:")
                logger.info(f"    Appearances: {brand_data['total_frames']}")
                logger.info(f"    Screen time: {brand_data['total_time']:.1f}s ({brand_data.get('appearance_percentage', 0):.1f}%)")
                logger.info(f"    Avg confidence: {brand_data.get('avg_confidence', 0):.2f}")
        else:
            logger.info("No brands detected in video")
        
        # Save report
        if args.save_report and args.output:
            report_path = args.output / f"report_{args.input.stem}.json"
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Report saved to: {report_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())