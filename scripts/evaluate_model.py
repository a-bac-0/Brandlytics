#!/usr/bin/env python3
"""
Evaluate trained model performance
"""
import argparse
import yaml
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.detector import BrandDetector
from src.utils.metrics import DetectionMetrics
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--model', type=Path, required=True, help='Trained model path')
    parser.add_argument('--data', type=Path, help='Test dataset YAML')
    parser.add_argument('--config', type=Path, default=Path('config/config.yaml'), help='Config file')
    parser.add_argument('--output', type=Path, help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        logger.info(f"Loading model: {args.model}")
        
        # Load model using ultralytics for evaluation
        model = YOLO(str(args.model))
        
        # Run validation
        if args.data:
            logger.info(f"Evaluating on dataset: {args.data}")
            metrics = model.val(data=str(args.data))
            
            # Print results
            logger.info("\n=== Evaluation Results ===")
            logger.info(f"mAP@0.5: {metrics.box.map50:.4f}")
            logger.info(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
            logger.info(f"Precision: {metrics.box.mp:.4f}")
            logger.info(f"Recall: {metrics.box.mr:.4f}")
            
            # Per-class metrics
            if hasattr(metrics.box, 'maps'):
                logger.info("\nPer-class mAP@0.5:")
                class_names = config.get('brands', [])
                for i, map_score in enumerate(metrics.box.maps):
                    brand_name = class_names[i]['name'] if i < len(class_names) else f"Class_{i}"
                    logger.info(f"  {brand_name}: {map_score:.4f}")
        else:
            logger.warning("No test dataset provided, skipping detailed evaluation")
        
        # Model info
        logger.info(f"\nModel Summary:")
        logger.info(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
        logger.info(f"Model size: {args.model.stat().st_size / (1024*1024):.1f} MB")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())