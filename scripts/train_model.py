import os
import yaml
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from src.data.data_loader import BrandDatasetLoader
from src.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Train Brand Detection Model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    parser.add_argument('--data', type=str, required=True, help='Dataset YAML file path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Training device')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger = setup_logger('training', config.get('logging', {}))
    
    # Initialize model
    model_config = config.get('model', {})
    if model_config.get('pretrained', True):
        model = YOLO('yolov8n.pt')  # Start with pretrained YOLO
    else:
        model = YOLO('yolov8n.yaml')  # Start from scratch
    
    logger.info(f"Starting training with {args.epochs} epochs")
    
    # Training parameters
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': model_config.get('input_size', [640, 640])[0],
        'device': args.device,
        'project': 'models/trained',
        'name': 'brand_detection',
        'save_period': 10,
        'patience': config.get('training', {}).get('early_stopping_patience', 10),
        'lr0': config.get('training', {}).get('learning_rate', 0.01),
        'optimizer': config.get('training', {}).get('optimizer', 'auto'),
        'augment': config.get('augmentation', {}).get('enable', True),
        'hsv_h': config.get('augmentation', {}).get('hue', 0.015),
        'hsv_s': config.get('augmentation', {}).get('saturation', 0.7),
        'hsv_v': config.get('augmentation', {}).get('brightness', 0.4),
        'flipud': config.get('augmentation', {}).get('vertical_flip', 0.0),
        'fliplr': config.get('augmentation', {}).get('horizontal_flip', 0.5),
        'degrees': config.get('augmentation', {}).get('rotation', 0.0),
    }
    
    # Start training
    try:
        results = model.train(**train_params)
        logger.info(f"Training completed successfully")
        logger.info(f"Best model saved to: {results.save_dir}")
        
        # Validation
        metrics = model.val()
        logger.info(f"Validation mAP50: {metrics.box.map50:.4f}")
        logger.info(f"Validation mAP50-95: {metrics.box.map:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()