#!/usr/bin/env python3
"""
Create YOLO dataset configuration from image folders
"""
import os
import yaml
import argparse
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_yolo_dataset(data_dir: Path, output_dir: Path, train_split: float = 0.7, val_split: float = 0.2):
    """
    Create YOLO dataset structure from organized image folders
    
    Args:
        data_dir: Directory containing brand folders with images
        output_dir: Output directory for YOLO dataset
        train_split: Percentage for training set
        val_split: Percentage for validation set
    """
    
    # Create output structure
    dataset_dir = output_dir / "dataset"
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get brand folders
    brand_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    brand_to_id = {folder.name: i for i, folder in enumerate(brand_folders)}
    
    logger.info(f"Found {len(brand_folders)} brand categories: {list(brand_to_id.keys())}")
    
    all_files = []
    for brand_folder in brand_folders:
        brand_name = brand_folder.name
        brand_id = brand_to_id[brand_name]
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in brand_folder.iterdir() 
                 if f.suffix.lower() in image_extensions]
        
        for img_file in images:
            all_files.append({
                'image_path': img_file,
                'brand_name': brand_name,
                'brand_id': brand_id
            })
    
    logger.info(f"Total images found: {len(all_files)}")
    
    # Split dataset
    train_files, temp_files = train_test_split(all_files, train_size=train_split, random_state=42)
    val_size = val_split / (1 - train_split)
    val_files, test_files = train_test_split(temp_files, train_size=val_size, random_state=42)
    
    # Process each split
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        logger.info(f"Processing {split_name} split: {len(files)} images")
        
        for file_info in files:
            img_path = file_info['image_path']
            brand_id = file_info['brand_id']
            
            # Copy image
            new_img_name = f"{brand_id}_{img_path.stem}_{img_path.suffix}"
            dst_img_path = dataset_dir / split_name / 'images' / new_img_name
            shutil.copy2(img_path, dst_img_path)
            
            # Create dummy label (whole image for now - you'd need actual annotations)
            label_content = f"{brand_id} 0.5 0.5 1.0 1.0\n"  # Center, full image
            label_path = dataset_dir / split_name / 'labels' / f"{new_img_name.split('.')[0]}.txt"
            with open(label_path, 'w') as f:
                f.write(label_content)
    
    # Create dataset YAML
    yaml_content = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(brand_folders),
        'names': {i: name for name, i in brand_to_id.items()}
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    logger.info(f"Dataset created successfully!")
    logger.info(f"Dataset config saved to: {yaml_path}")
    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description='Create YOLO dataset from organized folders')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory with brand image folders')
    parser.add_argument('--output-dir', type=Path, default=Path('data'), help='Output directory')
    parser.add_argument('--train-split', type=float, default=0.7, help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    create_yolo_dataset(args.data_dir, args.output_dir, args.train_split, args.val_split)
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
