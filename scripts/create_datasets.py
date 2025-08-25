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
    dataset_dir = output_dir