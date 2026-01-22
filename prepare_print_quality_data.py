"""
Script to prepare print quality dataset for training.

This script helps organize print sample images into train/test splits
for training INP-Former for print quality detection.

Dataset Structure:
    print_quality_data/
    ├── train/
    │   ├── good_print/          (normal, good quality prints)
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── defective_print/     (optional: defective prints for reference)
    │       ├── image1.jpg
    │       └── ...
    └── test/
        ├── good_print/
        │   └── ...
        └── defective_print/
            └── ...
"""

import os
import shutil
import argparse
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrintQualityDatasetPreparer:
    def __init__(self, output_dir='./print_quality_data'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_directory_structure(self):
        """Create standard directory structure"""
        dirs = [
            self.output_dir / 'train' / 'good_print',
            self.output_dir / 'train' / 'defective_print',
            self.output_dir / 'test' / 'good_print',
            self.output_dir / 'test' / 'defective_print',
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def organize_images(self, source_dir, train_ratio=0.8):
        """
        Organize images from source directory into train/test splits.
        
        Args:
            source_dir: Path to directory containing print quality images
            train_ratio: Ratio of images to use for training
        """
        source_path = Path(source_dir)
        
        if not source_path.exists():
            logger.error(f"Source directory not found: {source_dir}")
            logger.info("Creating empty directory structure for manual population")
            self.create_directory_structure()
            return
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in source_path.rglob('*')
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            logger.warning(f"No image files found in {source_dir}")
            self.create_directory_structure()
            return
        
        logger.info(f"Found {len(image_files)} images")
        
        # Split into train/test
        train_files, test_files = train_test_split(
            image_files,
            train_size=train_ratio,
            random_state=42
        )
        
        logger.info(f"Train: {len(train_files)}, Test: {len(test_files)}")
        
        # Copy files to train folder (assuming all are good quality for training)
        for file_path in tqdm(train_files, desc="Copying train images"):
            try:
                dest = self.output_dir / 'train' / 'good_print' / file_path.name
                shutil.copy2(file_path, dest)
            except Exception as e:
                logger.error(f"Error copying {file_path}: {e}")
        
        # Copy files to test folder
        for file_path in tqdm(test_files, desc="Copying test images"):
            try:
                dest = self.output_dir / 'test' / 'good_print' / file_path.name
                shutil.copy2(file_path, dest)
            except Exception as e:
                logger.error(f"Error copying {file_path}: {e}")
        
        logger.info("Dataset organization completed!")
    
    def create_sample_images(self, num_samples=20):
        """
        Create sample synthetic images for testing purposes.
        These represent good quality prints.
        """
        logger.info(f"Creating {num_samples} sample images...")
        
        self.create_directory_structure()
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            # Create a simple synthetic "print" image
            # In practice, use your actual print images
            img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
            
            # Add some structure to simulate printed content
            img[50:100, 50:100] = 50  # Dark region
            img[150:200, 150:200] = 230  # Light region
            
            # Save to train folder
            img_pil = Image.fromarray(img)
            train_path = self.output_dir / 'train' / 'good_print' / f'sample_train_{i:04d}.png'
            img_pil.save(train_path)
            
            # Also save some to test folder (80% train, 20% test)
            if i % 5 == 0:
                test_path = self.output_dir / 'test' / 'good_print' / f'sample_test_{i:04d}.png'
                img_pil.save(test_path)
        
        logger.info(f"Sample images created in {self.output_dir}")
    
    def print_structure(self):
        """Print directory structure"""
        logger.info("\nDataset Structure:")
        for root, dirs, files in os.walk(self.output_dir):
            level = root.replace(str(self.output_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            logger.info(f'{indent}{os.path.basename(root)}/')
            sub_indent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Show first 3 files
                logger.info(f'{sub_indent}{file}')
            if len(files) > 3:
                logger.info(f'{sub_indent}... and {len(files) - 3} more files')


def main():
    parser = argparse.ArgumentParser(description='Prepare print quality dataset')
    parser.add_argument('--output_dir', type=str, default='./print_quality_data',
                        help='Output directory for organized dataset')
    parser.add_argument('--source_dir', type=str, default=None,
                        help='Source directory containing print images (optional)')
    parser.add_argument('--create_samples', action='store_true',
                        help='Create synthetic sample images for testing')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of synthetic samples to create')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of images to use for training')
    
    args = parser.parse_args()
    
    preparer = PrintQualityDatasetPreparer(args.output_dir)
    
    if args.create_samples:
        logger.info("Creating synthetic sample images...")
        preparer.create_sample_images(args.num_samples)
    elif args.source_dir:
        logger.info(f"Organizing images from {args.source_dir}...")
        preparer.organize_images(args.source_dir, args.train_ratio)
    else:
        logger.info("Creating empty directory structure")
        preparer.create_directory_structure()
    
    preparer.print_structure()
    logger.info("\nDataset preparation completed!")


if __name__ == '__main__':
    main()
