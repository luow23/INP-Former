"""
Inference script for print quality detection using trained INP-Former model.
This script loads a trained model and predicts quality scores for print samples.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import argparse
import logging
from PIL import Image
import cv2
from tqdm import tqdm
import json
from datetime import datetime

# Model imports
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

# Utils
from dataset import get_data_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrintQualityInference:
    def __init__(self, model_path, encoder='dino_vits14', device=None):
        """
        Initialize inference model.
        
        Args:
            model_path: Path to trained model checkpoint
            encoder: Vision transformer encoder name
            device: Device to use (cuda/cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_name = encoder
        self.model_path = model_path
        
        # Setup transforms
        self.data_transform, _ = get_data_transforms(224, 224)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load encoder
        encoder = vit_encoder.load(self.encoder_name)
        
        # Get embedding dimensions
        if 'small' in self.encoder_name:
            embed_dim, num_heads = 384, 6
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        elif 'base' in self.encoder_name:
            embed_dim, num_heads = 768, 12
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        elif 'large' in self.encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise ValueError("Encoder must be small, base, or large")
        
        # Build model components
        bottleneck = nn.ModuleList([
            Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)
        ])
        
        inp_extractor = nn.ModuleList([
            Aggregation_Block(embed_dim, num_heads, num_heads * 64)
            for _ in range(len(target_layers))
        ])
        
        inp_guided_decoder = nn.ModuleList([
            Aggregation_Block(embed_dim, num_heads, num_heads * 64)
            for _ in range(len(target_layers))
        ])
        
        inp_prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(20, embed_dim))
            for _ in range(1)
        ])
        
        # Create model
        model = INP_Former(
            encoder,
            target_layers,
            bottleneck,
            inp_extractor,
            inp_guided_decoder,
            inp_prototypes,
            args=None
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def predict_image(self, image_path):
        """
        Predict quality score for a single image.
        
        Args:
            image_path: Path to print sample image
            
        Returns:
            Quality score (lower = better quality, higher = more defects)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.data_transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            # Get reconstruction error (anomaly score)
            anomaly_map = self.model.predict(image_tensor)
        
        # Calculate quality score (normalize and invert)
        quality_score = float(anomaly_map.mean().cpu().numpy())
        
        return quality_score
    
    def batch_predict(self, image_dir, output_file=None):
        """
        Predict quality scores for all images in a directory.
        
        Args:
            image_dir: Directory containing print images
            output_file: Optional file to save results as JSON
            
        Returns:
            Dictionary with results
        """
        image_dir = Path(image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Find all images
        image_files = sorted([
            f for f in image_dir.rglob('*')
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            logger.warning(f"No images found in {image_dir}")
            return {}
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'predictions': []
        }
        
        logger.info(f"Processing {len(image_files)} images...")
        
        for image_path in tqdm(image_files):
            try:
                quality_score = self.predict_image(str(image_path))
                results['predictions'].append({
                    'image': str(image_path),
                    'quality_score': quality_score,
                    'status': 'good' if quality_score < 0.5 else 'defective'
                })
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results['predictions'].append({
                    'image': str(image_path),
                    'error': str(e)
                })
        
        # Calculate statistics
        scores = [p['quality_score'] for p in results['predictions'] if 'quality_score' in p]
        if scores:
            results['statistics'] = {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'good_count': sum(1 for p in results['predictions'] if p.get('status') == 'good'),
                'defective_count': sum(1 for p in results['predictions'] if p.get('status') == 'defective')
            }
        
        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        return results
    
    def generate_quality_report(self, results):
        """Generate a formatted quality report"""
        if not results.get('predictions'):
            logger.warning("No predictions available for report")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("PRINT QUALITY DETECTION REPORT")
        logger.info("=" * 60)
        
        if 'statistics' in results:
            stats = results['statistics']
            logger.info(f"\nStatistics:")
            logger.info(f"  Mean Quality Score: {stats['mean_score']:.4f}")
            logger.info(f"  Std Dev: {stats['std_score']:.4f}")
            logger.info(f"  Range: [{stats['min_score']:.4f}, {stats['max_score']:.4f}]")
            logger.info(f"  Good Samples: {stats['good_count']}")
            logger.info(f"  Defective Samples: {stats['defective_count']}")
        
        logger.info(f"\nPredictions ({len(results['predictions'])} total):")
        logger.info("-" * 60)
        
        for pred in results['predictions'][:10]:  # Show first 10
            image_name = Path(pred['image']).name
            if 'quality_score' in pred:
                score = pred['quality_score']
                status = pred['status']
                logger.info(f"  {image_name:<40} | Score: {score:>7.4f} | {status}")
            else:
                logger.info(f"  {image_name:<40} | Error: {pred.get('error')}")
        
        if len(results['predictions']) > 10:
            logger.info(f"  ... and {len(results['predictions']) - 10} more images")
        
        logger.info("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Inference for print quality detection')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to image or directory of images for inference')
    parser.add_argument('--encoder', type=str, default='dino_vits14',
                        choices=['dino_vits14', 'dino_base14', 'dino_large14'],
                        help='Vision transformer encoder')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file to save results as JSON')
    parser.add_argument('--batch_mode', action='store_true',
                        help='Process directory instead of single image')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = PrintQualityInference(args.model_path, args.encoder)
    
    # Run inference
    if args.batch_mode or os.path.isdir(args.image_path):
        results = inference.batch_predict(args.image_path, args.output_file)
        inference.generate_quality_report(results)
    else:
        score = inference.predict_image(args.image_path)
        logger.info(f"\nQuality Score: {score:.4f}")
        logger.info(f"Status: {'Good' if score < 0.5 else 'Defective'}")


if __name__ == '__main__':
    main()
