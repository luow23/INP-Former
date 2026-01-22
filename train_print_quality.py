"""
Training script for INP-Former to detect print quality issues
without focusing on content inside the print sample.

Purpose: Quality inspection of printed materials (ink saturation, clarity, defects)
"""

import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Dataset imports
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Model imports
from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

# Utils
from utils import setup_seed, WarmCosineScheduler
from optimizers import StableAdamW

warnings.filterwarnings("ignore")

class PrintQualityTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_seed(args.seed)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.args.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logger initialized. Log file: {log_file}")
        
    def setup_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def prepare_data(self):
        """Prepare training and test data"""
        self.logger.info("Preparing datasets...")
        
        # Data transforms
        data_transform, gt_transform = get_data_transforms(
            self.args.input_size, 
            self.args.crop_size
        )
        
        # Load data from specified paths
        train_path = os.path.join(self.args.data_path, 'train')
        test_path = os.path.join(self.args.data_path, 'test')
        
        # Verify paths exist
        if not os.path.exists(train_path):
            self.logger.error(f"Training data path not found: {train_path}")
            raise FileNotFoundError(f"Training data path not found: {train_path}")
        
        if not os.path.exists(test_path):
            self.logger.warning(f"Test data path not found: {test_path}")
            self.logger.warning("Using training data for testing (not recommended)")
            test_path = train_path
        
        # Create datasets
        train_data = ImageFolder(root=train_path, transform=data_transform)
        test_data = ImageFolder(root=test_path, transform=data_transform)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )
        
        self.test_loader = DataLoader(
            test_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        self.logger.info(f"Training samples: {len(train_data)}")
        self.logger.info(f"Test samples: {len(test_data)}")
        
        return train_data, test_data
        
    def build_model(self):
        """Build INP-Former model for print quality detection"""
        self.logger.info("Building model...")
        
        # Load vision transformer encoder
        encoder = vit_encoder.load(self.args.encoder)
        self.logger.info(f"Loaded encoder: {self.args.encoder}")
        
        # Get embedding dimensions based on encoder size
        if 'small' in self.args.encoder:
            embed_dim, num_heads = 384, 6
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        elif 'base' in self.args.encoder:
            embed_dim, num_heads = 768, 12
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        elif 'large' in self.args.encoder:
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
        
        # Intrinsic Normal Prototypes (INP)
        inp_prototypes = nn.ParameterList([
            nn.Parameter(torch.randn(self.args.inp_num, embed_dim))
            for _ in range(1)
        ])
        
        # Build full model
        model = INP_Former(
            encoder,
            target_layers,
            bottleneck,
            inp_extractor,
            inp_guided_decoder,
            inp_prototypes,
            args=self.args
        ).to(self.device)
        
        self.logger.info(f"Model created with embedding dim: {embed_dim}")
        
        return model, target_layers
        
    def build_optimizer(self, model):
        """Build optimizer and scheduler"""
        self.logger.info("Building optimizer...")
        
        optimizer = StableAdamW(
            model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = WarmCosineScheduler(
            optimizer,
            warmup_epochs=self.args.warmup_epochs,
            total_epochs=self.args.epochs,
            steps_per_epoch=len(self.train_loader)
        )
        
        return optimizer, scheduler
        
    def train_epoch(self, model, optimizer, scheduler, epoch):
        """Train one epoch"""
        model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            loss = model(images)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        self.logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        return avg_loss
        
    def validate(self, model, epoch):
        """Validate model on test data"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, _ in tqdm(self.test_loader, desc="Validating"):
                images = images.to(self.device)
                loss = model(images)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        self.logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_loss:.4f}")
        return avg_loss
        
    def save_checkpoint(self, model, epoch, loss):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.args.output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'model_epoch_{epoch:03d}_loss_{loss:.4f}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Keep only last 3 checkpoints
        checkpoints = sorted(checkpoint_dir.glob('model_epoch_*.pth'))
        if len(checkpoints) > 3:
            old_checkpoint = checkpoints[0]
            old_checkpoint.unlink()
            self.logger.info(f"Removed old checkpoint: {old_checkpoint.name}")
        
    def train(self):
        """Main training loop"""
        self.logger.info("=" * 60)
        self.logger.info("Starting Print Quality Detection Training")
        self.logger.info("=" * 60)
        
        # Prepare data
        train_data, test_data = self.prepare_data()
        
        # Build model
        model, target_layers = self.build_model()
        
        # Build optimizer
        optimizer, scheduler = self.build_optimizer(model)
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            # Train
            train_loss = self.train_epoch(model, optimizer, scheduler, epoch)
            
            # Validate
            val_loss = self.validate(model, epoch)
            
            # Save checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(model, epoch, val_loss)
                self.logger.info(f"Best model updated! Loss: {best_loss:.4f}")
        
        self.logger.info("=" * 60)
        self.logger.info("Training completed successfully!")
        self.logger.info(f"Best validation loss: {best_loss:.4f}")
        self.logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train INP-Former for print quality detection')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./print_quality_data',
                        help='Path to dataset (should have train/ and test/ folders)')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Crop size for data augmentation')
    
    # Model arguments
    parser.add_argument('--encoder', type=str, default='dino_vits14',
                        choices=['dino_vits14', 'dino_base14', 'dino_large14'],
                        help='Vision transformer encoder')
    parser.add_argument('--inp_num', type=int, default=20,
                        help='Number of intrinsic normal prototypes')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./print_quality_output',
                        help='Output directory for checkpoints and logs')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = PrintQualityTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
