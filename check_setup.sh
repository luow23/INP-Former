#!/bin/bash
# Quick setup and verification script for print quality detection

echo "=========================================="
echo "INP-Former Print Quality Detection Setup"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"
echo ""

# Check CUDA availability
echo "2. Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU found:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
else
    echo "   ⚠️  No GPU found - training will be slow on CPU"
    echo ""
fi

# Check PyTorch
echo "3. Checking PyTorch..."
python -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || echo "   ⚠️  PyTorch not installed"
echo ""

# Check required packages
echo "4. Checking required packages..."
packages=("PIL" "cv2" "numpy" "sklearn" "torch" "torchvision" "timm")
for package in "${packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "   ✓ $package"
    else
        echo "   ✗ $package (missing)"
    fi
done
echo ""

# Check data directory
echo "5. Checking data directories..."
if [ -d "./print_quality_data" ]; then
    echo "   ✓ print_quality_data/ exists"
    train_count=$(find ./print_quality_data/train -type f | wc -l)
    test_count=$(find ./print_quality_data/test -type f | wc -l)
    echo "     - Training images: $train_count"
    echo "     - Test images: $test_count"
else
    echo "   ✗ print_quality_data/ not found"
    echo "     Run: python prepare_print_quality_data.py --create_samples"
fi
echo ""

# Check model files
echo "6. Checking model structure..."
if [ -f "./models/uad.py" ]; then
    echo "   ✓ Model files found"
else
    echo "   ⚠️  Some model files may be missing"
fi
echo ""

# Show available scripts
echo "7. Available scripts:"
echo "   ✓ train_print_quality.py - Training script"
echo "   ✓ inference_print_quality.py - Inference script"
echo "   ✓ prepare_print_quality_data.py - Data preparation"
echo ""

# Quick start instructions
echo "=========================================="
echo "Quick Start Instructions"
echo "=========================================="
echo ""
echo "Step 1: Prepare data (if not done)"
echo "  python prepare_print_quality_data.py --create_samples"
echo ""
echo "Step 2: Start training"
echo "  python train_print_quality.py \\"
echo "    --data_path ./print_quality_data \\"
echo "    --output_dir ./print_quality_output \\"
echo "    --epochs 50 \\"
echo "    --batch_size 4"
echo ""
echo "Step 3: Run inference"
echo "  python inference_print_quality.py \\"
echo "    --model_path ./print_quality_output/checkpoints/model_epoch_*.pth \\"
echo "    --image_path ./print_quality_data/test/good_print/ \\"
echo "    --batch_mode"
echo ""
echo "=========================================="
echo ""
