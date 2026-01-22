#!/bin/bash
# Quick reference for print quality detection workflow

PROJECT_DIR="/home/gabap/Project/INP-Former-OCV"
PYTHON="$PROJECT_DIR/.venv/bin/python"

echo "=========================================="
echo "INP-Former Print Quality Detection"
echo "Quick Reference"
echo "=========================================="
echo ""

# Function to show status
show_status() {
    echo "📊 Current Setup Status:"
    echo "  Project: $PROJECT_DIR"
    echo "  Python: $(${PYTHON} --version)"
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  Data files: $(find $PROJECT_DIR/print_quality_data -type f 2>/dev/null | wc -l)"
    echo "  Checkpoints: $(find $PROJECT_DIR/print_quality_output -name '*.pth' 2>/dev/null | wc -l)"
    echo ""
}

# Function to show training command
show_train() {
    echo "🚀 Training Commands:"
    echo ""
    echo "1. Quick training (30 epochs):"
    echo "   cd $PROJECT_DIR && ${PYTHON} train_print_quality.py --epochs 30 --batch_size 8"
    echo ""
    echo "2. Full training (50 epochs):"
    echo "   cd $PROJECT_DIR && ${PYTHON} train_print_quality.py --epochs 50 --batch_size 8"
    echo ""
    echo "3. Fast training (10 epochs, for testing):"
    echo "   cd $PROJECT_DIR && ${PYTHON} train_print_quality.py --epochs 10 --batch_size 4"
    echo ""
}

# Function to show inference command
show_inference() {
    echo "🔍 Inference Commands:"
    echo ""
    
    # Get latest checkpoint
    LATEST=$(ls -t $PROJECT_DIR/print_quality_output/checkpoints/model_*.pth 2>/dev/null | head -1)
    
    if [ -z "$LATEST" ]; then
        echo "⚠️  No checkpoint found. Please train first!"
        echo ""
        return
    fi
    
    CHECKPOINT_NAME=$(basename "$LATEST")
    echo "Latest checkpoint: $CHECKPOINT_NAME"
    echo ""
    
    echo "1. Single image:"
    echo "   cd $PROJECT_DIR && ${PYTHON} inference_print_quality.py \\"
    echo "     --model_path '$LATEST' \\"
    echo "     --image_path ./path/to/image.jpg"
    echo ""
    
    echo "2. Batch test (with JSON output):"
    echo "   cd $PROJECT_DIR && ${PYTHON} inference_print_quality.py \\"
    echo "     --model_path '$LATEST' \\"
    echo "     --image_path ./print_quality_data/test/good_print/ \\"
    echo "     --batch_mode --output_file results.json"
    echo ""
    
    echo "3. View results:"
    echo "   cat $PROJECT_DIR/results.json"
    echo ""
}

# Function to show data prep
show_data() {
    echo "📁 Data Preparation:"
    echo ""
    echo "1. Create synthetic samples (already done):"
    echo "   cd $PROJECT_DIR && ${PYTHON} prepare_print_quality_data.py --create_samples --num_samples 50"
    echo ""
    echo "2. Prepare from your images:"
    echo "   cd $PROJECT_DIR && ${PYTHON} prepare_print_quality_data.py \\"
    echo "     --source_dir /path/to/your/print/images \\"
    echo "     --train_ratio 0.8"
    echo ""
}

# Function to show monitoring
show_monitor() {
    echo "📈 Monitoring:"
    echo ""
    echo "1. Watch GPU usage (during training):"
    echo "   watch -n 1 nvidia-smi"
    echo ""
    echo "2. View training logs:"
    echo "   tail -f $PROJECT_DIR/print_quality_output/logs/training_*.log"
    echo ""
    echo "3. Check training progress:"
    echo "   ls -lth $PROJECT_DIR/print_quality_output/checkpoints/"
    echo ""
}

# Main menu
if [ "$1" = "status" ]; then
    show_status
elif [ "$1" = "train" ]; then
    show_train
elif [ "$1" = "inference" ]; then
    show_inference
elif [ "$1" = "data" ]; then
    show_data
elif [ "$1" = "monitor" ]; then
    show_monitor
elif [ "$1" = "all" ]; then
    show_status
    show_train
    show_inference
    show_data
    show_monitor
else
    echo "Usage: $0 [status|train|inference|data|monitor|all]"
    echo ""
    show_status
    show_train
    show_inference
fi

echo "=========================================="
echo "📖 For more help, see:"
echo "   - README_PRINT_QUALITY.md"
echo "   - PRINT_QUALITY_GUIDE.md"
echo "   - SETUP_SUMMARY_VN.md"
echo "=========================================="
