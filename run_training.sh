#!/bin/bash
# Training script for print quality detection

echo "=========================================="
echo "Starting INP-Former Training"
echo "=========================================="
echo ""

# Set Python environment
PYTHON="/home/gabap/Project/INP-Former-OCV/.venv/bin/python"

# Training parameters
DATA_PATH="./print_quality_data"
OUTPUT_DIR="./print_quality_output"
EPOCHS=30
BATCH_SIZE=8
LEARNING_RATE=0.0001
ENCODER="dino_vits14"

echo "Configuration:"
echo "  Data Path: $DATA_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Encoder: $ENCODER"
echo ""
echo "Starting training at $(date)"
echo ""

# Run training
$PYTHON train_print_quality.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --encoder "$ENCODER" \
    --input_size 224 \
    --crop_size 224

echo ""
echo "Training completed at $(date)"
echo ""
echo "Next step: Run inference with the trained model"
echo "  $PYTHON inference_print_quality.py \\"
echo "    --model_path $OUTPUT_DIR/checkpoints/model_epoch_*.pth \\"
echo "    --image_path $DATA_PATH/test/good_print/ \\"
echo "    --batch_mode"
