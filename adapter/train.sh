#!/usr/bin/env bash
# Train an Engram FoundationModels adapter using Apple's adapter training toolkit.
#
# Prerequisites:
#   1. Download adapter training toolkit from Apple Developer
#   2. Install requirements: pip install -r <toolkit>/requirements.txt
#   3. Generate training data: python generate_data.py
#
# Usage:
#   ./train.sh <path-to-toolkit>
#
# Example:
#   ./train.sh ~/Downloads/adapter_training_toolkit_v26_0_0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLKIT_DIR="${1:?Usage: $0 <path-to-adapter-training-toolkit>}"

# Validate toolkit path
if [ ! -f "$TOOLKIT_DIR/examples/train_adapter.py" ]; then
    echo "Error: adapter training toolkit not found at $TOOLKIT_DIR"
    echo "Expected: $TOOLKIT_DIR/examples/train_adapter.py"
    exit 1
fi

# Configuration
TRAIN_DATA="$SCRIPT_DIR/engram_train.jsonl"
VALID_DATA="$SCRIPT_DIR/engram_valid.jsonl"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
OUTPUT_DIR="$SCRIPT_DIR/output"
ADAPTER_NAME="engram_memory"

EPOCHS=5
LEARNING_RATE=1e-4
BATCH_SIZE=2
MAX_SEQ_LENGTH=2048

# Step 1: Generate training data if not present
if [ ! -f "$TRAIN_DATA" ]; then
    echo "==> Generating training data..."
    python "$SCRIPT_DIR/generate_data.py" --output-dir "$SCRIPT_DIR"
fi

echo "==> Training data: $(wc -l < "$TRAIN_DATA") samples"
echo "==> Validation data: $(wc -l < "$VALID_DATA") samples"

# Step 2: Train adapter
echo ""
echo "==> Training adapter (epochs=$EPOCHS, lr=$LEARNING_RATE, batch=$BATCH_SIZE)..."
mkdir -p "$CHECKPOINT_DIR"

cd "$TOOLKIT_DIR"
python -m examples.train_adapter \
    --train-data "$TRAIN_DATA" \
    --eval-data "$VALID_DATA" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --batch-size "$BATCH_SIZE" \
    --max-sequence-length "$MAX_SEQ_LENGTH" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --checkpoint-frequency 1 \
    --precision bf16-mixed \
    --warmup-epochs 1

# Step 3: Find best checkpoint (lowest validation loss)
BEST_CKPT=$(ls -1 "$CHECKPOINT_DIR"/adapter_epoch_*.pt 2>/dev/null | tail -1)
if [ -z "$BEST_CKPT" ]; then
    echo "Error: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi
echo ""
echo "==> Best checkpoint: $BEST_CKPT"

# Step 4: Export to .fmadapter
echo ""
echo "==> Exporting .fmadapter..."
mkdir -p "$OUTPUT_DIR"

python -m export.export_fmadapter \
    --output-dir "$OUTPUT_DIR" \
    --adapter-name "$ADAPTER_NAME" \
    --checkpoint "$BEST_CKPT" \
    --author "Engram" \
    --description "Engram memory tool-calling adapter for Apple FoundationModels"

ADAPTER_PATH="$OUTPUT_DIR/${ADAPTER_NAME}.fmadapter"
echo ""
echo "==> Adapter exported: $ADAPTER_PATH"
echo ""
echo "To use in your app:"
echo '  let adapter = try SystemLanguageModel.Adapter(fileURL: URL(filePath: "'"$ADAPTER_PATH"'"))'
echo '  let session = LanguageModelSession(model: .default, adapter: adapter, tools: EngramTools.all(memoryTools: mt))'
