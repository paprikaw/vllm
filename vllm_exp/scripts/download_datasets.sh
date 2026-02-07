#!/bin/bash
# Download benchmark datasets for vllm_exp
# Usage: ./download_datasets.sh [dataset_name] [output_dir]
#   dataset_name: burstgpt, sharegpt (default: burstgpt)
#   output_dir: directory to save dataset (default: /data/datasets)

set -e

DATASET_NAME="${1:-burstgpt}"
OUTPUT_DIR="${2:-/data/datasets}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

case "$DATASET_NAME" in
    burstgpt|burst_gpt|BurstGPT)
        FILENAME="BurstGPT_without_fails_2.csv"
        URL="https://github.com/HPMLL/BurstGPT/releases/download/v1.1/$FILENAME"
        OUTPUT_PATH="$OUTPUT_DIR/$FILENAME"
        
        echo "Downloading BurstGPT dataset..."
        echo "  URL: $URL"
        echo "  Output: $OUTPUT_PATH"
        
        if [ -f "$OUTPUT_PATH" ]; then
            echo "  File already exists, skipping download."
        else
            wget -q --show-progress -O "$OUTPUT_PATH" "$URL"
            echo "  Download complete!"
        fi
        
        echo ""
        echo "To use BurstGPT in your experiment config:"
        echo "  benchmark:"
        echo "    dataset_name: burstgpt"
        echo "    dataset_path: $OUTPUT_PATH"
        ;;
        
    sharegpt|ShareGPT)
        FILENAME="ShareGPT_V3_unfiltered_cleaned_split.json"
        URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/$FILENAME"
        OUTPUT_PATH="$OUTPUT_DIR/$FILENAME"
        
        echo "Downloading ShareGPT dataset..."
        echo "  URL: $URL"
        echo "  Output: $OUTPUT_PATH"
        
        if [ -f "$OUTPUT_PATH" ]; then
            echo "  File already exists, skipping download."
        else
            wget -q --show-progress -O "$OUTPUT_PATH" "$URL"
            echo "  Download complete!"
        fi
        
        echo ""
        echo "To use ShareGPT in your experiment config:"
        echo "  benchmark:"
        echo "    dataset_name: sharegpt"
        echo "    dataset_path: $OUTPUT_PATH"
        ;;
        
    *)
        echo "Unknown dataset: $DATASET_NAME"
        echo ""
        echo "Supported datasets:"
        echo "  burstgpt  - BurstGPT real-world LLM serving traces"
        echo "  sharegpt  - ShareGPT conversation dataset"
        echo ""
        echo "Usage: $0 [dataset_name] [output_dir]"
        echo "  dataset_name: burstgpt, sharegpt (default: burstgpt)"
        echo "  output_dir: directory to save dataset (default: /data/datasets)"
        exit 1
        ;;
esac

echo ""
echo "Done! Dataset is ready at: $OUTPUT_PATH"
