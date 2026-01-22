#!/bin/bash
# Single Request Trace Experiment Script
# This script runs a single request through three different GPU configurations
# to trace hidden states at each step

set -e

VLLM_DIR="/home/bxb1/vllm_workbench/vllm"
LOG_DIR="$VLLM_DIR/logs/agent/single_req_trace"
PYTHON="/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python"

# Clean previous logs
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# Export environment variables
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo "Single Request Trace Experiment"
echo "=============================================="
echo "Log directory: $LOG_DIR"
echo ""

run_test() {
    local test_name=$1
    local config_file=$2
    local node=$3
    
    echo "=============================================="
    echo "Running: $test_name"
    echo "Config: $config_file"
    echo "Node: $node"
    echo "=============================================="
    
    if [ "$node" == "local" ]; then
        # Run locally (A100 or cross-node from A100)
        cd "$VLLM_DIR"
        $PYTHON -m vllm_exp.run \
            --config "$config_file" \
            --log-dir "$LOG_DIR/$test_name/"
    else
        # Run on remote node via SSH (L40)
        echo "Running on remote node: $node"
        ssh $node "cd $VLLM_DIR && \
            export RAY_DEDUP_LOGS=0 && \
            export CUDA_VISIBLE_DEVICES=0 && \
            $PYTHON -m vllm_exp.run \
                --config $config_file \
                --log-dir $LOG_DIR/$test_name/"
    fi
    
    echo ""
    echo "$test_name completed!"
    echo ""
}

# Menu for selecting which test to run
echo "Select test to run:"
echo "1) A100 Single Node (PP=2 on spartan-gpgpu169)"
echo "2) L40 Single Node (PP=2 on spartan-gpgpu003) [requires SSH]"
echo "3) Cross Node (A100 -> L40)"
echo "4) Run ALL tests"
echo "5) Exit"
echo ""
read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        run_test "a100_single" "$VLLM_DIR/vllm_exp/configs/single_req_trace_a100.yaml" "local"
        ;;
    2)
        run_test "l40_single" "$VLLM_DIR/vllm_exp/configs/single_req_trace_l40.yaml" "spartan-gpgpu003"
        ;;
    3)
        run_test "cross_node" "$VLLM_DIR/vllm_exp/configs/single_req_trace_cross.yaml" "spartan-gpgpu003"
        ;;
    4)
        echo "Running all tests sequentially..."
        run_test "a100_single" "$VLLM_DIR/vllm_exp/configs/single_req_trace_a100.yaml" "local"
        sleep 5
        run_test "l40_single" "$VLLM_DIR/vllm_exp/configs/single_req_trace_l40.yaml" "spartan-gpgpu003"
        sleep 5
        run_test "cross_node" "$VLLM_DIR/vllm_exp/configs/single_req_trace_cross.yaml" "spartan-gpgpu003"
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo "=============================================="
echo "Test completed!"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "To analyze results, run:"
echo "  python $VLLM_DIR/scripts/analyze_single_req_trace.py"
echo "=============================================="
