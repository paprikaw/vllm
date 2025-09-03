#!/bin/bash
set -euo pipefail
# 加上一个trap，如果脚本被中断，则打印当前的进程信息
cleanup() {
  echo "[INFO] 脚本被中断或出错，正在清理进程 VLLM..."
  ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs -r kill -9
}
trap cleanup INT ERR

###############################
# ===== 参数配置区域 ========
###############################
# MPS
if [ -z "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE+x}" ]; then
    echo "错误: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 环境变量未设置"
    exit 1
fi

# 延迟配置
DELAY_LIST=(10ms)

# 日志与目录
CONFIG_NAME=${CONFIG_NAME:-"DEBUG"} # 容器的GPU和Memory的配置
PROJECT_NAME=${PROJECT_NAME:-"memory"} # 当前expeirment的purpose
LOG_DIR=${LOG_DIR:-/root/vllm_workbench/logs/$PROJECT_NAME/$CONFIG_NAME}
mkdir -p "$LOG_DIR"

# 模型与数据
MODEL_PATH=${MODEL_PATH:-/root/.cache/huggingface/Qwen2.5-32B-Instruct-AWQ/}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-32B}
DATASET_PATH=${DATASET_PATH:-/root/ShareGPT_V3_unfiltered_cleaned_split.json}

# vLLM 配置
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-5000}
SLEEP_BEFORE_KILL=${SLEEP_BEFORE_KILL:-40}
VLLM_PORT=${VLLM_PORT:-8000}
RAY_PORT=${RAY_PORT:-6379}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-true}
ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH:-false}
ENABLE_NSIGHT=${ENABLE_NSIGHT:-false}

# Benchmark 配置
NUM_REQUESTS=${NUM_REQUESTS:-124}
REQUEST_RATE_LIST=(1.0)
VLLM_PP_LAYER_PARTITION_LIST=("16,48")
export NCCL_DEBUG=INFO
export NCCL_CUMEM_HOST_ENABLE=0
###############################
# ===== 启动服务部分 ========
###############################

# Print 重要的实验参数
echo "[INFO] 重要的实验参数："
echo "--------------------------------"
echo "DATASET_PATH: $DATASET_PATH"
echo "PIPELINE_PARALLEL_SIZE: $PIPELINE_PARALLEL_SIZE"
echo "GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
echo "ENABLE_NSIGHT: $ENABLE_NSIGHT"
echo "NUM_REQUESTS: $NUM_REQUESTS"
echo "REQUEST_RATE_LIST: ${REQUEST_RATE_LIST[@]}"
echo "CHUNKED_PREFILL: $CHUNKED_PREFILL"
echo "ENABLE_CUDA_GRAPH: $ENABLE_CUDA_GRAPH"
echo "--------------------------------"

export VLLM_PP_LAYER_PARTITION="16,48"
echo "[INFO] 正式运行 vLLM, 模型并行层数: $VLLM_PP_LAYER_PARTITION"
CUR_LOG_DIR="$LOG_DIR/${VLLM_PP_LAYER_PARTITION}"
mkdir -p "$CUR_LOG_DIR"
SERVER_LOG_FILE="$CUR_LOG_DIR/vllm-server.log"
# 构造 vLLM serve 参数
SERVE_ARGS="--pipeline-parallel-size $PIPELINE_PARALLEL_SIZE \
            --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
            --max-model-len $MAX_MODEL_LEN \
            --served-model-name $MODEL_NAME \
            --distributed-executor-backend ray \
            --disable-mm-preprocessor-cache \
            --disable-log-requests"
  
[ "$CHUNKED_PREFILL" = "true" ] && SERVE_ARGS="$SERVE_ARGS --enable-chunked-prefill"
[ "$ENABLE_CUDA_GRAPH" = "false" ] && SERVE_ARGS="$SERVE_ARGS --enforce-eager"
[ "$ENABLE_NSIGHT" = "true" ] && SERVE_ARGS="$SERVE_ARGS --ray-workers-use-nsight"

set +e
  vllm serve "$MODEL_PATH" $SERVE_ARGS
set -e
tail -f /dev/null  # 等待模型 load 完成

for REQUEST_RATE in "${REQUEST_RATE_LIST[@]}"; do
  echo "[INFO] 开始 Benchmark 测试, 请求速率: $REQUEST_RATE"
  BENCHMARK_LOG_FILE="$CUR_LOG_DIR/benchmark-${REQUEST_RATE}.log"
  python3 /root/vllm_workbench/vllm/benchmarks/benchmark_serving.py \
      --num-prompts $NUM_REQUESTS \
      --request-rate $REQUEST_RATE \
      --backend openai-chat \
      --model "$MODEL_PATH" \
      --endpoint /v1/chat/completions \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_PATH" \
      --base-url http://head:$VLLM_PORT \
      --served-model-name "$MODEL_NAME" \
      --goodput tpot:300 ttft:5000 \
      --ignore-eos \
      --temperature 0 \
      --seed 42 > "$BENCHMARK_LOG_FILE"
done
ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs -r kill -9
      sleep 10
echo "[INFO] 实验 完成 ✅"