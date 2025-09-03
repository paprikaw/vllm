#!/bin/bash
set -euo pipefail
# 加上一个trap，如果脚本被中断，则打印当前的进程信息
cleanup() {
  echo "[INFO] 脚本被中断或出错，正在清理进程 VLLM..."
  # 找到 PID 并用默认 kill（即 SIGTERM）
  ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs -r kill
  if [ "$STATUS" = "RUNNING" ]; then
    echo "[INFO] 脚本被中断或出错，清理当前不完整的benchmark file: $BENCHMARK_LOG_FILE"
    # if [ -f "$BENCHMARK_LOG_FILE" ]; then
    #   rm $BENCHMARK_LOG_FILE
    # fi
  fi
}
trap cleanup INT ERR

IS_COVER=true
STATUS=IDLE


###############################
# ===== 参数配置区域 ========
###############################
# MPS
export TEST_MIGRATION=1
VLLM_PP_LAYER_PARTITION_LIST=("8,56")
REQUEST_RATE_LIST=(2.5)
if [ -z "${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE+x}" ]; then
    echo "错误: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 环境变量未设置"
    exit 1
fi

# 日志与目录
# CONFIG_NAME=${CONFIG_NAME:-"DEBUG"} # 容器的GPU和Memory的配置
PROJECT_NAME=${PROJECT_NAME:-"512,64-512,256"} # 当前expeirment的purpose
LOG_DIR=${LOG_DIR:-/root/vllm_workbench/logs/$PROJECT_NAME/$CONFIG_NAME}
mkdir -p "$LOG_DIR"

export NCCL_DEBUG=INFO
export NCCL_CUMEM_HOST_ENABLE=0
export VLLM_NCCL_SO_PATH="/usr/lib/x86_64-linux-gnu/libnccl.so.2.21.5"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnccl.so.2.21.5"
export VLLM_TORCH_PROFILER_DIR=/root/vllm_workbench/profiler/$PROJECT_NAME/$CONFIG_NAME/
export RAY_CGRAPH_get_timeout=10000
export BENCHMARK_CONFIG_PATH="/root/vllm_workbench/experiments/deployment_configs/benchmark_config.json"
export DEPLOYMENT_CONFIG_PATH="/root/vllm_workbench/experiments/deployment_configs/delay.json"


# 模型与数据
MODEL_PATH=${MODEL_PATH:-/root/.cache/huggingface/Qwen3-32B-AWQ}
MODEL_NAME=${MODEL_NAME:-Qwen3-32B-AWQ}
DATASET_PATH=${DATASET_PATH:-/root/ShareGPT_V3_unfiltered_cleaned_split.json}

# vLLM 配置
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-10000}
SLEEP_BEFORE_KILL=${SLEEP_BEFORE_KILL:-40}
VLLM_PORT=${VLLM_PORT:-8000}
RAY_PORT=${RAY_PORT:-6379}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-true}
ENABLE_CUDA_GRAPH=${ENABLE_CUDA_GRAPH:-false}
ENABLE_NSIGHT=${ENABLE_NSIGHT:-false}
PROFILE=${PROFILE:-false}

# 延迟配置
DELAY_LIST=(0ms)

# Benchmark 配置
NUM_REQUESTS=${NUM_REQUESTS:-300}
export PATTERN_BATCH_SIZE=${PATTERN_BATCH_SIZE:-150}
# VLLM_PP_LAYER_PARTITION_LIST=("24,40")
RANDOM_INPUT_LEN_LIST=("128")
RANDOM_OUTPUT_LEN_LIST=("128")
MIGRATION_INTERVAL_LIST=("20")
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
echo "RANDOM_INPUT_LEN_LIST: ${RANDOM_INPUT_LEN_LIST[@]}"
echo "RANDOM_OUTPUT_LEN_LIST: ${RANDOM_OUTPUT_LEN_LIST[@]}"
echo "LAYER_PARTITION_LIST: ${VLLM_PP_LAYER_PARTITION_LIST[@]}"
echo "--------------------------------"


for DELAY in "${DELAY_LIST[@]}"; do
  export DELAY=$DELAY
  echo "[INFO] 设置网络延迟: $DELAY"
  bash ../tc.sh
  for VLLM_PP_LAYER_PARTITION in "${VLLM_PP_LAYER_PARTITION_LIST[@]}"; do
    for MIGRATION_INTERVAL in "${MIGRATION_INTERVAL_LIST[@]}"; do
      export MIGRATION_INTERVAL=$MIGRATION_INTERVAL
      export VLLM_PP_LAYER_PARTITION=$VLLM_PP_LAYER_PARTITION
      echo "[INFO] 正式运行 vLLM, 模型并行层数: $VLLM_PP_LAYER_PARTITION"
      CUR_LOG_DIR="$LOG_DIR/${DELAY}-${VLLM_PP_LAYER_PARTITION}"
      mkdir -p "$CUR_LOG_DIR"
      SERVER_LOG_FILE="$CUR_LOG_DIR/vllm-server.log"
      # 构造 vLLM serve 参数
      SERVE_ARGS="--pipeline-parallel-size $PIPELINE_PARALLEL_SIZE \
                  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
                  --max-model-len $MAX_MODEL_LEN \
                  --served-model-name $MODEL_NAME \
                  --distributed-executor-backend ray \
                  --disable-log-requests \
                  --no-enable-prefix-caching \
                  --scheduler-cls vllm.v1.core.sched.dynamic_scheduler.DynamicScheduler \
                  --worker-cls vllm.v1.worker.dynamic_gpu_worker.DynamicGPUWorker"

      [ "$CHUNKED_PREFILL" = "true" ] && SERVE_ARGS="$SERVE_ARGS --enable-chunked-prefill"
      [ "$ENABLE_CUDA_GRAPH" = "false" ] && SERVE_ARGS="$SERVE_ARGS --enforce-eager"
      [ "$ENABLE_NSIGHT" = "true" ] && SERVE_ARGS="$SERVE_ARGS --ray-workers-use-nsight"

      vllm serve "$MODEL_PATH" $SERVE_ARGS  > "${SERVER_LOG_FILE}" 2>&1 &
      STATUS=RUNNING
      # 当SERVER_LOG_FILE 中出现 INFO:     Started server process [1040]时，开始测试
      while ! grep -q "INFO:     Started server process" "$SERVER_LOG_FILE"; do
        sleep 3
      done
      for REQUEST_RATE in "${REQUEST_RATE_LIST[@]}"; do
        for RANDOM_INPUT_LEN in "${RANDOM_INPUT_LEN_LIST[@]}"; do
          for RANDOM_OUTPUT_LEN in "${RANDOM_OUTPUT_LEN_LIST[@]}"; do
            INPUT_LEN=$RANDOM_INPUT_LEN
            OUTPUT_LEN=$RANDOM_OUTPUT_LEN
            if [ "$TEST_MIGRATION" = "1" ]; then
              BENCHMARK_LOG_FILE="$CUR_LOG_DIR/benchmark-migration-${REQUEST_RATE}-${INPUT_LEN}-${OUTPUT_LEN}-${MAX_MODEL_LEN}.log"
            else
              BENCHMARK_LOG_FILE="$CUR_LOG_DIR/benchmark-nomigration${REQUEST_RATE}-${INPUT_LEN}-${OUTPUT_LEN}-${MAX_MODEL_LEN}.log"
            fi
            if [ -f "$BENCHMARK_LOG_FILE" ] && [ "$IS_COVER" = "false" ]; then
              echo "[INFO] 结果文件已存在，跳过"
              continue
            fi
            BENCHMARK_ARGS="--num-prompts $NUM_REQUESTS \
                  --request-rate $REQUEST_RATE \
                  --backend openai-chat \
                  --model "$MODEL_PATH" \
                  --endpoint /v1/chat/completions \
                  --base-url http://head:$VLLM_PORT \
                  --dataset-name pattern \
                  --served-model-name "$MODEL_NAME" \
                  --goodput tpot:300 ttft:5000 \
                  --temperature 0 \
                  --seed 42 \
                  --pattern-batch-size $PATTERN_BATCH_SIZE"
                  # --dataset-name sharegpt \
                  # --dataset-path "$DATASET_PATH" \
            [ "$PROFILE" = "true" ] && BENCHMARK_ARGS="$BENCHMARK_ARGS --profile"
            echo "[INFO] 开始 Benchmark 测试, 请求速率: $REQUEST_RATE, Input Length: $INPUT_LEN, Output Length: $OUTPUT_LEN"
            python3 /root/vllm_workbench/vllm/benchmarks/benchmark_serving.py \
            $BENCHMARK_ARGS > "$BENCHMARK_LOG_FILE" 2>&1
            sleep 5
          done
        done
      done
      ps aux | grep vllm | grep -v grep | awk '{print $2}' | xargs -r kill
      sleep 10
    done
  done
done

STATUS=IDLE
if [ "$TEST_MIGRATION" = "1" ]; then
  SERVER_LOG_FILE_SPECIFIC="$CUR_LOG_DIR/server-migration-${REQUEST_RATE}-${INPUT_LEN}-${OUTPUT_LEN}-${MAX_MODEL_LEN}.log"
else
  SERVER_LOG_FILE_SPECIFIC="$CUR_LOG_DIR/server-nomigration${REQUEST_RATE}-${INPUT_LEN}-${OUTPUT_LEN}-${MAX_MODEL_LEN}.log"
fi
cp $SERVER_LOG_FILE $SERVER_LOG_FILE_SPECIFIC
echo "[INFO] Benchmark 完成 ✅"