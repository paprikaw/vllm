#!/bin/bash
# 三测试对比脚本：A100单节点 vs L40单节点 vs A100→L40跨节点
# 用于验证 GPU 异构是否导致 intermediate states 不兼容

set -e

VENV="/data/gpfs/projects/punim2715/vllm_workbench/.venv/bin/python"
VLLM_DIR="/home/bxb1/vllm_workbench/vllm"
LOG_DIR="/home/bxb1/vllm_workbench/vllm/logs/agent"
CONFIG_DIR="/home/bxb1/vllm_workbench/vllm/vllm_exp/configs"

# L40 节点信息
L40_NODE="spartan-gpgpu003"

echo "=============================================="
echo "GPU 异构测试脚本"
echo "=============================================="
echo ""
echo "测试计划："
echo "  1. A100 单节点 (spartan-gpgpu169): 验证 A100 基准"
echo "  2. L40 单节点 (spartan-gpgpu003): 验证 L40 是否能正常工作 [需要SSH]"  
echo "  3. A100→L40 跨节点: 测试 GPU 异构通信"
echo ""
echo "日志目录: $LOG_DIR"
echo "=============================================="
echo ""

# 函数：运行本地测试
run_local_test() {
    local test_name=$1
    local config_file=$2
    
    echo ""
    echo "======================================"
    echo "开始测试: $test_name (本地)"
    echo "配置文件: $config_file"
    echo "时间: $(date)"
    echo "======================================"
    
    $VENV -m vllm_exp.run \
        --config "$config_file" \
        --log-dir "$LOG_DIR" \
        2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ $test_name 完成 (exit code: $exit_code)"
    else
        echo ""
        echo "✗ $test_name 失败 (exit code: $exit_code)"
    fi
    
    echo "等待 10 秒确保日志写入完成..."
    sleep 10
    
    return $exit_code
}

# 函数：在远程节点运行测试 (L40)
run_remote_test() {
    local test_name=$1
    local config_file=$2
    local remote_node=$3
    
    echo ""
    echo "======================================"
    echo "开始测试: $test_name (远程: $remote_node)"
    echo "配置文件: $config_file"
    echo "时间: $(date)"
    echo "======================================"
    
    # SSH 到远程节点运行测试
    ssh "$remote_node" "cd $VLLM_DIR && $VENV -m vllm_exp.run --config $config_file --log-dir $LOG_DIR" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ $test_name 完成 (exit code: $exit_code)"
    else
        echo ""
        echo "✗ $test_name 失败 (exit code: $exit_code)"
    fi
    
    echo "等待 10 秒确保日志写入完成..."
    sleep 10
    
    return $exit_code
}

# 检查 Ray 集群状态
echo "检查 Ray 集群状态..."
ray status
echo ""

# 询问用户要运行哪些测试
echo "请选择要运行的测试 (可多选，用空格分隔):"
echo "  1 - A100 单节点 (debug_single_node.yaml) [本地]"
echo "  2 - L40 单节点 (debug_single_node_l40.yaml) [SSH到$L40_NODE]"
echo "  3 - A100→L40 跨节点 (debug_cross_node.yaml) [本地]"
echo "  a - 运行所有测试"
echo "  q - 退出"
echo ""
read -p "请输入选择 [1/2/3/a/q]: " choice

case $choice in
    1)
        run_local_test "A100 单节点" "$CONFIG_DIR/debug_single_node.yaml"
        ;;
    2)
        run_remote_test "L40 单节点" "$CONFIG_DIR/debug_single_node_l40.yaml" "$L40_NODE"
        ;;
    3)
        run_local_test "A100→L40 跨节点" "$CONFIG_DIR/debug_cross_node.yaml"
        ;;
    a|A)
        echo ""
        echo "将依次运行所有三个测试..."
        echo ""
        
        # 测试 1: A100 单节点 (本地)
        run_local_test "A100 单节点" "$CONFIG_DIR/debug_single_node.yaml"
        
        # 测试 2: L40 单节点 (SSH到L40节点)
        run_remote_test "L40 单节点" "$CONFIG_DIR/debug_single_node_l40.yaml" "$L40_NODE"
        
        # 测试 3: 跨节点 (本地)
        run_local_test "A100→L40 跨节点" "$CONFIG_DIR/debug_cross_node.yaml"
        
        echo ""
        echo "=============================================="
        echo "所有测试完成！"
        echo "=============================================="
        echo ""
        echo "日志文件位置:"
        echo "  A100 单节点: $LOG_DIR/project-debug_single_node_a100/"
        echo "  L40 单节点:  $LOG_DIR/project-debug_single_node_l40/"
        echo "  跨节点:      $LOG_DIR/project-debug_cross_node/"
        echo ""
        echo "分析命令示例:"
        echo "  grep 'PP_SAMPLE_INPUT\\|PP_LM_HEAD\\|PP_SAMPLE.*status' <log_file> | head -100"
        ;;
    q|Q)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择: $choice"
        exit 1
        ;;
esac

echo ""
echo "完成时间: $(date)"
