# 跨节点 Ray Worker 部署功能实现

## 新增功能

在 `vllm_exp` 中添加了 `rank_to_node` 配置选项，允许用户精确指定每个 rank 应该部署在哪个 Ray 节点上。

## 配置说明

### YAML 配置示例

```yaml
network:
  delays: [0]
  # KV synchronizer 使用的 IP 映射
  rank_to_ip:
    0: "spartan-gpgpu169"
    1: "spartan-gpgpu003"
  # 新增：指定每个 rank 部署在哪个 Ray 节点
  # 使用 Ray 节点的 IP 地址
  rank_to_node:
    0: "172.26.93.169"    # rank 0 部署在 node 1
    1: "172.26.93.45"     # rank 1 部署在 node 2
```

### 获取 Ray 节点 IP

```bash
# 方法 1: 使用 ray status
ray status --verbose

# 方法 2: 使用 Python
python3 -c "
import ray
ray.init(address='auto')
for node in ray.nodes():
    print(f'IP: {node[\"NodeManagerAddress\"]}, Alive: {node[\"Alive\"]}')
"
```

## 修改的文件

### 1. `vllm_exp/data.py`
- 在 `NetworkCfg` 中添加 `rank_to_node: Dict[int, str]` 字段

### 2. `vllm/engine/arg_utils.py`
- 添加 `ray_rank_to_node` 字段到 `EngineArgs`
- 添加 `--ray-rank-to-node` CLI 参数（接受 JSON 格式字符串）

### 3. `vllm/config.py`
- 在 `ParallelConfig` 中添加 `ray_rank_to_node: Optional[Dict[int, str]]` 字段

### 4. `vllm_exp/experiments.py`
- 在 `start_vllm()` 和 `start_vllm_with_raw_logging()` 中通过 CLI 参数传递：
  - `--ray-rank-to-node`: JSON 格式的 rank->node IP 映射

### 5. `vllm/executor/ray_utils.py`
- 修改 `initialize_ray_cluster()` 函数：
  - 从 `parallel_config.ray_rank_to_node` 读取映射
  - 为每个 rank 创建带有 `node:<IP>` 约束的 bundle
  - 当指定跨节点部署时使用 `STRICT_SPREAD` 策略

## 工作原理

1. 用户在 YAML 配置中指定 `rank_to_node` 映射
2. `vllm_exp` 将此映射通过 `--ray-rank-to-node` CLI 参数传递给 vLLM
3. vLLM 的 `arg_utils.py` 解析参数并存入 `ParallelConfig.ray_rank_to_node`
4. vLLM 启动时，`ray_utils.py` 从 `parallel_config` 读取映射
5. 创建 placement group 时，为每个 rank 的 bundle 添加 `node:<IP>` 资源约束
6. Ray 会将 worker 严格调度到指定的节点上

## 验证方法

运行实验后，检查日志中的 NCCL 输出：

```bash
grep "nNodes" /path/to/server.log
```

期望看到：
```
nRanks 2 nNodes 2 localRanks 1 localRank 0  # 两个节点，每个节点一个 rank
```

而不是：
```
nRanks 2 nNodes 1 localRanks 2 localRank 0  # 单节点，两个 rank
```

## 注意事项

1. **IP 地址格式**: `rank_to_node` 必须使用 Ray 报告的 IP 地址，而不是 hostname
2. **节点可用性**: 指定的节点必须在 Ray 集群中且有可用 GPU
3. **与 rank_to_ip 的区别**:
   - `rank_to_ip`: 供 KV synchronizer 用于跨节点通信
   - `rank_to_node`: 供 Ray 用于 worker 节点调度

---
生成时间: 2026-01-20
