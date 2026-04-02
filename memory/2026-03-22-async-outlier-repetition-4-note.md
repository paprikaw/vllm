# Async 离群优值实验（Repetition 4）补充结论

## 结论
在 async 模式下，`Repetition 4` 是一个很关键的反例：
- 总 migration time 最长：`22.5s`
- 但用户侧性能最好：`TTFT 1156ms`、`TPOT 309.84ms`、`Throughput 1.67 req/s`

这说明 **总 migration time 不是决定 QoS 的主指标**。

## 日志对应
来自 async 运行：
- `after weight loading, time taken: 16.7s`
- `wait for kv patch preparation take 17.1s`
- `receive kv tensor finished, time taken 169ms`
- `bind kv cache time taken: 815ms`
- `after listen to kv cache patches, time taken: 653ms`
- `migration process time taken: 22.5s`

## 解释
这次实验的 22.5s 主要被两段“隐藏时间”拉长：
1. target 侧慢 weight loading
2. source 侧长时间等待 receiver patch-ready

但真正影响在线请求的暴露阶段很短：
- patch catch-up / apply 只有 `653ms`

因此：
- headline migration 很长
- exposed stop / slowdown 很短
- QoS 反而最好

## 对优化方向的启示
优化 async 时，不能只盯 `migration process time taken`。
更应该拆成：
1. hidden overlap time
2. exposed switchover / patch catch-up time

当前更值得优先优化的是：
- `after listen to kv cache patches`
- receiver ready 之前的长等待是否会外溢到请求关键路径

而不是简单追求把总 migration time 压短。
