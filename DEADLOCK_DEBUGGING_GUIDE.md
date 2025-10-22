# 死锁调试指南

## 使用方法

### 方法1：手动触发栈追踪（推荐）

当程序疑似卡死时，可以通过发送信号来打印所有线程的栈：

```bash
# 1. 找到进程PID
ps aux | grep vllm

# 2. 发送 SIGUSR1 信号触发栈转储
kill -SIGUSR1 <PID>

# 3. 查看日志输出，所有线程的栈会打印在日志中
```

日志会输出类似：
```
🔍 DUMPING ALL THREAD STACKS (for deadlock debugging)
📌 Thread: MainThread (ID: 139876543210)
  File: /path/to/file.py:123
    in function_name
    some_code_line
...
```

### 方法2：使用超时检测上下文

在关键代码段使用 `DeadlockTimeoutContext` 自动检测死锁：

```python
from vllm.v1.worker.dynamic_gpu_worker import DeadlockTimeoutContext

# 替换原来的 with 语句
# 原来：
# with self.model_runner.forward_lock:
#     do_something()

# 改为：
with DeadlockTimeoutContext(self.model_runner.forward_lock, 
                            "forward_lock", 
                            timeout=30.0):
    do_something()
```

如果锁在 30 秒内无法获取，会自动：
1. 打印所有线程的栈
2. 抛出 TimeoutError 异常

### 方法3：使用 faulthandler 自动超时检测

取消 `__init__` 中的注释，启用全局超时检测：

```python
# 在 DynamicGPUWorker.__init__ 中
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)  # 60秒无响应则打印栈
```

### 方法4：使用 py-spy 外部工具

安装并使用 py-spy（无需修改代码）：

```bash
# 安装
pip install py-spy

# 实时查看栈
sudo py-spy top --pid <PID>

# 生成火焰图
sudo py-spy record -o profile.svg --pid <PID> --duration 30
```

## 常见死锁模式

### 模式1：锁顺序不一致

```python
# 线程A
with lock1:
    with lock2:
        do_something()

# 线程B  
with lock2:
    with lock1:
        do_something()
```

**解决方案**：统一锁的获取顺序

### 模式2：在持有锁时 wait()

```python
with lock_a:
    with condition_b:
        condition_b.wait()  # 错误：可能导致其他线程无法获取 lock_a
```

**解决方案**：分离锁的使用，先释放外层锁再 wait()

### 模式3：忘记释放锁

```python
lock.acquire()
if error:
    return  # 错误：忘记释放锁
lock.release()
```

**解决方案**：始终使用 with 语句

## 调试技巧

1. **添加详细日志**：在获取/释放锁前后打印日志
2. **使用锁超时**：`lock.acquire(timeout=10)`
3. **可视化线程状态**：使用 `threading.enumerate()` 查看所有活动线程
4. **减少锁的作用域**：只在必要时持有锁，尽快释放

## 环境变量

```bash
# 启用调试模式
export VLLM_DEBUG_RAISE=1
export VLLM_DEBUG_ASSERT_KV=1
```

