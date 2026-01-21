# NCCL Network Interface Issue Report

**Date**: 2026-01-20  
**Status**: Analysis Complete  

## Problem Summary

When running multi-node vLLM with `ray_rank_to_node` configuration, NCCL initialization fails with:
```
RuntimeError: NCCL error: unhandled system error (run with NCCL_DEBUG=INFO for details)
```

## Root Cause Analysis

### Network Configuration
The current node (`spartan-gpgpu169`) has multiple network interfaces:
```
172.26.113.39/23 eno8303       <- NCCL incorrectly selected this
172.26.110.38/22 bond0
172.26.93.169/22 bond0.3027    <- Ray is using this IP
```

### The Problem
1. **Ray placement group** is configured with IPs from the `172.26.93.x` subnet
   - Rank 0: `172.26.93.169` (spartan-gpgpu169)
   - Rank 1: `172.26.93.45` (spartan-gpgpu003)

2. **NCCL bootstrap** automatically selects the wrong network interface:
   ```
   Bootstrap : Using eno8303:172.26.113.39<0>
   ```

3. Since the two nodes cannot communicate through the `172.26.113.x` subnet, NCCL initialization fails.

### Evidence from Logs
```log
INFO 01-20 16:20:14 [ray_utils.py:353] Using user-specified rank-to-node mapping from config: {0: '172.26.93.169', 1: '172.26.93.45'}
...
spartan-gpgpu169:123683:123683 [0] NCCL INFO Bootstrap : Using eno8303:172.26.113.39<0>
...
RuntimeError: NCCL error: unhandled system error
```

## Solutions

### Solution 1: Set Environment Variables (Recommended Quick Fix)

Add the following to your experiment config file (`migration_test_A100.yaml`):

```yaml
envs:
  # ... existing envs ...
  NCCL_SOCKET_IFNAME: "bond0.3027"
  VLLM_HOST_IP: "172.26.93.169"
  GLOO_SOCKET_IFNAME: "bond0.3027"  # Also recommended for PyTorch distributed
```

**Important**: Each node needs its own `VLLM_HOST_IP`. For multi-node setups, you may need to set this per-node or use a helper script.

### Solution 2: Modify Code to Auto-detect Network Interface

We can enhance the `dynamic_ray_distributed_executor.py` to automatically determine the correct network interface based on the Ray node IP.

**Location**: [dynamic_ray_distributed_executor.py](vllm/v1/executor/dynamic_ray_distributed_executor.py#L200-L230)

Add a helper function to detect the correct network interface:

```python
import subprocess

def get_network_interface_for_ip(target_ip: str) -> Optional[str]:
    """Find the network interface that has the given IP address."""
    try:
        result = subprocess.run(
            ["ip", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=5
        )
        lines = result.stdout.split('\n')
        current_interface = None
        for line in lines:
            if ': ' in line and not line.startswith(' '):
                # Line like "3: bond0.3027: <BROADCAST..."
                parts = line.split(': ')
                if len(parts) >= 2:
                    current_interface = parts[1].split('@')[0]
            if f'inet {target_ip}/' in line and current_interface:
                return current_interface
    except Exception as e:
        logger.warning(f"Failed to detect network interface for IP {target_ip}: {e}")
    return None
```

Then in `_init_workers_ray`, after setting environment variables:

```python
# Auto-detect and set NCCL_SOCKET_IFNAME if not already set
if 'NCCL_SOCKET_IFNAME' not in os.environ:
    ifname = get_network_interface_for_ip(driver_ip)
    if ifname:
        logger.info(f"Auto-detected network interface for NCCL: {ifname}")
        for args in all_args_to_update_environment_variables:
            args['NCCL_SOCKET_IFNAME'] = ifname
            args['GLOO_SOCKET_IFNAME'] = ifname
        os.environ['NCCL_SOCKET_IFNAME'] = ifname
        os.environ['GLOO_SOCKET_IFNAME'] = ifname
```

### Solution 3: Use Network Interface Name Pattern

If your cluster consistently uses the same interface naming (e.g., `bond0.*` for cross-node traffic):

```yaml
envs:
  NCCL_SOCKET_IFNAME: "bond0"  # Matches bond0, bond0.3027, etc.
```

## Additional Recommendations

1. **Verify network connectivity** between nodes using the correct interface:
   ```bash
   ping -I bond0.3027 172.26.93.45
   ```

2. **Test NCCL directly** before running vLLM:
   ```bash
   export NCCL_DEBUG=TRACE
   export NCCL_SOCKET_IFNAME=bond0.3027
   # Run a simple torch distributed test
   ```

3. **Secondary bug fix**: The error handling in `dynamic_core.py` has an `UnboundLocalError`. This should be fixed to properly handle exceptions:

   **Location**: [dynamic_core.py](vllm/v1/engine/dynamic_core.py#L1188-L1203)
   
   ```python
   # Current code has:
   #   if engine_core is None:  # UnboundLocalError if exception occurred
   
   # Should be:
   engine_core = None  # Initialize before try block
   try:
       engine_core = DynamicEngineCoreProc(*args, **kwargs)
       ...
   except Exception as e:
       ...
   ```

## Testing

After applying the fix, run:
```bash
python3 -m vllm_exp.run \
  --config /home/bxb1/data/vllm_workbench/vllm/vllm_exp/configs/migration_test_A100.yaml \
  --log-dir /home/bxb1/data/vllm_workbench/vllm/logs/agent/
```

Expected: NCCL should now initialize correctly using the `bond0.3027` interface.
