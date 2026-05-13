# NCCL Grow/Shrink Probe

This directory is an isolated probe for NCCL communicator grow/shrink APIs.
It vendors `nvidia-nccl-cu12==2.30.4` under `vendor/` so the test does not
depend on the Spartan NCCL module version.

Spartan module status observed on this node:

- Available NCCL modules only go up to `NCCL/2.22.3-CUDA-12.4.1`.
- Current vLLM Python/Torch NCCL is `2.26.2`.
- Vendored NCCL is `2.30.4` and exports `ncclCommGrow`,
  `ncclCommShrink`, and `ncclCommGetUniqueId`.

Build:

```bash
cd nccl_grow_shrink_probe
source ./env.sh
./build.sh
```

Run on one node with at least 3 visible GPUs:

```bash
cd nccl_grow_shrink_probe
source ./env.sh
./run_local.sh
```

The demo starts ranks 0 and 1 in an initial communicator, keeps rank 2 outside
that communicator, grows to 3 ranks, runs an all-reduce, shrinks back to ranks
0 and 1, and runs another all-reduce.

Check which NCCL this experiment will use:

```bash
cd nccl_grow_shrink_probe
source ./env.sh
python ./scripts/check_nccl_symbols.py
```

Important API detail observed in NCCL 2.30.4: existing ranks call
`ncclCommGrow(parent, new_world_size, uid_or_null, -1, ...)`; only the new rank
passes its actual rank id.
