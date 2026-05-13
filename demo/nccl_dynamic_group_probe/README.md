# NCCL Dynamic Group Probe

This is a minimal probe for measuring whether creating a new NCCL process
group during an active GPU workload perturbs the workload already running on
the existing GPUs.

The important semantic constraint is that NCCL/PyTorch cannot add a completely
new process to an already-created communicator. The extra GPU must already have
a rank in the default world process group. This probe therefore starts all
ranks up front, creates an initial NCCL group that excludes one rank, runs a
GPU-heavy workload on the initial group, then creates a second NCCL group that
includes the previously idle rank.

## What It Measures

- Initial NCCL group creation time.
- Dynamic expanded NCCL group creation time.
- First collective time on the expanded group, which forces NCCL communicator
  use and includes the newly added GPU.
- Per-iteration wall-clock and CUDA-event latency for the active ranks before,
  during, and after dynamic group creation.

The workload is intentionally simple: repeated GEMMs plus optional NCCL
collectives on the initial group. This keeps the test small while exercising
GPU compute, CUDA streams, and NCCL communicator activity.

## Single-Node A100 Run

From the repository root:

```bash
cd nccl_dynamic_group_probe
./scripts/run_local_a100.sh
```

By default this launches 4 ranks on the current node, uses ranks `0,1,2` as the
initial group, and dynamically creates an expanded group containing rank `3`.

Useful overrides:

```bash
MATRIX_SIZE=8192 TOTAL_SEC=30 TRIGGER_SEC=10 ./scripts/run_local_a100.sh
```

Results are written under `nccl_dynamic_group_probe/results/`.

## Two-Node SSH Run

The two-node script assumes the shared project filesystem is visible on both
nodes and uses SSH to start one `torchrun` worker group per node:

```bash
cd nccl_dynamic_group_probe
./scripts/run_ssh_two_nodes.sh
```

Defaults:

- `NODE0=spartan-gpgpu066`
- `NODE1=spartan-gpgpu007`
- `GPUS_PER_NODE=4`
- initial group size = world size - 1
- added rank = last global rank

Useful overrides:

```bash
NODE0=spartan-gpgpu066 NODE1=spartan-gpgpu007 \
NCCL_SOCKET_IFNAME=bond0.3027 \
MATRIX_SIZE=6144 TOTAL_SEC=30 TRIGGER_SEC=10 \
./scripts/run_ssh_two_nodes.sh
```

For mixed A100/L40S nodes, compare latency per rank rather than averaging
blindly across devices.

On Spartan, the hostnames observed in this session resolve to the
`bond0.3027` network (`172.26.92.181` and `172.26.93.49`). If cross-node NCCL
hangs at the first barrier, set `NCCL_SOCKET_IFNAME=bond0.3027`.

## Output Files

Each run directory contains:

- `rank_*.jsonl`: raw per-rank events and iteration timings.
- `summary.json`: summarized timing by rank and phase.
- `node*.log`: stdout/stderr for multi-node SSH runs.

The most useful fields in `summary.json` are:

- `events.dynamic_new_group_sec`
- `events.expanded_first_all_reduce_sec`
- `iteration_stats.*.before.wall_ms`
- `iteration_stats.*.during.wall_ms`
- `iteration_stats.*.after.wall_ms`

If `during.max` or `during.p95` spikes relative to `before`, the dynamic NCCL
group creation or first expanded collective likely disturbed the active
workload.

## Async Switch Demo

`async_group_switch_demo.py` is a higher-level demo for the reconfiguration
pattern:

1. Start all ranks in the default world.
2. Create an initial NCCL group that excludes the last rank.
3. Active ranks continue compute plus NCCL communication on the old group.
4. A background thread creates the expanded NCCL group.
5. At an iteration boundary, all ranks enter a safe point.
6. Ranks validate the expanded group, install new local runtime state, and
   continue on the expanded group.

Single-node run:

```bash
./scripts/run_async_switch_local.sh
```

Two-node SSH run:

```bash
./scripts/run_async_switch_ssh_two_nodes.sh
```

The important summary fields are:

- `async_expanded_group_create_sec`: background group creation time.
- `safe_point_pause_sec`: measured pause from entering the safe point until
  new state is installed.
- `expanded_first_collective_sec`: first collective on the expanded group.

This still requires the added GPU rank to be launched up front in the default
world. The demo minimizes the switch pause by moving `new_group` creation
before the safe point, but the final state transition remains coordinated.

## Dynamic World Reinit Demo

`dynamic_world_reinit_demo.py` demonstrates the stronger pattern where the new
GPU process does not exist at initial launch:

1. Start an old default world with 3 ranks on GPUs `0,1,2`.
2. Keep GPU `3` unused, with no Python worker process.
3. Rank 0 dynamically spawns a new Python process on GPU `3`.
4. The new process initializes CUDA and waits for the reconfiguration signal.
5. Old ranks enter a safe point, destroy the old default process group, and all
   four ranks initialize a new default world.
6. All four ranks install version-1 runtime state and continue collectives.

Run it on a 4-GPU node:

```bash
./scripts/run_dynamic_world_reinit_local.sh
```

The local script uses a `file://` rendezvous for the rebuilt world by default.
Set `NEW_RDZV_BACKEND=tcp` to test TCP rendezvous instead.

Important summary fields:

- `new_process_spawn_to_ready_sec`: launch plus CUDA prewarm time for the fresh
  process.
- `new_world_init_sec`: time to initialize the new `world_size=4` process
  group.
- `safe_point_pause_sec`: old-world stop to new-world state installed.
- `new_world_first_collective_sec`: first all-reduce on the rebuilt world.

Unlike `async_group_switch_demo.py`, this does not reserve a distributed rank
for the added GPU. It still requires an unused GPU to be available, and the old
ranks must pause at a safe point because the default world is destroyed and
rebuilt.
