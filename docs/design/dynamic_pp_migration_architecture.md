---
title: Dynamic Pipeline-Parallel Reconfiguration
---
[](){ #dynamic-pp-migration-architecture }

# Dynamic Pipeline-Parallel Reconfiguration

This document describes an experimental extension to vLLM V1 that makes the
pipeline-parallel layout reconfigurable at runtime. The goal is to support
pipeline-parallel reconfiguration.

In upstream vLLM, the pipeline-parallel layout is effectively fixed for the
lifetime of the engine. This extension turns the layout into runtime state. A
configuration change can move layers between ranks, activate or deactivate
pipeline ranks, and migrate the associated KV state so that in-flight requests
remain consistent across the transition.

## Terminology

This document uses a small set of terms consistently:

- **PP layout**: the runtime mapping from pipeline ranks to layer ranges. Empty
  ranges represent inactive ranks.
- **Layer assignment**: the set of model layers a worker rank should run under a
  given PP layout.
- **KV state transfer**: the movement of KV cache state required when the PP
  layout changes.
- **Migration boundary**: the point where the scheduler commits the target PP
  layout for new batches.


## Architecture Overview

The following diagram highlights the extension boundary. The blue area is the
existing vLLM V1 execution structure. The orange area is the dynamic
reconfiguration control plane. The green area is the runtime data plane needed
to update layer assignment and transfer KV state.

In the prototype implementation, most `Dynamic*` components are implemented as
extensions of their corresponding vLLM V1 components. For example,
`DynamicScheduler` inherits most of the scheduler interface and behavior, then
adds or overrides the parts needed for runtime PP layout changes. The diagram
therefore distinguishes between the upstream component being extended, such as
`V1 Scheduler`, and the dynamic component that carries the experimental
reconfiguration behavior, such as `DynamicScheduler`.

```mermaid
flowchart LR
    subgraph U["Existing vLLM V1 execution structure"]
        direction TB
        Engine["V1 EngineCore"]
        Scheduler["V1 Scheduler"]
        Executor["Ray Distributed Executor"]
        Worker["GPU Worker / ModelRunner"]
        KVCache["KV Cache"]

        Engine --> Scheduler
        Engine --> Executor
        Executor --> Worker
        Worker --> KVCache
    end

    subgraph D["Dynamic PP reconfiguration extension"]
        direction TB
        DynamicConfig["DynamicConfig<br/>PP layout<br/>migration policy"]
        DynamicEngine["DynamicEngineCore<br/>Runtime layout manager"]
        DynamicScheduler["DynamicScheduler<br/>One active PP layout"]
        DynamicOutput["DynamicSchedulerOutput<br/>Batch PP layout"]
        DynamicExecutor["DynamicRayDistributedExecutor<br/>Runtime PP routing<br/>no compiled DAG"]
        DynamicWorker["DynamicGPUWorker<br/>Apply layer assignment<br/>KV state transfer"]
    end

    subgraph R["Runtime migration data plane"]
        direction TB
        DynamicModel["Dynamic model<br/>Layer assignment"]
        KVSync["KV synchronizer<br/>Patch send/apply"]
        PPRouting["Runtime PP topology<br/>Per-batch active ranks"]
    end

    DynamicEngine -. "extends" .-> Engine
    DynamicScheduler -. "extends" .-> Scheduler
    DynamicExecutor -. "extends" .-> Executor
    DynamicWorker -. "extends" .-> Worker

    DynamicConfig --> DynamicEngine
    DynamicEngine --> DynamicScheduler
    DynamicScheduler --> DynamicOutput
    DynamicOutput --> DynamicExecutor
    DynamicExecutor --> DynamicWorker
    DynamicExecutor --> PPRouting
    PPRouting --> DynamicWorker

    DynamicWorker --> DynamicModel
    DynamicWorker --> KVSync
    KVSync --> KVCache

    DynamicWorker -->|"migration progress"| DynamicExecutor
    DynamicExecutor -->|"global migration state"| DynamicEngine
    DynamicEngine -->|"switch active layout at boundary"| DynamicScheduler

    classDef upstream fill:#eef3ff,stroke:#6b7fd7,color:#111;
    classDef extension fill:#fff4e6,stroke:#d9822b,color:#111;
    classDef runtime fill:#eaf7ea,stroke:#4e9a51,color:#111;

    class Engine,Scheduler,Executor,Worker,KVCache upstream;
    class DynamicConfig,DynamicEngine,DynamicScheduler,DynamicOutput,DynamicExecutor,DynamicWorker extension;
    class DynamicModel,KVSync,PPRouting runtime;
```

## Component Responsibilities

`DynamicConfig` is a new configuration component added to `VllmConfig`, rather
than a subclass of an existing execution component. It carries experimental
dynamic settings such as PP layer partitioning, migration configuration paths,
and autoscaling-related knobs.

`DynamicEngineCore` extends the V1 `EngineCore`. It owns the runtime
reconfiguration lifecycle: receiving the target PP layout, deciding when a
migration is needed, coordinating the executor and scheduler, and committing the
target layout once the consistency boundary is reached.

`DynamicScheduler` extends the V1 `Scheduler`. It keeps scheduling requests
under a single active PP layout. During migration it can emit batches that still
belong to the old layout, and it only changes the active layout when the engine
reaches the boundary batch.

`DynamicSchedulerOutput` is a dynamic counterpart to the V1 `SchedulerOutput`.
In the current prototype it is a separate dataclass, not a Python subclass. It
carries the normal scheduler output fields plus the batch-local PP layout and
migration metadata, so workers execute according to the batch-local layout
rather than assuming a single static global layout.

`DynamicRayDistributedExecutor` extends the V1 `RayDistributedExecutor`. It
coordinates distributed workers and runtime PP routing. This path intentionally
does not use a compiled DAG for PP execution: the active PP stages and their
connections can change at runtime, so the executor derives the PP topology from
the batch-local layout instead of relying on a static execution graph.

`DynamicGPUWorker` extends the V1 `Worker`. It performs the worker-local part of
the change: updating layer assignment, participating in KV state transfer,
applying KV patches, and finalizing local state after the migration boundary.

The KV migration path is part of the correctness model, not an optimization.
Changing layer assignment without transferring the corresponding KV state is not
enough to preserve request execution across a PP layout change.

## Migration Sequence

The runtime transition has two important properties:

First, old-layout batches are allowed to continue using the layout they were
scheduled with. Second, the scheduler only starts emitting target-layout batches
after the migration boundary. In the current prototype, the engine decides when
to request that boundary by repeatedly polling all ranks for KV-transfer
progress. Once the gap between the latest generated migration token and the
sender/receiver transfer progress is below a threshold, the engine asks the
scheduler to emit the boundary batch.

```mermaid
sequenceDiagram
    actor Control as Control API / migration thread
    participant Engine as DynamicEngineCore
    participant Scheduler as DynamicScheduler
    participant Executor as DynamicRayDistributedExecutor
    participant Worker as DynamicGPUWorker
    participant KV as KV synchronizer

    Control->>Engine: change_configuration(target_pp_layout)
    Engine->>Engine: Compare current and target layouts
    Engine->>Executor: Prepare target layer assignment
    Executor->>Worker: Load or expose layers required by target layout
    Engine->>Executor: Start KV migration
    Executor->>Worker: Install migration sender/receiver state

    loop Until max migration_lag <= threshold
        Engine->>Scheduler: Schedule next batch
        Scheduler-->>Engine: Batch with old pp_layer_config
        Engine->>Executor: Execute batch
        Executor->>Worker: Run with batch-local PP layout
        Worker->>KV: Send/apply KV patches
        Engine->>Executor: Query migration progress on all ranks
        Executor->>Worker: Read sender/receiver token progress
        Worker-->>Executor: Sent/applied tokens per peer
        Executor-->>Engine: max migration_lag
        Note over Engine,Executor: If the gap is still above the threshold,<br/>keep scheduling old-layout batches and polling again.
    end

    Engine->>Scheduler: Request layout switch boundary
    Scheduler-->>Engine: Boundary batch

    Engine->>Executor: Execute boundary batch
    Executor->>Worker: Finalize migration
    Worker->>KV: Ensure required KV patches are applied
    Worker->>Worker: Cleanup old layer weights and kv cache 
    Executor-->>Engine: Migration complete
    Engine->>Scheduler: Resume scheduling under target layout
```

## Layout Switch Boundary

The layout switch is a single boundary event. Before the boundary, newly
scheduled batches still use the current PP layout. After the boundary, newly
scheduled batches use the target PP layout.

```mermaid
stateDiagram-v2
    [*] --> CurrentLayout

    CurrentLayout: Scheduler emits current-layout batches
    CurrentLayout --> MigrationActive: change_configuration(target_pp_layout)

    MigrationActive: Prepare target layer assignment
    MigrationActive: Transfer KV state while current-layout batches continue
    MigrationActive: Poll sender/receiver token progress
    MigrationActive --> BoundaryBatch: migration_lag <= threshold

    BoundaryBatch: Scheduler emits boundary batch
    BoundaryBatch: Scheduler commits target PP layout
    BoundaryBatch --> TargetLayout: boundary batch dispatched

    TargetLayout: Scheduler emits target-layout batches
```

## Autoscaling Model

Autoscaling is a specific case of PP reconfiguration where the set of active
ranks changes. A shrink operation is represented as a target PP layout where
some ranks become inactive because their layer ranges are empty. An expand
operation assigns non-empty layer ranges to previously inactive ranks.
Therefore, autoscaling shares the same control logic for KV state transfer,
weight loading, and PP layout propagation as regular PP reconfiguration, with
additional logic to manage ranks that join or leave during autoscaling.
