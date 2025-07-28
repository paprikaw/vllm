# SPDX-License-Identifier: Apache-2.0

import copy
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
from transformers import AutoTokenizer
from vllm.logger import init_logger
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.dynamic_core import DynamicEngineCore
from vllm.v1.engine.core import EngineCore
from vllm.v1.core.sched.dynamic_scheduler import DynamicScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.executor.dynamic_ray_distributed_executor import DynamicRayDistributedExecutor
from vllm.v1.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.v1.worker.dynamic_gpu_worker import DynamicGPUWorker
from vllm.v1.executor.abstract import Executor, UniProcExecutor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.core.sched.dynamic_scheduler import MigrationStatus

from ...utils import create_new_process_for_each_test
logger = init_logger(__name__)
if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

MODEL_NAME = "/root/.cache/huggingface/Qwen/Qwen3-32B-AWQ"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
PROMPT = "Hello my name is Robert and I love quantization kernels ha"
PROMPT_TOKENS = TOKENIZER(PROMPT).input_ids



def make_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=str(uuid.uuid4()),
        prompt_token_ids=PROMPT_TOKENS,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=SamplingParams(),
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
    )

@create_new_process_for_each_test()
def test_engine_core_migration(monkeypatch: pytest.MonkeyPatch):
    """
    Test that the engine can handle multiple concurrent batches.
    """
    assert len(PROMPT_TOKENS) == 12
    
    def make_request_with_max_tokens(req_id: int,
                                     max_tokens: int) -> EngineCoreRequest:
        request = make_request()
        request.request_id = req_id
        request.sampling_params.max_tokens = max_tokens
        return request

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_PP_LAYER_PARTITION", "48,16")
        m.setenv("VLLM_PIPELINE_MEMORY_LIMIT", "24GB,24GB")
        m.setenv("RAY_DEDUP_LOGS", "0")
        engine_args = EngineArgs(
            model=MODEL_NAME,
            pipeline_parallel_size=2,
            gpu_memory_utilization=0.85,
            max_model_len=5000,
            max_num_batched_tokens=10,
            max_num_seqs=2,
            distributed_executor_backend="ray",
            enable_prefix_caching=False,
            enforce_eager=True,
            scheduler_cls=DynamicScheduler,
            worker_cls="vllm.v1.worker.dynamic_gpu_worker.DynamicGPUWorker"
            )
        vllm_config = engine_args.create_engine_config()
        executor_class = DynamicRayDistributedExecutor
        engine_core = DynamicEngineCore(vllm_config=vllm_config,
                                 executor_class=executor_class,
                                 log_stats=False)
        assert isinstance(engine_core.scheduler, DynamicScheduler)
        assert engine_core.batch_queue is not None
        # Add two requests in a row. Each request have 12 prompt tokens.
        req0 = make_request_with_max_tokens(0, 5)
        engine_core.add_request(req0)
        req1 = make_request_with_max_tokens(1, 5)
        engine_core.add_request(req1)
        req2 = make_request_with_max_tokens(2, 5)
        engine_core.add_request(req2)

        print("# Schedule Batch 1: (10, req0) #")
        assert engine_core.step_with_batch_queue() is None
        assert engine_core.batch_queue.qsize() == 1
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert len(scheduler_output.num_scheduled_tokens) == 1
        assert scheduler_output.num_scheduled_tokens[req0.request_id] == 10

        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 10
        assert engine_core.scheduler.requests[req1.request_id].num_computed_tokens == 0
        assert engine_core.scheduler.requests[req2.request_id].num_computed_tokens == 0
        assert engine_core.scheduler.get_num_unfinished_requests() == 3

        future = engine_core.migrate_layers(0, 1, 1)

        print("### Schedule Batch 2: (2, req0)")
        # Only one request is scheduled with old layer config
        assert engine_core.step_with_batch_queue() is None
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert len(scheduler_output.num_scheduled_tokens) == 1
        assert scheduler_output.num_scheduled_tokens[req0.request_id] == 2

        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 12
        assert engine_core.scheduler.requests[req1.request_id].num_computed_tokens == 0
        assert engine_core.scheduler.requests[req2.request_id].num_computed_tokens == 0
        assert engine_core.scheduler.get_num_unfinished_requests() == 3

        print("### Batch queue is full. Finish Batch 1 with no output")
        engine_core.step_with_batch_queue()
        assert engine_core.scheduler.get_num_unfinished_requests() == 3

        print("### Schedule Batch 3: (10, req1) with new layer config #")
        engine_core.step_with_batch_queue()
        assert engine_core.batch_queue.qsize() == 2
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert len(scheduler_output.num_scheduled_tokens) == 1
        assert scheduler_output.num_scheduled_tokens[req1.request_id] == 10
        # num_computed_tokens should have been updated immediately.
        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 12
        assert engine_core.scheduler.requests[req1.request_id].num_computed_tokens == 10
        assert engine_core.scheduler.requests[req2.request_id].num_computed_tokens == 0
        assert engine_core.scheduler.get_num_unfinished_requests() == 3

        print("# Batch queue is full. Finish Batch 2. Get first token of req0#")
        output = engine_core.step_with_batch_queue()
        assert len(output.outputs) == 1
        print(f"output: {output}")

        assert engine_core.scheduler.requests[req0.request_id].num_tokens == 13

        print("# Schedule Batch 4: (1, req0) in the decoding stage.")
        engine_core.step_with_batch_queue()
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert len(scheduler_output.num_scheduled_tokens) == 1
        assert scheduler_output.num_scheduled_tokens[req0.request_id] == 1
        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 13
        assert engine_core.scheduler.requests[req1.request_id].num_computed_tokens == 10
        assert engine_core.scheduler.requests[req2.request_id].num_computed_tokens == 0

        print("# Batch queue is full. Finish Batch 3 with no output#")
        engine_core.step_with_batch_queue()

        print("# Schedule Batch 5: (2, req1) and (8,req2)")
        engine_core.step_with_batch_queue()
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert scheduler_output.num_scheduled_tokens[req1.request_id] == 2
        assert scheduler_output.num_scheduled_tokens[req2.request_id] == 8
        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 13
        assert engine_core.scheduler.requests[req1.request_id].num_computed_tokens == 12
        assert engine_core.scheduler.requests[req2.request_id].num_computed_tokens == 8

        print("# Batch queue is full, Finish batch 4")
        output = engine_core.step_with_batch_queue()
        assert output is not None
        assert len(output.outputs) == 1
        print(f"output: {output}")
        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 13
        assert engine_core.scheduler.requests[req1.request_id].num_computed_tokens == 12
        assert engine_core.scheduler.requests[req2.request_id].num_computed_tokens == 8

        print("# Schedule old Batch 6 (1,req0)")
        engine_core.step_with_batch_queue()
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert len(scheduler_output.num_scheduled_tokens) == 1
        assert scheduler_output.num_scheduled_tokens[req0.request_id] == 1
        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 14
        assert engine_core.scheduler.requests[req1.request_id].num_computed_tokens == 12
        assert engine_core.scheduler.requests[req2.request_id].num_computed_tokens == 8

        print("# Batch queue is full, Finish batch 5, get the first token of req1")
        output = engine_core.step_with_batch_queue()
        assert len(output.outputs) == 1
        print(f"output: {output}")

        assert engine_core.scheduler.requests[req1.request_id].num_tokens == 13

        print("# Schedule Batch 7 (1, req1) and (4, req2), note request 1 is in decoding stage, but we enable chunk prefill")
        engine_core.step_with_batch_queue()
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert len(scheduler_output.num_scheduled_tokens) == 2
        assert scheduler_output.num_scheduled_tokens[req1.request_id] == 1
        assert scheduler_output.num_scheduled_tokens[req2.request_id] == 4
        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 14
        assert engine_core.scheduler.requests[req1.request_id].num_computed_tokens == 13
        assert engine_core.scheduler.requests[req2.request_id].num_computed_tokens == 12

        print("# Batch queue is full, Finish batch 6, get the decoded token of req0")
        output = engine_core.step_with_batch_queue()
        assert len(output.outputs) == 1
        print(f"output: {output}")

        print("# Schedule old Batch 8 (1,req0)")
        engine_core.step_with_batch_queue()
        scheduler_output = engine_core.batch_queue.queue[-1][1]
        assert len(scheduler_output.num_scheduled_tokens) == 1
        assert scheduler_output.num_scheduled_tokens[req0.request_id] == 1
        assert engine_core.scheduler.requests[req0.request_id].num_computed_tokens == 15
        assert engine_core.scheduler.requests[req1.request_id].num_computed_tokens == 13
        assert engine_core.scheduler.requests[req2.request_id].num_computed_tokens == 12

        print("# Batch queue is full, Finish batch 7, get the decoded token of req1 and req2")
        output = engine_core.step_with_batch_queue()
        assert len(output.outputs) == 2
        print(f"output: {output}")


        print(f"###########loop until request 0 is finished")
        step = 0
        # req_id = 0
        # expected_num_tokens = [
        #     engine_core.scheduler.requests[0].num_tokens + 1,
        #     engine_core.scheduler.requests[1].num_tokens + 1,
        # ]

        while engine_core.scheduler.get_num_unfinished_requests() == 3:
            if step % 2 == 0:
                print(f"########### scheduler_output after step {step}")
                output = engine_core.step_with_batch_queue()
                assert output is None
                scheduler_output = engine_core.batch_queue.queue[-1][1]
                print(f"scheduler_output: {scheduler_output}")
            else:
                print(f"############output after step {step}")
                output = engine_core.step_with_batch_queue()
                assert output is not None
                print(f"output: {output}")
            step += 1

        print(f"############ after request 0 is finished, also get the result of req1 and req2")
        output = engine_core.step_with_batch_queue()
        assert output is not None
        print(f"output: {output}")

        engine_core.done_migration()
        step = 0
    
        print(f"###########loop until request 1 and 2 is finished")
        while engine_core.scheduler.get_num_unfinished_requests() > 0:
            if step % 2 == 0:
                print(f"########### scheduler_output after step {step}")
                output = engine_core.step_with_batch_queue()
                assert output is None
                scheduler_output = engine_core.batch_queue.queue[-1][1]
                print(f"scheduler_output: {scheduler_output}")
            else:
                print(f"############output after step {step}")
                output = engine_core.step_with_batch_queue()
                assert output is not None
                print(f"output: {output}")
            step += 1