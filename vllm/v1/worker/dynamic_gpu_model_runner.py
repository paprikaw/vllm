# SPDX-License-Identifier: Apache-2.0
from hmac import new
import threading
import copy
import gc
import time
import weakref
from typing import TYPE_CHECKING, Optional, Union, Tuple
import sys
from compressed_tensors import Tensor
import humanize
from matplotlib.pylab import dtype
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from vllm._custom_ops import flexi_reshape_and_cache_flash
from vllm.attention.dynamic_layer import FlexiAttention
from vllm.attention.layer import Attention
from vllm.attention import AttentionType
from vllm.config import (get_layers_from_vllm_config)
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.dynamic_utils import FlexiKVTensorMeta
from vllm.distributed.parallel_state import get_pp_group
from vllm.dynamic_utils import ForegroundBackgroundGate
from vllm.sampling_params import SamplingType
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.v1.attention.backends.flash_attn import CommonAttentionMetadata, FlashAttentionMetadata
from vllm.v1.utils import dynamic_bind_kv_cache, dynamic_flexi_bind_kv_cache, human_readable_size, human_readable_duration

from vllm.sequence import IntermediateTensors
from vllm.utils import (LazyLoader)
from vllm.kv_allocator import kv_allocator
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheSpec, KVCacheConfig,
                                        SlidingWindowSpec)
from vllm.v1.utils import extract_layer_index
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, SpecDecodeMetadata
from vllm.v1.utils import dynamic_bind_kv_cache, dynamic_bind_single_kv_tensor
from vllm.v1.core.sched.dynamic_output import DynamicSchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput, EMPTY_MODEL_RUNNER_OUTPUT
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.block_table import PtrTable
from vllm.model_executor.models.dynamic_qwen3 import DynamicQwen3ForCausalLM
from vllm.model_executor.model_loader.dynamic_qwen3_loader import CustomModelLoader
from collections import defaultdict
from vllm.config import set_current_vllm_config
from threading import Lock
from vllm.v1.spec_decode.eagle import EagleProposer
from bitarray import bitarray
from vllm.v1.core.dynamic_kv_cache_utils import compact_cache_with_record
from vllm.distributed.kv_transfer.kv_connector.dynamic_kv_synchronizer import DynamicKVSynchronizer
from vllm.v1.worker.utils import get_flexi_kv_cache, get_flexi_kv_cache_multi_stream

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
from vllm.logger import init_logger
logger = init_logger(__name__)


class DynamicGPUModelRunner(GPUModelRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_lock = Lock()
        self.fbgate = ForegroundBackgroundGate()
        self.key_caches: list[list[int]] = []
        self.value_caches: list[list[int]] = []
        self.key_cache_ptrs: list[int] = []
        self.value_cache_ptrs: list[int] = []
        self.page_meta: Optional[torch.Tensor] = None
        # For flexi_direct implementation: per-layer pointer tensors
        # Each tensor has shape (num_blocks,) and dtype uint64, containing pointers to KV pages
        self.k_ptr_tensors: list[torch.Tensor] = []
        self.v_ptr_tensors: list[torch.Tensor] = []
        # PtrTable instances for flexi_direct - will be lazily initialized when num_layers is known
        self.k_ptr_table: Optional["PtrTable"] = None
        self.v_ptr_table: Optional["PtrTable"] = None
        # Track the start_layer that corresponds to the committed PtrTable
        # During migration, model.start_layer changes but PtrTable stays the same,
        # so we need to use this value for correct indexing until commit_ptr_tables() is called
        self._ptr_table_start_layer: int = 0
        # Create CustomModelLoader for weight preloading
        self.custom_loader = CustomModelLoader(self.vllm_config.load_config)
        # Inference stream for stream-specific synchronization (set by worker)
        self.inference_stream: Optional[torch.cuda.Stream] = None
        self._pending_k_ptr_table: Optional["PtrTable"] = None
        self._pending_v_ptr_table: Optional["PtrTable"] = None
    
    def set_inference_stream(self, stream: torch.cuda.Stream) -> None:
        """Set the inference stream for stream-specific synchronization."""
        self.inference_stream = stream
        logger.info(f"Inference stream set: {stream}")
    
    def stream_synchronize(self) -> None:
        """Synchronize only the inference stream, not the entire device.
        
        This avoids deadlock with _listen_loop's CUDA operations by not waiting
        for operations on other streams (like the default stream used by _listen_loop).
        Falls back to device synchronize if no inference stream is set.
        """
        if self.inference_stream is not None:
            self.inference_stream.synchronize()
        else:
            torch.cuda.synchronize()

    def load_model(self) -> None:
        """Override to use CustomModelLoader with weight preloading."""
        logger.info("Starting to load model %s...", self.model_config.model)
        from vllm.utils import DeviceMemoryProfiler, GiB_bytes
        from vllm.distributed.parallel_state import prepare_communication_buffer_for_model
        
        with DeviceMemoryProfiler() as m:
            time_before_load = time.perf_counter()
            logger.info(f"start to load model, vllm_config: {self.vllm_config}")
            
            # Use CustomModelLoader which will preload all weights
            self.model = self.custom_loader.load_model(
                vllm_config=self.vllm_config,
                model_config=self.model_config
            )
            
            if self.lora_config:
                logger.info(f"applied lora config")
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)
            if hasattr(self, "drafter"):
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
            if self.use_aux_hidden_state_outputs:
                self.model.set_aux_hidden_state_layers(
                    self.model.get_eagle3_aux_hidden_state_layers())
            time_after_load = time.perf_counter()
        
        self.model_memory_usage = m.consumed_memory
        logger.info("Model loading took %.4f GiB and %.6f seconds",
                    self.model_memory_usage / GiB_bytes,
                    time_after_load - time_before_load)
        prepare_communication_buffer_for_model(self.model)
    def initialize_kv_cache_for_layers(self, 
            kv_cache_specs: dict[str, KVCacheSpec],
            kv_cache_size: int,
            kv_cache_num_blocks: int,
            kv_synchronizer: DynamicKVSynchronizer,
            layers: Tuple[int, int],
            ) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"before initialize_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        assert len(self.attn_backends) == 1, "Only one attention backend is supported for now"
        kv_caches: dict[str, torch.Tensor] = {} 
        logger.info(f"kv_cache_specs: {kv_cache_specs}")
        for layer_name, kv_cache_spec in kv_cache_specs.items():
                layer_index = extract_layer_index(layer_name)
                if layer_index not in range(layers[0], layers[1]+1):
                    continue
                assert kv_cache_size % kv_cache_spec.page_size_bytes == 0
                num_blocks = kv_cache_size // kv_cache_spec.page_size_bytes
                assert num_blocks >= kv_cache_num_blocks
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    logger.info(f"layer_name: {layer_name}, kv_cache_shape: {kv_cache_shape}, dtype: {dtype}")
                    kv_caches[layer_name] = torch.zeros(kv_cache_shape,
                                                        dtype=dtype,
                                                        device=self.device)
                    dynamic_bind_single_kv_tensor(
                        layer_index=layer_index,
                        start_layer=self.model.model.start_layer,
                        end_layer=self.model.model.end_layer,
                        forward_context=self.vllm_config.compilation_config.static_forward_context,
                        kv_synchronizer=kv_synchronizer,
                        runner=self,
                        kv_tensor=kv_caches[layer_name],
                    )
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")
                # Added the kv cache spec to kv cache config.
                assert len(self.kv_cache_config.kv_cache_groups) == 1
                self.kv_cache_config.kv_cache_groups[0].layer_names.append(layer_name)
        if self.speculative_config and self.speculative_config.use_eagle():
            raise NotImplementedError("Eagle is not supported for dynamic weights")

        self._update_start_layer()
        if has_kv_transfer_group():
            raise NotImplementedError("KV transfer group is not supported for dynamic weights")

        logger.info(f"after initialize_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        return

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        layer_config: Tuple[int, int],
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        """
        Execute model with flexi_direct support.
        Builds k_ptr_tables/v_ptr_tables from block_table and passes them to forward context.
        """
        time_start = time.time()
        logger.info(f"getting forward lock taking {human_readable_duration(time.time() - time_start)}")
        with self.fbgate.foreground():
            if not isinstance(self.model, DynamicQwen3ForCausalLM):
                raise AssertionError(f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
            self.model.set_sched_layers(layer_config[0], layer_config[1])
            try:
                result = self._execute_model(scheduler_output, intermediate_tensors)
            except Exception as e:
                logger.exception(f"Error in execute_model: {e}")
                for request in scheduler_output.scheduled_new_reqs:
                    for block_id in request.block_ids:
                        logger.info(f"request {request.request_id} block id list: {block_id}")
                time.sleep(1)
                raise
            return result

    def _dynamic_prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[dict[str, FlashAttentionMetadata], torch.Tensor,
               Optional[SpecDecodeMetadata], Optional[torch.Tensor], Optional[torch.Tensor]]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)
        # Update PtrTables for flexi_direct after block_table has been committed
        # This runs asynchronously on GPU using efficient index_select operations
        k_ptr_tables_tensor = None
        v_ptr_tables_tensor = None
        if self.k_ptr_tensors and self.v_ptr_tensors:
            time_start = time.time()
            num_reqs = self.input_batch.num_reqs
            
            # PtrTable should already be initialized via commit_ptr_tables()
            # called during dynamic_initialize_kv_cache_flexi or finish_migration
            if self.k_ptr_table is None or self.v_ptr_table is None:
                raise RuntimeError(
                    f"PtrTable not initialized. "
                    "Ensure commit_ptr_tables() is called after KV cache initialization."
                )
            
            # Use the committed PtrTable's num_layers, NOT len(k_ptr_tensors)
            # During migration, k_ptr_tensors may have been extended with new layer slots,
            # but only the committed layers should participate in inference.
            # The new layers will become active after commit_ptr_tables() is called
            # at the end of migration (in finish_migration).
            
            # Get the block_table from input_batch (it's already on GPU after commit)
            block_table = self.input_batch.block_table[0].get_device_tensor()
            
            time_start = time.time()
            # Update ptr_tables using efficient GPU operations
            k_ptr_tables_tensor = self.k_ptr_table.update_from_block_table_and_ptr_tensors(
                block_table, self.k_ptr_tensors, num_reqs)
            v_ptr_tables_tensor = self.v_ptr_table.update_from_block_table_and_ptr_tensors(block_table, self.v_ptr_tensors, num_reqs)
            logger.info(f"kv ptr_tables update took {human_readable_duration(time.time() - time_start)}")
        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_scheduled_tokens])
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping for each KV cache group.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            block_size = kv_cache_group_spec.kv_cache_spec.block_size
            block_table: BlockTable = self.input_batch.block_table[
                kv_cache_group_id]
            # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            # block_table_indices: -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
            # where K is the max_num_blocks_per_req and the block size is 2.
            # NOTE(woosuk): We can't simply use `token_indices // block_size`
            # here because M (max_model_len) is not necessarily divisible by
            # block_size.
            block_table_indices = (
                req_indices * block_table.max_num_blocks_per_req +
                positions_np // block_size)
            block_table_cpu = block_table.get_cpu_tensor()
            block_numbers = block_table_cpu.flatten(
            )[block_table_indices].numpy()
            block_offsets = positions_np % block_size

            np.add(
                block_numbers * block_size,
                block_offsets,
                out=block_table.slot_mapping_np[:total_num_scheduled_tokens])
            # logger.info(f"block_table_indices: {block_table_indices}")
            # logger.info(f"block_table_cpu: {block_table_cpu.flatten()}")
            # logger.info(f"block_numbers: {block_numbers}")
            # logger.info(f"block_offsets: {block_offsets}")
            # logger.info(f"block_table.slot_mapping_np: {block_table.slot_mapping_np[:total_num_scheduled_tokens]}")

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        # Copy the tensors to the GPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions_cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions[:total_num_scheduled_tokens].copy_(
                self.positions_cpu[:total_num_scheduled_tokens],
                non_blocking=True)

        self.query_start_loc[:num_reqs + 1].copy_(
            self.query_start_loc_cpu[:num_reqs + 1], non_blocking=True)
        self.seq_lens[:num_reqs].copy_(self.seq_lens_cpu[:num_reqs],
                                       non_blocking=True)

        # Fill unused with -1. Needed for reshape_and_cache
        self.seq_lens[num_reqs:].fill_(0)
        self.query_start_loc[num_reqs + 1:].fill_(-1)

        query_start_loc = self.query_start_loc[:num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc, seq_lens=seq_lens)

        attn_metadata: dict[str, FlashAttentionMetadata] = {}
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):

            # Prepare for cascade attention if enabled & beneficial.
            common_prefix_len = 0
            if self.cascade_attn_enabled:
                common_prefix_len = self._compute_cascade_attn_prefix_len(
                    num_scheduled_tokens,
                    scheduler_output.
                    num_common_prefix_blocks[kv_cache_group_id],
                    kv_cache_group_spec.kv_cache_spec,
                    self.attn_metadata_builders[kv_cache_group_id],
                )

            attn_metadata_i = (
                self.attn_metadata_builders[kv_cache_group_id].build(
                    num_reqs=num_reqs,
                    num_actual_tokens=total_num_scheduled_tokens,
                    max_query_len=max_num_scheduled_tokens,
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata))
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return attn_metadata, logits_indices, spec_decode_metadata, k_ptr_tables_tensor, v_ptr_tables_tensor

    @torch.inference_mode()
    def _execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:

        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output)
        time_start_prepare = time.time()
        # Prepare the decoder inputs.
        attn_metadata, logits_indices, spec_decode_metadata, k_ptr_tables_tensor, v_ptr_tables_tensor = (
            self._dynamic_prepare_inputs(scheduler_output))
        
        # DEBUG: Log critical batch state for cross-node synchronization debugging
        pp_rank = get_pp_group().rank
        
        # Generate same step_id as in dynamic_utils.py for correlation
        all_sched_req_ids = sorted(scheduler_output.num_scheduled_tokens.keys())
        step_id = hash(tuple(all_sched_req_ids)) % 100000
        
        # Compare scheduler_output req_ids vs input_batch req_ids
        batch_req_ids = list(self.input_batch.req_ids)
        num_reqs = self.input_batch.num_reqs
        
        # Check if batch_req_ids match scheduler req_ids (critical for PP correctness)
        batch_req_set = set(batch_req_ids)
        sched_req_set = set(all_sched_req_ids)
        missing_in_batch = sched_req_set - batch_req_set  # In scheduler but not in batch
        extra_in_batch = batch_req_set - sched_req_set    # In batch but not in scheduler
        
        if missing_in_batch or (extra_in_batch and scheduler_output.total_num_scheduled_tokens > 0):
            logger.warning(f"[PP_MISMATCH] rank={pp_rank} step_id={step_id} "
                          f"BATCH/SCHED MISMATCH! missing_in_batch={len(missing_in_batch)}, "
                          f"extra_in_batch={len(extra_in_batch)}")
        
        # Log batch state for debugging
        num_computed_tokens_first5 = self.input_batch.num_computed_tokens_cpu[:min(5, num_reqs)].tolist()
        scheduled_tokens_per_req = [scheduler_output.num_scheduled_tokens.get(req_id, 0) 
                                    for req_id in batch_req_ids[:5]]
        # logger.info(f"[PP_BATCH] rank={pp_rank} step_id={step_id} batch_num_reqs={num_reqs} "
        #            f"sched_num_reqs={len(all_sched_req_ids)} "
        #            f"batch_req_ids_first5={[r[-8:] for r in batch_req_ids[:5]]} "
        #            f"num_computed_first5={num_computed_tokens_first5} "
        #            f"scheduled_tokens_first5={scheduled_tokens_per_req}")

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.vllm_config.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                from vllm.utils import round_up
                num_input_tokens = round_up(num_scheduled_tokens, tp_size)
            else:
                num_input_tokens = num_scheduled_tokens

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)
        else:
            mm_embeds = []

        if self.is_multimodal_model and get_pp_group().is_first_rank:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None
        if self.uses_mrope:
            positions = self.mrope_positions[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)
        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        # Get start_layer for computing local layer index in flexi_direct
        # IMPORTANT: Use _ptr_table_start_layer (not model.start_layer) because during migration,
        # model.start_layer changes when weights are loaded but PtrTable stays the same.
        # Using model.start_layer would cause incorrect indexing into PtrTable.
        start_layer = self._ptr_table_start_layer
        with set_forward_context(attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens,
                                 k_ptr_tables=k_ptr_tables_tensor,
                                 v_ptr_tables=v_ptr_tables_tensor,
                                 start_layer=start_layer):
            self.maybe_setup_kv_connector(scheduler_output)
            try:
                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )
            except Exception as e:
                time.sleep(2)
                logger.info(f"Exception during model forward: {e}")
                raise e
            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
                self.get_finished_kv_transfers(scheduler_output))
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            if not broadcast_pp_output:
                return hidden_states
            assert isinstance(hidden_states, IntermediateTensors)
            get_pp_group().send_tensor_dict(hidden_states.tensors,
                                            all_gather_group=get_tp_group())
            logits = None
        else:
            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)
            #  After compute_logits
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group().broadcast_tensor_dict(
                model_output_broadcast_data, src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            assert logits is not None
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)
        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        
        # DEBUG: Add synchronize points to find the real blocking location
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output,
        )
        
        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
        if not self.use_spec_decode:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        elif self.speculative_config.method == "ngram":
            assert isinstance(self.drafter, NgramProposer)
            spec_token_ids = self.generate_draft_token_ids(
                valid_sampled_token_ids, sampling_metadata)
        elif self.speculative_config.method == "medusa":

            assert isinstance(self.drafter, MedusaProposer)
            if max_gen_len == 1:
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                for num_draft, tokens in zip(
                        spec_decode_metadata.num_draft_tokens,
                        valid_sampled_token_ids):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1

                indices = torch.tensor(indices,
                                       device=sample_hidden_states.device)
                hidden_states = sample_hidden_states[indices]

            spec_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )
        elif self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # TODO(woosuk): Refactor the loop.
            next_token_ids: list[int] = []
            for i, token_ids in enumerate(valid_sampled_token_ids):
                if token_ids:
                    # Common case.
                    next_token_id = token_ids[-1]
                else:
                    # Partial prefill (rare case).
                    # Get the next token id from the request state.
                    req_id = self.input_batch.req_ids[i]
                    req_state = self.requests[req_id]
                    seq_len = (req_state.num_computed_tokens +
                               scheduler_output.num_scheduled_tokens[req_id])
                    next_token_id = req_state.get_token_id(seq_len)
                next_token_ids.append(next_token_id)
            next_token_ids = torch.tensor(next_token_ids,
                                          dtype=torch.int32,
                                          device=self.device)
            # At this moment, we assume all eagle layers belong to the same KV
            # cache group, thus using the same attention metadata.
            eagle_attn_metadata = attn_metadata[
                self.drafter.attn_layer_names[0]]


            # NOTE: deepseek_mtp uses MLA which does not have `block_table`
            if hasattr(eagle_attn_metadata, "block_table"):
                block_table = eagle_attn_metadata.block_table
            else:
                block_table = None

            if spec_decode_metadata is None:
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids[:num_scheduled_tokens]
                target_positions = positions[:num_scheduled_tokens]
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states],
                        dim=-1)
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
                target_slot_mapping = eagle_attn_metadata.slot_mapping
                cu_num_tokens = eagle_attn_metadata.query_start_loc
            else:
                # TODO(woosuk): Refactor this.
                num_draft_tokens = spec_decode_metadata.num_draft_tokens
                num_rejected_tokens = [
                    n + 1 - len(valid_sampled_token_ids[i]) if n > 0 else 0
                    for i, n in enumerate(num_draft_tokens)
                ]
                num_rejected_tokens_tensor = async_tensor_h2d(
                    num_rejected_tokens,
                    dtype=torch.int32,
                    target_device=self.device,
                    pin_memory=True)
                num_tokens = num_scheduled_tokens - sum(num_rejected_tokens)
                cu_num_tokens, token_indices = self.drafter.prepare_inputs(
                    eagle_attn_metadata.query_start_loc,
                    num_rejected_tokens_tensor,
                    num_tokens,
                )
                target_token_ids = self.input_ids[token_indices]
                target_positions = positions[token_indices]
                if self.use_aux_hidden_state_outputs:
                    target_hidden_states = torch.cat(
                        [h[token_indices] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[token_indices]
                target_slot_mapping = eagle_attn_metadata.slot_mapping[
                    token_indices]
            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                target_slot_mapping=target_slot_mapping,
                next_token_ids=next_token_ids,
                cu_num_tokens=cu_num_tokens,
                block_table=block_table,
                sampling_metadata=sampling_metadata,
            )
            spec_token_ids = draft_token_ids.tolist()

        # Clear KVConnector state after all KVs are generated.
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        )

    def profile_run(self) -> None: 
        """
        Profile run can be run during the model is running,
        Needed to acquire a lock
        """
        with self.forward_lock:
            super().profile_run()

    @torch.inference_mode()
    def initialize_intermediate_states(
        self,
    ) -> None:

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        num_tokens = self.max_num_tokens
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = max_num_reqs if num_tokens >= max_num_reqs else num_tokens
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        if not get_pp_group().is_first_rank:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = (
                    self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device))

    def get_kv_cache_spec_for_layers(self, layer_range: Tuple[int, int]) -> dict[str, KVCacheSpec]:
        """
        Get the KV cache spec for the given layers.
        Copy from get_kv_cache_spec() but only return the KV cache spec for the given layers.
        """
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            if extract_layer_index(layer_name) not in range(layer_range[0], layer_range[1]+1):
                continue
            # TODO: Support other attention modules, e.g., cross-attention
            if attn_module.attn_type == AttentionType.DECODER:
                if attn_module.sliding_window is not None:
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window,
                        use_mla=use_mla)
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")
        return kv_cache_spec


    # def bind_layer_kv_tensor(self, layer_index: int, kv_tensor: torch.Tensor) -> None:
    #     """Bind a single layer's KV tensor to runner caches and forward context.

    #     - 更新本 runner 的 `self.kv_caches`
    #     - 将 forward context 中对应 Attention 的 `kv_cache[ve]` 指向该张量
    #     - 如有必要，补齐 kv_cache_config 的 layer_names，确保后续 attn_metadata 构建覆盖到该层

    #     线程安全：内部获取 forward_lock。
    #     """
    #     if not isinstance(self.model, DynamicQwen3ForCausalLM):
    #         raise AssertionError(
    #             f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
    #     # Determine local index in runner kv cache list
    #     start_layer: int = self.model.model.start_layer
    #     end_layer: int = self.model.model.end_layer
    #     assert layer_index >= start_layer and layer_index < end_layer, f"Layer {layer_index} outside of current model range [{start_layer}, {end_layer}]"

    #     local_index = layer_index - start_layer
    #     assert local_index < len(self.kv_caches), f"Local index {local_index} is out of range, kv_tensor length: {len(self.kv_caches)}"
    #     logger.info(f"bind kv tensor for {layer_index}, local_index={local_index}, kv_tensor length: {len(self.kv_caches)}")
    #     # Basic sanity: non-empty tensor
    #     assert isinstance(kv_tensor, torch.Tensor) and kv_tensor.numel() > 0, (
    #         f"Binding empty KV tensor for layer {layer_index}")
    #     self.kv_caches[local_index] = kv_tensor
    #     # Bind to forward context
    #     layer_name: str = self.get_layer_name_for_index(layer_index)
    #     fctx: dict[str, "Attention"] = \
    #         self.vllm_config.compilation_config.static_forward_context
    #     if layer_name not in fctx:
    #         raise KeyError(
    #             f"No attention layer named {layer_name} in forward_context.")

    #     attn_module = fctx[layer_name]
    #     attn_module.kv_cache = [kv_tensor]

    #     # 3) 补齐 kv_cache_config 的 layer_names，保证后续 attn_metadata 覆盖
    #     group = self.kv_cache_config.kv_cache_groups[0]
    #     if layer_name not in group.layer_names:
    #         # 按 layer_index 位置插入，保持有序
    #         insert_idx = len(group.layer_names)
    #         target_idx = extract_layer_index(layer_name)
    #         for i, name in enumerate(group.layer_names):
    #             if extract_layer_index(name) > target_idx:
    #                 insert_idx = i
    #                 break
    #         group.layer_names.insert(insert_idx, layer_name)

    # def flexi_bind_layer_kv_tensor(self, layer_index: int, kv_tensor: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    #     """Bind a single layer's KV tensor to runner caches and forward context.

    #     - 更新本 runner 的 `self.kv_caches`
    #     - 将 forward context 中对应 Attention 的 `kv_cache[ve]` 指向该张量
    #     - 如有必要，补齐 kv_cache_config 的 layer_names，确保后续 attn_metadata 构建覆盖到该层

    #     线程安全：内部获取 forward_lock。
    #     """
    #     if not isinstance(self.model, DynamicQwen3ForCausalLM):
    #         raise AssertionError(
    #             f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
    #     # Determine local index in runner kv cache list
    #     start_layer: int = self.model.model.start_layer
    #     end_layer: int = self.model.model.end_layer
    #     assert layer_index >= start_layer and layer_index < end_layer, f"Layer {layer_index} outside of current model range [{start_layer}, {end_layer}]"
    #     local_index = layer_index - start_layer
    #     assert len(self.key_caches) == len(self.value_caches) == len(self.key_cache_ptrs) == len(self.value_cache_ptrs), "key_caches and value_caches length mismatch"
    #     assert local_index < len(self.key_caches), f"Local index {local_index} is out of range, key_cache length: {len(self.key_caches)}"
    #     logger.info(f"bind kv tensor for {layer_index}, local_index={local_index}, kv_tensor length: {len(self.key_caches)}")

    #     key_cache_list, value_cache_list, key_cache_ptr, value_cache_ptr = self._get_flexi_kv_cache_from_gathered_kv_tensor(slot_mapping, kv_tensor)

    #     self.key_caches[local_index] = key_cache_list
    #     self.value_caches[local_index] = value_cache_list
    #     self.key_cache_ptrs[local_index] = key_cache_ptr
    #     self.value_cache_ptrs[local_index] = value_cache_ptr

    #     # Bind to forward context
    #     layer_name: str = self.get_layer_name_for_index(layer_index)
    #     fctx: dict[str, "Attention"] = \
    #         self.vllm_config.compilation_config.static_forward_context
    #     if layer_name not in fctx:
    #         raise KeyError(
    #             f"No attention layer named {layer_name} in forward_context.")
    #     attn_module = fctx[layer_name]
    #     assert isinstance(attn_module, FlexiAttention), f"Attention module for layer {layer_name} is not FlexiAttention"
    #     attn_module.key_cache = key_cache_list
    #     attn_module.value_cache = value_cache_list
    #     attn_module.key_dev_ptr = key_cache_ptr
    #     attn_module.value_dev_ptr = value_cache_ptr

    #     # 3) 补齐 kv_cache_config 的 layer_names，保证后续 attn_metadata 覆盖
    #     group = self.kv_cache_config.kv_cache_groups[0]
    #     if layer_name not in group.layer_names:
    #         # 按 layer_index 位置插入，保持有序
    #         insert_idx = len(group.layer_names)
    #         target_idx = extract_layer_index(layer_name)
    #         for i, name in enumerate(group.layer_names):
    #             if extract_layer_index(name) > target_idx:
    #                 insert_idx = i
    #                 break
            # group.layer_names.insert(insert_idx, layer_name)

    def has_layer(self, layer_index: int) -> bool:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"debug: ---------------------has_layer: {layer_index} in range {self.model.model.start_layer} to {self.model.model.end_layer}")
        return layer_index in range(self.model.model.start_layer, self.model.model.end_layer)

    def add_layers(self, layers_list: list[Tuple[int, int]], device: torch.device) -> None:
        if not isinstance(self.model, DynamicQwen3ForCausalLM):
            raise AssertionError(f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
        time_start = time.time()
        # gc.collect()
        # torch.cuda.empty_cache()
        # available_memory = torch.cuda.mem_get_info()[0]
        # num_added_layers = sum([layer[1] - layer[0] + 1 for layer in layers_list])
        # time_wait = 0
        # while available_memory < num_added_layers * self.model.get_layer_weight_size() and time_wait < 10:
        #     logger.info(f"before add layer, empty cache")
        #     time_wait += 1
        #     time.sleep(0.1)
        #     gc.collect()
        #     torch.cuda.empty_cache()
        #     available_memory = torch.cuda.mem_get_info()[0]
        #     logger.info(f"waiting for available memory to be enough, time_wait: {time_wait}, available_memory: {available_memory}")
        # assert available_memory > num_added_layers * self.model.get_layer_weight_size(), \
        #     f"Available memory: {human_readable_size(available_memory)} is not enough for {num_added_layers} layers"

        model: DynamicQwen3ForCausalLM = self.model
        # Use the existing custom_loader that has preloaded weights
        with set_current_vllm_config(self.vllm_config):
            for layers in layers_list:
                assert len(layers) == 2
                self.custom_loader.load_qwen3_layers(self.vllm_config, self.model_config, layers, model, device)
        logger.info(f"Model Runner, Added layers {layers_list} to model on device {device}, took {time.time() - time_start:.2f} seconds")

    def reinitialize_kv_cache(self, kv_cache_config: KVCacheConfig, kv_synchronizer: DynamicKVSynchronizer) -> None:
        """
        Copied from initialize_kv_cache() but only called when reinitialize the kv cache.
        The only difference is that we don't need to initialize the attention backend again.
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        assert False
        self.kv_cache_config = kv_cache_config
        kv_caches: dict[str, torch.Tensor] = {}
        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks
                # Handle known KV cache spec types explicitly for clarity.
                if isinstance(kv_cache_spec, (FullAttentionSpec, SlidingWindowSpec)):
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    kv_caches[layer_name] = torch.zeros(
                        kv_cache_shape, dtype=dtype, device=self.device)
                elif isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    kv_caches[layer_name] = torch.zeros(
                        kv_cache_shape, dtype=dtype, device=self.device)
                else:
                    raise ValueError(
                        f"Unknown KV cache spec type: {type(kv_cache_spec).__name__}.")

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        # dynamic_bind_kv_cache(
        #     kv_caches,
        #     self.vllm_config.compilation_config.static_forward_context,
        #     self.kv_caches,
        #     kv_synchronizer=kv_synchronizer)
        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def get_single_kv_tensor_size(self) -> int:
        """Return the size in bytes of a single layer's KV cache tensor.

        If KV caches are not initialized yet, returns 0.
        """
        assert hasattr(self, "kv_caches") and self.kv_caches and isinstance(self.kv_caches[0], torch.Tensor) 

        t: torch.Tensor = self.kv_caches[0]
        return int(t.numel() * t.element_size())

    def remove_layers(self, layers_list: list[Tuple[int, int]], device: torch.device) -> None:
        with device:
            torch.cuda.set_device(device)
            if not isinstance(self.model, DynamicQwen3ForCausalLM):
                raise AssertionError(f"model is not a DynamicQwen3ForCausalLM: {self.model.__class__.__name__}")
            logger.info(f"before delete_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            for layers in layers_list:
                logger.info(f"Deleting layers {layers}")
                self.model.delete_layers(layers)

            deleted_layers = set(layer for layers in layers_list for layer in range(layers[0], layers[1]+1))
            logger.info(f"deleted layer set {deleted_layers}")
            # Delete the layer from forward context
            self.vllm_config.compilation_config.static_forward_context = {
                layer_name: attn_module for layer_name, attn_module in self.vllm_config.compilation_config.static_forward_context.items()
                if extract_layer_index(layer_name) not in deleted_layers
            }
            for i in range(len(self.kv_caches)):
                logger.info(f"kv_caches[{i}] shape: {self.kv_caches[i].shape}")
            logger.info(f"after delete_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def dynamic_initialize_kv_cache(self, 
                                    kv_cache_config: KVCacheConfig, 
                                    dynamic_kv_synchronizer: DynamicKVSynchronizer,
                                    num_blocks: int
                                    ) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        self.kv_cache_config = kv_cache_config
        self.initialize_attn_backend(kv_cache_config)

        kv_caches: dict[str, torch.Tensor] = {}

        # Verify kv_cache_config.num_blocks matches the passed num_blocks
        # Both should be set by dynamic_core to the same value
        assert num_blocks == kv_cache_config.num_blocks, (
            f"num_blocks mismatch: passed={num_blocks}, config={kv_cache_config.num_blocks}. "
            "dynamic_core should update kv_cache_configs before calling dynamic_initialize_from_config"
        )

        for i, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                assert num_blocks >= kv_cache_config.num_blocks
                if isinstance(kv_cache_spec, AttentionSpec):
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    kv_caches[layer_name] = torch.zeros(kv_cache_shape,
                                                        dtype=dtype,
                                                        device=self.device)
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")
                logger.info(f"kv_caches shape: {kv_cache_shape}")
        
        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        dynamic_bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
            dynamic_kv_synchronizer
            )
        del kv_caches
        logger.info(f"debug---------------- init kv cache done, kv block num: {len(self.kv_caches[0][0])}")
        if has_kv_transfer_group():
            assert False
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def dynamic_initialize_kv_cache_flexi(self, kv_cache_config: KVCacheConfig, kv_synchronizer: DynamicKVSynchronizer, num_blocks: int) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        self.kv_cache_config = kv_cache_config
        self.initialize_attn_backend(kv_cache_config)

        kv_caches: dict[str, torch.Tensor] = {} 
        key_cache_ptrs: dict[str, int] = {}
        value_cache_ptrs: dict[str, int] = {}
        key_caches: dict[str, list[int]] = {}
        value_caches: dict[str, list[int]] = {}
        assert len(kv_cache_config.kv_cache_groups) == 1, "Only one KV cache group is supported."
        assert len(self.attn_backends) == 1, "Only one attention backend is supported."

        kv_cache_group = kv_cache_config.kv_cache_groups[0]
        kv_cache_spec = kv_cache_group.kv_cache_spec
        assert isinstance(kv_cache_spec, AttentionSpec), "KV cache spec must be an AttentionSpec."
        self.kv_cache_dtype = kv_cache_spec.dtype
        kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
            num_blocks, kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
        self.kv_cache_shape = kv_cache_shape
        # Verify kv_cache_config.num_blocks matches the passed num_blocks
        # Both should be set by dynamic_core to the same value
        assert num_blocks == kv_cache_config.num_blocks, (
            f"num_blocks mismatch: passed={num_blocks}, config={kv_cache_config.num_blocks}. "
            "dynamic_core should update kv_cache_configs before calling dynamic_initialize_from_config"
        )
        assert len(kv_cache_shape) == 5 # (2, nkvblocks, blockdim, n_head, headdim)
        block_shape = kv_cache_shape[2:]
        # Create page_meta tensor with correct shape [block_size, num_heads]
        # This tensor's dtype AND strides are used by flexi flash attention kernels
        # to understand the KV cache memory layout
        # block_shape is (block_size, num_heads, head_size)
        block_size, num_heads, head_size = block_shape
        
        # Get the actual KV cache dtype (not the string "auto")
        from vllm.utils import get_kv_cache_torch_dtype
        kv_cache_torch_dtype = get_kv_cache_torch_dtype(
            self.kv_cache_dtype, self.model_config.dtype
        )
        
        # CRITICAL FIX: Use zeros() instead of empty() to avoid garbage data
        # page_meta provides stride/shape info to kernels. If it contains uninitialized
        # garbage values, kernels may compute wrong offsets leading to:
        # 1. Corrupted outputs (jebrish text)
        # 2. CUDA illegal memory access errors
        # This is especially critical during migration when page_meta is reused
        # for apply_one_patch_to_kv_cache operations.
        self.page_meta = torch.zeros((block_size, num_heads, head_size), dtype=kv_cache_torch_dtype, device=self.device)
        logger.info(f"page_meta initialized: shape={self.page_meta.shape}, strides={self.page_meta.stride()}, dtype={self.page_meta.dtype}")

        start_time = time.time()
        for layer_name in kv_cache_group.layer_names:
            key_caches[layer_name], value_caches[layer_name], key_cache_ptrs[layer_name], value_cache_ptrs[layer_name], _ = kv_allocator.allocate_with_cuda_async(kv_cache_shape[1], list(block_shape), self.kv_cache_dtype, self.device)
        
        logger.info(f"time to intialize kv blocks:{human_readable_duration(time.time() - start_time)}")
        # logger.info(f"key cache shape: {key_caches[kv_cache_group.layer_names[0]][0].shape}, value cache shape:{value_caches[kv_cache_group.layer_names[0]][0].shape}")
        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)
        
        dynamic_flexi_bind_kv_cache(
            key_caches,
            value_caches,
            key_cache_ptrs,
            value_cache_ptrs,
            self.vllm_config.compilation_config.static_forward_context,
            kv_synchronizer,
            self.key_caches,
            self.value_caches,
            self.key_cache_ptrs,
            self.value_cache_ptrs,
            self.page_meta,
            self.k_ptr_tensors,
            self.v_ptr_tensors,
        )
        # Initialize PtrTable stacked tensors after binding all KV caches
        self.commit_ptr_tables(self.k_ptr_tensors, self.v_ptr_tensors, is_first_time=True)
        
        del kv_caches
        if has_kv_transfer_group():
            assert False
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def prepare_ptr_tables(self, num_layers: int) -> None:
        """
        Prepare new stacked tensors and store them for later commit.
        
        Call this immediately after KV cache binding completes (e.g., after migration
        binding loop finishes). This does the expensive work of building stacked tensors
        and stores them in _pending_k_stacked/_pending_v_stacked.
        
        IMPORTANT: This does NOT replace the current PtrTable. The current PtrTable
        continues to be used for inference. Only commit_ptr_tables() will switch
        to the new PtrTable atomically.
        
        The prepared tensors can then be atomically committed via commit_ptr_tables().
        """
        assert self.k_ptr_tensors and  self.v_ptr_tensors
        
        
        # Create new PtrTable instances for the new layer count, but do NOT replace
        # the current ones yet. They are stored as pending and will be committed later.
        from vllm.utils import cdiv
        max_num_blocks_per_req = cdiv(self.max_model_len, 
                                      self.cache_config.block_size)
        
        # Create pending PtrTables (these will replace current ones on commit)
        self._pending_k_ptr_table = PtrTable(
            max_num_reqs=self.max_num_reqs,
            max_num_blocks_per_req=max_num_blocks_per_req,
            num_layers=num_layers,
            pin_memory=self.pin_memory,
            device=self.device,
        )
        self._pending_v_ptr_table = PtrTable(
            max_num_reqs=self.max_num_reqs,
            max_num_blocks_per_req=max_num_blocks_per_req,
            num_layers=num_layers,
            pin_memory=self.pin_memory,
            device=self.device,
        )
        
        # Prepare new stacked tensors using the pending PtrTables
        # self._pending_k_stacked = self._pending_k_ptr_table.prepare_stacked_tensors(self.k_ptr_tensors)
        # self._pending_v_stacked = self._pending_v_ptr_table.prepare_stacked_tensors(self.v_ptr_tensors)
        
        logger.info(f"prepare_ptr_tables: prepared stacked tensors for {num_layers} layers")

    def commit_ptr_tables(self, k_ptr_tensor_list: list[Tensor], v_ptr_tensor_list: list[Tensor], is_first_time: Optional[bool] = False, target_start_layer: Optional[int] = None) -> None:
        """
        Atomically switch to the prepared PtrTables and stacked tensors.
        
        Call this after prepare_ptr_tables() to atomically commit the prepared
        PtrTables and stacked tensors. This is a fast pointer assignment operation.
        
        If prepare_ptr_tables() was not called, this will call it first.
        
        This method:
        1. Replaces k_ptr_table/v_ptr_table with the pending ones
        2. Commits the pending stacked tensors to the new PtrTables
        
        Args:
            target_start_layer: If provided, use this as the start_layer for PtrTable indexing.
                               This is critical during migration when the model's start_layer
                               will change AFTER this commit (e.g., after delete_layers).
                               If None, uses current model.start_layer.
        """
        # This can happen for:
        # 1. is_first_time=True (initial setup)
        # 2. Sender calling commit after deleting layers (never called prepare)
        if self._pending_k_ptr_table is None:
            self.prepare_ptr_tables(len(k_ptr_tensor_list))
        
        # Verify layer counts match
        assert len(k_ptr_tensor_list) == self._pending_k_ptr_table.num_layers, \
            f"k_ptr_tensor_list length {len(k_ptr_tensor_list)} != pending k_ptr_table num_layers {self._pending_k_ptr_table.num_layers}"
        assert len(v_ptr_tensor_list) == self._pending_v_ptr_table.num_layers, \
            f"v_ptr_tensor_list length {len(v_ptr_tensor_list)} != pending v_ptr_table num_layers {self._pending_v_ptr_table.num_layers}" 
        # Update start_layer to match target configuration (not current model state)
        # This is critical during migration when model.start_layer will change AFTER this commit
        old_start_layer = self._ptr_table_start_layer
        if target_start_layer is not None:
            self._ptr_table_start_layer = target_start_layer
        else:
            self._ptr_table_start_layer = getattr(self.model.model, 'start_layer', 0)
        logger.info(f"commit_ptr_tables: updating _ptr_table_start_layer from {old_start_layer} to {self._ptr_table_start_layer}")

        # Atomic switch: replace current PtrTables with pending ones
        self.k_ptr_table = self._pending_k_ptr_table
        self.v_ptr_table = self._pending_v_ptr_table
        
        # Commit the stacked tensors to the new PtrTables
        self.k_ptr_table.set_ptr_tensors(k_ptr_tensor_list)
        self.v_ptr_table.set_ptr_tensors(v_ptr_tensor_list)
        
        # Clear pending state
        self._pending_k_ptr_table = None
        self._pending_v_ptr_table = None
        
        logger.info(f"commit_ptr_tables: committed stacked tensors with {self.k_ptr_table.num_layers} layers, "
                    f"_ptr_table_start_layer updated to {self._ptr_table_start_layer}")

    def update_kv_ptr_tensor(self, k_ptr_tensor_list, v_ptr_tensor_list) -> None:
        """
        Update the cached stacked tensors to reflect current k_ptr_tensors/v_ptr_tensors.
        
        Use this after resize/compact operations where the ptr_tensors content changes
        (different pointer values or different num_blocks) but num_layers stays the same.
        
        This also updates _ptr_table_start_layer to match the current model.start_layer,
        which is critical after layer deletion/addition.
        """
        time_start = time.time()
        assert self.k_ptr_table is not None and self.k_ptr_tensors
        assert self.v_ptr_table is not None and self.v_ptr_tensors
        assert len(k_ptr_tensor_list) == self.k_ptr_table.num_layers
        assert len(v_ptr_tensor_list) == self.v_ptr_table.num_layers

        
        self.k_ptr_table.set_ptr_tensors(k_ptr_tensor_list)
        self.v_ptr_table.set_ptr_tensors(v_ptr_tensor_list)
        
        logger.info(f"refresh_ptr_tensors_cache: updated stacked tensors cache in {human_readable_duration(time.time() - time_start)}")

    def flexi_atomic_switch_kv_cache_config_for_layers(self, layers_list: list[Tuple[int, int]]) -> None:
        '''
        目前release_kv_cache依赖于model.start_layers来进行kv cache list的删除。
        因此只能先release kv cache再删除layers
        '''
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"before release_kv_cache_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        # Delete the kv cache from kv_cache list
        start_layer = self.model.model.start_layer
        self.key_caches = [
            key_cache for idx, key_cache in enumerate(self.key_caches) 
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]
        self.value_caches = [
            value_cache for idx, value_cache in enumerate(self.value_caches)
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]
        self.key_cache_ptrs = [
            ptr for idx, ptr in enumerate(self.key_cache_ptrs)
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]
        self.value_cache_ptrs = [
            ptr for idx, ptr in enumerate(self.value_cache_ptrs)
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]
        
        # Also update k_ptr_tensors and v_ptr_tensors for flexi_direct
        self.k_ptr_tensors = [
            ptr_tensor for idx, ptr_tensor in enumerate(self.k_ptr_tensors)
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]
        self.v_ptr_tensors = [
            ptr_tensor for idx, ptr_tensor in enumerate(self.v_ptr_tensors)
            if not any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list)
        ]
        # Note: PtrTable stacked tensors will be rebuilt atomically after migration completes
        # via commit_ptr_tables() to ensure atomic switch during inference

        # Delete layer name from kv_cache_config
        deleted_layers = set(layer for layers in layers_list for layer in range(layers[0], layers[1]+1))
        group = self.kv_cache_config.kv_cache_groups[0]
        group.layer_names = [
            name for name in group.layer_names
            if extract_layer_index(name) not in deleted_layers
        ]
        forward_context = self.vllm_config.compilation_config.static_forward_context
        deleted_layer_names = [name for name in forward_context.keys() if extract_layer_index(name) in deleted_layers]

        for layer_name in deleted_layer_names:
            del forward_context[layer_name]

    def atomic_switch_kv_cache_config_for_layers(self, layers_list: list[Tuple[int, int]]) -> None:
        '''
        目前release_kv_cache依赖于model.start_layers来进行kv cache list的删除。
        因此只能先release kv cache再删除layers
        
        NOTE: This function is used in non-flexi mode. Unlike flexi mode where
        the caller explicitly frees KV cache tensors, this function must release
        the GPU memory by removing references and calling gc.collect() + empty_cache().
        '''
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        
        logger.info(f"before atomic_switch_kv_cache_config_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, kv_caches length: {len(self.kv_caches)}")

        # Delete the kv cache from kv_cache list
        start_layer = self.model.model.start_layer
        # First, explicitly delete the tensors to be removed
        indices_to_remove = set()
        for idx in range(len(self.kv_caches)):
            if any(idx in range(layers[0]-start_layer, layers[1]-start_layer+1) for layers in layers_list):
                indices_to_remove.add(idx)
        
        # Explicitly delete the KV cache tensors before filtering
        for idx in sorted(indices_to_remove, reverse=True):
            if idx < len(self.kv_caches):
                del self.kv_caches[idx]
        
        # Delete layer name from kv_cache_config
        deleted_layers = set(layer for layers in layers_list for layer in range(layers[0], layers[1]+1))
        group = self.kv_cache_config.kv_cache_groups[0]
        group.layer_names = [
            name for name in group.layer_names
            if extract_layer_index(name) not in deleted_layers
        ]
        forward_context = self.vllm_config.compilation_config.static_forward_context
        deleted_layer_names = []
        for layer_name, attn_module in forward_context.items():
            if extract_layer_index(layer_name) in deleted_layers:
                deleted_layer_names.append(layer_name)
                attn_module.kv_cache = [torch.tensor([])]
        for layer_name in deleted_layer_names:
            del forward_context[layer_name]
        
        # Force garbage collection and clear CUDA cache to release GPU memory
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info(f"after atomic_switch_kv_cache_config_for_layers: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, model runner's kv cache length: {len(self.kv_caches)}")

    def release_kv_cache(self) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"before release_kv_cache: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        # 清除引用
        self.kv_caches = []
        self.kv_cache_config.kv_cache_groups[0].layer_names = []
        forward_context = self.vllm_config.compilation_config.static_forward_context 
        for _, attn_module in forward_context.items():
            attn_module.kv_cache = [torch.tensor([]) for _ in range(self.vllm_config.parallel_config.pipeline_parallel_size)]
        # 强制回收 + 清空缓存
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"after release_kv_cache: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    def compact_kv_cache(self, compacted_length: int, bitmap: bitarray) -> None:
        time_start = time.time()
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        logger.info(f"start to compact kv cache for layers {self.model.model.start_layer} to {self.model.model.end_layer}")
        num_blocks = len(bitmap)
        assert num_blocks == len(self.key_caches), f"bitmap length mismatch: num_blocks: {num_blocks} != kv_cache_tensor_length: {len(self.kv_caches[0][0])}"

        def is_used(idx):
            return bitmap[idx]
        migrate_record: dict[int, int] = {}
        if self.vllm_config.dynamic_config.enable_flexi_flash_attn:
            compact_cache_with_record(self._migrate_block_by_swapping_ptrs, is_used, compacted_length, num_blocks, migrate_record)
            # After swapping tensor references in Python lists, we must update the GPU pointer arrays
            # because prepare_flexi_kv_ptrs caches the data_ptr() of each tensor on GPU
            forward_context = self.vllm_config.compilation_config.static_forward_context
            for layer_name, attn_module in forward_context.items():
                idx = extract_layer_index(layer_name) - self.model.model.start_layer
                # Free old GPU pointer arrays to avoid memory leak
                old_k_ptrs, old_v_ptrs = self.key_cache_ptrs[idx], self.value_cache_ptrs[idx]
                kv_allocator.free_page_list(old_k_ptrs, self.device)
                kv_allocator.free_page_list(old_v_ptrs, self.device)
                # Allocate new GPU pointer arrays with updated tensor addresses
                self.key_cache_ptrs[idx], self.value_cache_ptrs[idx] = kv_allocator.prepare_flexi_kv_ptrs(
                    self.key_caches[idx], self.value_caches[idx])
                assert isinstance(attn_module, FlexiAttention)
                attn_module.key_dev_ptr = self.key_cache_ptrs[idx]
                attn_module.value_dev_ptr = self.value_cache_ptrs[idx]
            logger.info(f"Updated GPU pointer arrays after compact")
        else:
            compact_cache_with_record(self._migrate_block_by_copy_data, is_used, compacted_length, num_blocks, migrate_record)
        logger.info(f"[debug]: migrate_record: {migrate_record}")

        # 遍历CachedRequestState更新block_ids
        for req_state in self.requests.values():
            for i, block_ids in enumerate(req_state.block_ids):
                for j, block_id in enumerate(block_ids):
                    if block_id in migrate_record:
                        req_state.block_ids[i][j] = migrate_record[block_id]
        # 遍历InputBatch中的block_table更新block_ids
        for block_table in self.input_batch.block_table:
            for row in block_table.block_table_np:
                for idx, block_id in enumerate(row):
                    if block_id in migrate_record:
                        row[idx] = migrate_record[block_id]
        
        # # ===== 关键修复：将更新后的 block table 同步到 GPU =====
        # # block_table_np 的修改已自动同步到 block_table_cpu (numpy view)
        # # 但必须显式调用 commit() 将 CPU tensor 拷贝到 GPU tensor
        # num_reqs = len(self.requests)
        # logger.info(f"[debug]: committing block table updates to GPU for {num_reqs} requests")
        # self.input_batch.block_table.commit(num_reqs)
        # logger.info(f"[debug]: block table committed to GPU")

        torch.cuda.synchronize()
        time_end = time.time()
        logger.info(f"compacted kv cache in {time_end - time_start} seconds")

    def resize_kv_cache(self, new_length: int) -> None:
        # 还没有实现在无enginelock情况下的resize_kv_cache
        # assert False
        with self.forward_lock:
            assert isinstance(self.model, DynamicQwen3ForCausalLM)
            logger.info(f"resizing kv cache from {len(self.kv_caches[0][0])} to {new_length}")
            time_start = time.time()
            forward_context = self.vllm_config.compilation_config.static_forward_context
            kv, kv_length, T, H, Dh = self.kv_caches[0].shape
            logger.info(f"num of kv tensors{len(self.kv_caches)}")
            for layer_name, attn_module in forward_context.items():
                logger.info(f"resizing kv cache for layer {layer_name}")
                idx = extract_layer_index(layer_name) - self.model.model.start_layer
                cache = self.kv_caches[idx]
                # 使用 zeros 而不是 empty 来避免未初始化的数据导致错误生成EOS
                tmp_cache = torch.zeros((kv, new_length, T, H, Dh), device=self.device, dtype=cache.dtype)
                logger.info(f"tmp_cache shape: {tmp_cache.shape}, cache shape: {cache.shape}")
                if new_length > kv_length:
                    # 只复制旧的有效部分，新增的部分已经是0了
                    tmp_cache[:, :kv_length, ...].copy_(cache[:, :kv_length, ...], non_blocking=True)
                else:
                    tmp_cache[:, :new_length, ...].copy_(cache[:, :new_length, ...], non_blocking=True)
                self.kv_caches[idx] = tmp_cache
                attn_module.kv_cache = [tmp_cache]
            time_end = time.time()

    # def flexi_resize_kv_cache(self, new_length: int) -> None:
    #     assert isinstance(self.model, DynamicQwen3ForCausalLM)
    #     logger.info(f"resizing kv cache from {len(self.key_caches)} to {new_length}")
    #     logger.info(f"before resize kv cache, available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")
    #     time_start = time.time()
    #     forward_context = self.vllm_config.compilation_config.static_forward_context
    #     T, H, Dh = self.key_caches[0][0].shape
    #     cache_length = len(self.key_caches[0])
    #     logger.info(f"num of kv tensors{cache_length}")

    #     if new_length < cache_length:
    #         for layer_name, attn_module in forward_context.items():
    #             logger.info(f"resizing kv cache for layer {layer_name}")
    #             idx = extract_layer_index(layer_name) - self.model.model.start_layer
    #             key_cache = self.key_caches[idx][:new_length]
    #             value_cache = self.value_caches[idx][:new_length]
    #             self.key_caches[idx] = key_cache
    #             self.value_caches[idx] = value_cache
    #             # Free old GPU pointer arrays before allocating new ones
    #             old_k_ptrs, old_v_ptrs = self.key_cache_ptrs[idx], self.value_cache_ptrs[idx]
    #             free_flexi_kv_ptrs(old_k_ptrs, old_v_ptrs)
    #             self.key_cache_ptrs[idx], self.value_cache_ptrs[idx] = prepare_flexi_kv_ptrs(key_cache, value_cache)
    #             assert isinstance(attn_module, FlexiAttention)
    #             attn_module.key_cache = key_cache
    #             attn_module.value_cache = value_cache
    #             attn_module.key_dev_ptr = self.key_cache_ptrs[idx]
    #             attn_module.value_dev_ptr = self.value_cache_ptrs[idx]
    #     elif new_length > cache_length:
    #         extended_kv_cache_shape = (T, H, Dh)
    #         for layer_name, attn_module in forward_context.items():
    #             new_allocated_block_num = new_length - cache_length
    #             new_allocated_key_cache, new_allocated_value_cache = get_flexi_kv_cache( new_allocated_block_num, extended_kv_cache_shape, self.kv_cache_dtype, self.device)
    #             idx = extract_layer_index(layer_name) - self.model.model.start_layer
    #             key_cache = self.key_caches[idx]
    #             value_cache = self.value_caches[idx]
    #             key_cache.extend(new_allocated_key_cache)
    #             value_cache.extend(new_allocated_value_cache)
    #             # Free old GPU pointer arrays before allocating new ones
    #             old_k_ptrs, old_v_ptrs = self.key_cache_ptrs[idx], self.value_cache_ptrs[idx]
    #             free_flexi_kv_ptrs(old_k_ptrs, old_v_ptrs)
    #             self.key_cache_ptrs[idx], self.value_cache_ptrs[idx] = prepare_flexi_kv_ptrs(key_cache, value_cache)
    #             assert isinstance(attn_module, FlexiAttention)
    #             attn_module.key_cache = key_cache
    #             attn_module.value_cache = value_cache
    #             attn_module.key_dev_ptr = self.key_cache_ptrs[idx]
    #             attn_module.value_dev_ptr = self.value_cache_ptrs[idx]
            
    #     time_end = time.time()
    #     logger.info(f"resized kv cache in {human_readable_duration(time_end - time_start)} seconds")
    #     logger.info(f"resized kv cache ,available gpu memory: {torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")

    def _migrate_block_by_copy_data(self, old_block_id: int, new_block_id: int, migrate_record: dict[int, int]):
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        assert len(self.kv_caches) != 0
        for cache in self.kv_caches:
            # loop for k and v tensor
            for i in range(2):
                cache[i][new_block_id].copy_(cache[i][old_block_id])
        migrate_record[old_block_id] = new_block_id

    def _migrate_block_by_swapping_ptrs(self, old_block_id: int, new_block_id: int, migrate_record: dict[int, int]):
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        assert len(self.key_caches) != 0
        assert len(self.key_cache_ptrs) != 0
        for key_cache in self.key_caches:
            key_cache[new_block_id], key_cache[old_block_id] = key_cache[ old_block_id], key_cache[ new_block_id]
        for value_cache in self.value_caches:
            value_cache[ new_block_id], value_cache[ old_block_id] = value_cache[ old_block_id], value_cache[ new_block_id]
        migrate_record[old_block_id] = new_block_id

    def _update_start_layer(self) -> None:
        assert isinstance(self.model, DynamicQwen3ForCausalLM)
        forward_context: dict[str, "Attention"] = self.vllm_config.compilation_config.static_forward_context
        new_start = min(extract_layer_index(layer_name) for layer_name in forward_context.keys())
        start_layer = self.model.model.start_layer

        # Sync model start if left-padded
        if new_start != start_layer:
            self.model.model.start_layer = new_start
    
    def get_flexi_kv_cache_from_gathered_kv_tensor(self, slot_mapping: torch.Tensor, layer: int,
                                 gathered_kv_tensor: torch.Tensor,
                                 block_num: int, stream: Optional[torch.cuda.Stream] = None) -> Tuple[list[int], list[int], int, int]:
        '''
        Based on current kv cache list shape and dtype, we allocate a new kv cache list
        and extract the data from gathered_kv_tensor to the new kv cache list.
        after that, we also flush the new kv cache to GPU ptrs.
        '''
        time_start = time.time()
        kv_cache_shape = self.kv_cache_shape
        assert len(kv_cache_shape) == 5 # (2, nkvblocks, blockdim, n_head, headdim)
        block_shape = kv_cache_shape[2:]
        kv_dtype = self.kv_cache_dtype
        with torch.cuda.stream(stream):
            key_cache, value_cache, key_cache_page_list, value_cache_page_list, _ = kv_allocator.allocate_with_cuda_async(block_num, list(block_shape), kv_dtype, self.device)
            logger.info(f"gathered_kv_tensor shape: {gathered_kv_tensor.shape}, kv_cache_shape: {kv_cache_shape}, block_shape:{block_shape}, block_num:{block_num}, key_cache length:{len(key_cache)}, value_cache length:{len(value_cache)}, slot_mapping shape:{slot_mapping.shape}, time spent: {human_readable_duration(time.time() - time_start)}")
            logger.info(f"key_cache_list_ptr {key_cache_page_list}, value cache list ptr:{value_cache_page_list}")
            # dummy_scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            assert gathered_kv_tensor.shape[-2:] == kv_cache_shape[-2:], f"gathered_kv_tensor shape {gathered_kv_tensor.shape} mismatch kv_cache_shape {kv_cache_shape}"
            model = self.model
            assert isinstance(model, DynamicQwen3ForCausalLM)
            # DynamicQwen3ForCausalLM.model is DynamicQwen3Model which has .layers
            layer_module = model.model.layers[layer]
            logger.info(f"before gather kv tensor using kernal, device:{torch.cuda.current_device()}, stream:{torch.cuda.current_stream()}")
            # Only call the kernel if there are tokens to process - empty slot_mapping causes CUDA error
            if slot_mapping.numel() > 0:
                flexi_reshape_and_cache_flash(gathered_kv_tensor[0], gathered_kv_tensor[1], key_cache_page_list,value_cache_page_list,  self.page_meta, self.page_meta, slot_mapping,"auto",layer_module.self_attn.attn._k_scale, layer_module.self_attn.attn._v_scale)

        return key_cache, value_cache, key_cache_page_list, value_cache_page_list


# def bind_kv_cache_for_layers(
#     start_layer_index: int,
#     kv_caches: dict[str, torch.Tensor],
#     forward_context: dict[str, "Attention"],
#     runner_kv_caches: list[torch.Tensor],
# ) -> int:
#     """
#     Bind the allocated KV cache of layers to ModelRunner and forward context.
#     """
#     # Bind kv_caches to ModelRunner (support discrete additions with padding)
#     # Convert kv_caches dict to a mapping indexed by global layer index.
#     index2name = defaultdict(list)
#     for layer_name in kv_caches:
#         layer_index = extract_layer_index(layer_name)
#         index2name[layer_index].append(layer_name)

#     # Extend the current runner's kv cache list
#     if index2name:
#         min_index, max_index = min(index2name.keys()), max(index2name.keys())
#         if min_index < start_layer_index:
#             pad_left = start_layer_index - min_index
#             runner_kv_caches[:0] = [torch.tensor([])] * pad_left
#             start_layer_index = min_index

#         max_local_index = max_index - start_layer_index
#         if max_local_index >= len(runner_kv_caches):
#             pad_right = max_local_index + 1 - len(runner_kv_caches)
#             runner_kv_caches.extend([torch.tensor([])] * pad_right)

#     for layer_index in sorted(index2name.keys()):
#         layer_names = index2name[layer_index]
#         if len(layer_names) > 1:
#             # One typical case is encoder-decoder model, e.g., bart.
#             # The cross attention and self attention in the same decoder layer
#             # has different layer_name but the same layer_index.
#             raise NotImplementedError
#         layer_name = layer_names[0]
#         local_index = layer_index - start_layer_index
#         runner_kv_caches[local_index] = kv_caches[layer_name]

#     # Bind kv_caches to forward context
#     for layer_name, kv_cache in kv_caches.items():
#         # NOTE: Use list because of v0 PP virtual engine.
#         forward_context[layer_name].kv_cache = [kv_cache]
#     # Return possibly updated start_layer_index (if left-padded)
#     return start_layer_index

def _debug_tensor_referrers(tensor, max_referrers=10):
    """打印引用该 tensor 的对象及可能的变量名或容器索引。"""
    logger.info("=" * 60)
    logger.info(f"🧠 Tensor device: {tensor.device}, shape: {tensor.shape}, dtype: {tensor.dtype}")
    logger.info(f"📦 Tensor refcount: {sys.getrefcount(tensor) - 1}")

    referrers = gc.get_referrers(tensor)
    logger.info(f"🔎 Found {len(referrers)} referrers for tensor (showing up to {max_referrers}):")

    for i, ref in enumerate(referrers[:max_referrers]):
        logger.info(f"{i}: type={type(ref)}, id={id(ref)}")

        # 如果是 dict，例如 locals()/globals() 或模块字段
        if isinstance(ref, dict):
            for k, v in ref.items():
                if v is tensor:
                    logger.info(f"    ↳ 📌 Found in dict key: {repr(k)}")

        # 如果是 list/tuple/set 等容器
        elif isinstance(ref, (list, tuple, set)):
            for idx, item in enumerate(ref):
                if item is tensor:
                    logger.info(f"    ↳ 🧩 Found in {type(ref).__name__} at index {idx}")

        # 如果是某个实例对象
        elif hasattr(ref, "__class__"):
            logger.info(f"    ↳ 🧷 Possibly from instance of: {ref.__class__.__name__}")
    logger.info("=" * 60)
