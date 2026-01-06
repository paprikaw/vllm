# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Optional

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.output_processor import (OutputProcessor)
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import (StatLoggerBase, StatLoggerFactory,
                                     setup_default_loggers)
from .async_llm import AsyncLLM
from .dynamic_core_client import DynamicAsyncMPClient
from ..executor.dynamic_ray_distributed_executor import DynamicRayDistributedExecutor 
from vllm.dynamic_config import MigrationConfig

logger = init_logger(__name__)


class DynamicAsyncLLM(AsyncLLM):
    """
    This class is used to adapt DynamicMPClient 
    """
    def __init__(
        self,
        vllm_config: VllmConfig,
        dynamic_config: MigrationConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
    ) -> None:
        """
        Create the DynamicAsyncLLM.
        The only difference between DynamicAsyncLLM and AsyncLLM is that
        DynamicAsyncLLM uses DynamicMPClient instead of AsyncMPClient.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            use_cached_outputs: Whether to use cached outputs.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.log_requests = log_requests
        self.log_stats = log_stats

        # Set up stat loggers; independent set for each DP rank.
        self.stat_loggers: list[list[StatLoggerBase]] = setup_default_loggers(
            vllm_config=vllm_config,
            log_stats=self.log_stats,
            engine_num=vllm_config.parallel_config.data_parallel_size,
            custom_stat_loggers=stat_loggers,
        )

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            lora_config=vllm_config.lora_config)

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            mm_registry=mm_registry,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(self.tokenizer,
                                                log_stats=self.log_stats)

        # EngineCore (starts the engine in background process).
        assert vllm_config.parallel_config.data_parallel_size == 1, "DynamicAsyncLLM only supports data parallel size 1"

        self.engine_core = DynamicAsyncMPClient(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            dynamic_config=dynamic_config,
        )
        if self.stat_loggers:
            for stat_logger in self.stat_loggers[0]:
                stat_logger.log_engine_initialized()
        self.output_handler: Optional[asyncio.Task] = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

    @classmethod
    def from_vllm_config_with_dynamic_config(
        cls,
        vllm_config: VllmConfig,
        dynamic_config: MigrationConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        disable_log_requests: bool = False,
        disable_log_stats: bool = False,
    ) -> "DynamicAsyncLLM":
        if not envs.VLLM_USE_V1:
            raise ValueError(
                "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False. "
                "This should not happen. As a workaround, try using "
                "AsyncLLMEngine.from_vllm_config(...) or explicitly set "
                "VLLM_USE_V1=0 or 1 and report this issue on Github.")

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            dynamic_config=dynamic_config,
            executor_class=DynamicRayDistributedExecutor,
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=not disable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
        )