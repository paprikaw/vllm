#!/usr/bin/env python3
"""
Log Metrics Analyzer V2
Analyzes various performance metrics from vLLM log files.

Refactored with configuration-driven architecture for easy metric addition.
"""

import re
import argparse
import statistics
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
from enum import Enum
import sys


class MetricType(Enum):
    """Type of metric storage and processing."""
    SIMPLE_LIST = "simple_list"  # Simple list of time values
    KEYED_DICT = "keyed_dict"    # Dict with keys (e.g., layer_id -> times)
    SPECIAL = "special"           # Custom processing (e.g., migration times)


class MetricCategory(Enum):
    """Category for grouping metrics in reports."""
    TIMELINE = "timeline"         # Timeline metrics for migration/KV operations
    MODEL_LOCK = "model_lock"     # Lock acquisition metrics
    FORWARDING = "forwarding"     # Model forwarding metrics  
    COMMUNICATION = "communication"  # Inter-node communication
    MEMORY = "memory"             # Memory allocation metrics
    COMPILED_DAG = "compiled_dag" # Ray CompiledDAG metrics
    OTHER = "other"               # Uncategorized metrics
    STOP_TIME = "stop_time"               # Stop time metrics


@dataclass
class MetricConfig:
    """Configuration for a single metric."""
    name: str
    display_name: str
    log_pattern: str  # Regex pattern to match in log
    time_group: int = 1  # Which regex group contains the time value
    metric_type: MetricType = MetricType.SIMPLE_LIST
    key_group: Optional[int] = None  # For KEYED_DICT, which group is the key
    show_in_report: bool = True
    custom_processor: Optional[Callable] = None  # Custom processing function
    description: str = ""
    category: MetricCategory = MetricCategory.OTHER  # Category for grouping


# ============================================================================
# METRIC DEFINITIONS - ADD NEW METRICS HERE
# ============================================================================
METRIC_CONFIGS = [
    # =========================================================================
    # TIMELINE METRICS - Migration and KV Cache Operations
    # =========================================================================
    MetricConfig(
        name="timeline_migration_total",
        display_name="[Timeline] Migration Process Total Time",
        log_pattern=r'\[timeline\]: migration process time taken: ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.TIMELINE,
        description="Total time for entire migration process"
    ),
    MetricConfig(
        name="timeline_compact_kv_total",
        display_name="[Timeline] Compact KV Cache Total Time",
        log_pattern=r'\[timeline\]: compact kv cache total time taken: ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.TIMELINE,
        description="Total time for KV cache compaction"
    ),
    MetricConfig(
        name="timeline_resize_kv_total",
        display_name="[Timeline] Total Resize KV Cache Time",
        log_pattern=r'\[timeline\]: total resize kv cache time taken: ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.TIMELINE,
        description="Total time for resizing KV cache"
    ),
    MetricConfig(
        name="timeline_weight_loading",
        display_name="[Timeline] After Weight Loading",
        log_pattern=r'\[timeline\]: after weight loading.*?taken: ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.TIMELINE,
        description="Time after weight loading"
    ),
        MetricConfig(
        name="timeline_bind_kv_cache_total",
        display_name="[Timeline] Bind KV Cache Total Time",
        log_pattern=r'\[timeline\]: bind kv cache time taken: ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.TIMELINE,
        description="Total time for binding KV cache"
    ),
    MetricConfig(
        name="timeline_receive_kv_tensor_total",
        display_name="[Timeline] Receive KV Tensor Total Time",
        log_pattern=r'\[timeline\]: receive kv tensor finished, time taken ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.TIMELINE,
        description="Total time for receiving KV tensor"
    ),

    MetricConfig(
        name="timeline_notify_kv_patches",
        display_name="[Timeline] Notify For KV Patches",
        log_pattern=r'\[timeline\]: notify for kv patches took ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.TIMELINE,
        description="Time to notify for KV patches"
    ),
    MetricConfig(
        name="timeline_wait_kv_patch",
        display_name="[Timeline] Wait For KV Patch Preparation",
        log_pattern=r'\[timeline\]: wait for kv patch preparation take ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.TIMELINE,
        description="Time waiting for KV patch preparation"
    ),
    MetricConfig(
        name="timeline_receive_kv_patches",
        display_name="[Timeline] After Receiving KV Cache Patches",
        log_pattern=r'\[timeline\]: after listen to kv cache patches.*?taken: ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.TIMELINE,
        description="Time after listening to KV cache patches"
    ),
    MetricConfig(
        name="timeline_remove_layers",
        display_name="[Timeline] After Remove Layers",
        log_pattern=r'\[timeline\]: after remove layers.*?taken: ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.TIMELINE,
        description="Time after removing layers"
    ),

    MetricConfig(
        name="timeline_add_layers_in_lock",
        display_name="[Timeline] Add Layers Within Lock Time",
        log_pattern=r'\[timeline\]: time within lock when add layers time taken: ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.STOP_TIME,
        description="Time spent within lock when adding layers"
    ),
    MetricConfig(
        name="timeline_engine_locking",
        display_name="[Timeline] Engine Locking Time",
        log_pattern=r'\[timeline\]: engine locking time: ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.STOP_TIME,
        description="Time spent acquiring engine lock during migration"
    ),
        MetricConfig(
        name="timeline_compact_kv_cache_lock",
        display_name="[Timeline] KV Cache Compaction Within Lock",
        log_pattern=r'\[timeline\]: kv cache compaction within lock take ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.STOP_TIME,
        description="Time spent compacting KV cache within lock"
    ),
    MetricConfig(
        name="timeline_before_kv_migration",
        display_name="[Timeline] Before KV Migration (Add Weights, Compact, Resize)",
        log_pattern=r'\[timeline\]: before start actual kv cache migration.*?: ([0-9.eE+-]+\s*[a-zµ]+)',
        metric_type=MetricType.SPECIAL,
        category=MetricCategory.OTHER,
        description="Time before actual KV migration starts"
    ),

    # =========================================================================
    # FORWARDING METRICS  
    # =========================================================================
    MetricConfig(
        name="engine_step_time",
        display_name="Process Engine Step Time",
        log_pattern=r'process the engine step, time: ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.FORWARDING,
    ),
    MetricConfig(
        name="after_layer_forwarding",
        display_name="After Layer Forwarding (All Layers)",
        log_pattern=r'after Layer forwarding took ([0-9.eE+-]+\s*[a-zµ]+), layer (\d+)',
        metric_type=MetricType.KEYED_DICT,
        key_group=2,
        category=MetricCategory.FORWARDING,
        description="Time for forwarding each layer"
    ),
    MetricConfig(
        name="after_forwarding",
        display_name="After Forwarding Took",
        log_pattern=r'after forwarding took ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.FORWARDING,
    ),
    MetricConfig(
        name="attn_forward",
        display_name="Attention Forward Times",
        log_pattern=r'\[timeline\] attention forward took ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.FORWARDING,
        description="Time taken for attention forward"
    ),
    MetricConfig(
        name="attention_forward",
        display_name="Attention Forward Took",
        log_pattern=r'attention forward took ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.FORWARDING,
    ),
    # =========================================================================
    # MODEL LOCK METRICS
    # =========================================================================
    MetricConfig(
        name="getting_model_lock",
        display_name="Getting Model Lock Taking",
        log_pattern=r'getting model lock taking (\S+)',
        category=MetricCategory.MODEL_LOCK,
    ),
    MetricConfig(
        name="getting_forward_lock",
        display_name="Getting Forward Lock Taking",
        log_pattern=r'getting forward lock taking (\S+)',
        category=MetricCategory.MODEL_LOCK,
    ),


    # =========================================================================
    # COMMUNICATION METRICS
    # =========================================================================
    MetricConfig(
        name="communication_time",
        display_name="Communication Time from Upstream",
        log_pattern=r'Communication time from upstream: ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.COMMUNICATION,
    ),

    # =========================================================================
    # MEMORY ALLOCATION METRICS
    # =========================================================================
    MetricConfig(
        name="memory_stress_tester",
        display_name="[Memory Allocation] MemoryStressTester Allocation Time",
        log_pattern=r'MemoryStressTester: cycle #.+alloc_time=([0-9.eE+-]+\s*ms)',
        category=MetricCategory.MEMORY,
    ),
    MetricConfig(
        name="allocate_kv",
        display_name="[Memory Allocation] Asynchronous KV Allocation Time",
        log_pattern=r'async kv allocation: .+alloc_time=([0-9.eE+-]+\s*ms)',
        category=MetricCategory.MEMORY,
    ),
    MetricConfig(
        name="weight_loading",
        display_name="Weight Loading Time (Individual Weights)",
        log_pattern=r'\[Weight Loading\] Loaded weight for .+ took ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.MEMORY,
    ),
    MetricConfig(
        name="weight_loading_lock",
        display_name="Weight Loading Lock Acquisition Time",
        log_pattern=r'\[Weight Loading\] Got lock before loading weight, took ([0-9.eE+-]+\s*[a-zµ]+)',
        category=MetricCategory.MEMORY,
    ),
    
    # =========================================================================
    # COMPILED DAG METRICS
    # =========================================================================
    MetricConfig(
        name="compiled_dag_write_lock",
        display_name="[CompiledDAG] WRITE Forward Lock Acquisition Time",
        log_pattern=r'\[CompiledDAG WRITE\].+time acuiring lock: ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.COMPILED_DAG,
    ),
    MetricConfig(
        name="compiled_dag_read_lock",
        display_name="[CompiledDAG] READ Forward Lock Acquisition Time",
        log_pattern=r'\[CompiledDAG READ\].+time acquiring lock: ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.COMPILED_DAG,
    ),
    MetricConfig(
        name="compiled_dag_write_complete",
        display_name="[CompiledDAG] WRITE Completion Time",
        log_pattern=r'\[CompiledDAG WRITE\].+Write completed, time taken: ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.COMPILED_DAG,
    ),
    MetricConfig(
        name="compiled_dag_read_complete",
        display_name="[CompiledDAG] READ Completion Time",
        log_pattern=r'\[CompiledDAG READ\].+Read completed, time taken: ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.COMPILED_DAG,
    ),
    MetricConfig(
        name="tensor_metadata_recv",
        display_name="[Ray.Comm.Read] Tensor Metadata Receive Time",
        log_pattern=r'received tensor metadata in ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.COMPILED_DAG,
    ),
    MetricConfig(
        name="dynamic_channel_recv",
        display_name="[Ray.Comm.Read] Tensors Received Time",
        log_pattern=r'\[DynamicChannel\] Tensor read completed in ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.COMPILED_DAG,
    ),
    MetricConfig(
        name="nccl_recv_buffer_alloc",
        display_name="[RAY.NCCLGROUP.Read] Recv Buffer Allocation Time",
        log_pattern=r'\[RAY\.NCCLGROUP\] Recv buffer allocated in ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.COMPILED_DAG,
    ),
    MetricConfig(
        name="nccl_recv_data",
        display_name="[RAY.NCCLGROUP.Read] Received Data From Comm Time",
        log_pattern=r'\[RAY\.NCCLGROUP\] received data from comm taking ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.COMPILED_DAG,
    ),
    MetricConfig(
        name="nccl_cuda_sync",
        display_name="[RAY.NCCLGROUP.Read] CUDA Synchronize Time",
        log_pattern=r'\[RAY\.NCCLGROUP\] cuda synchronize taking ([0-9.eE+-]+\s*seconds)',
        category=MetricCategory.COMPILED_DAG,
    ),
    MetricConfig(
        name="nccl_send_queued",
        display_name="[RAY.NCCLGROUP.Send] Send Queued To Peer Time",
        log_pattern=r'\[RAY\.NCCLGROUP\] send queued to peer.+in ([0-9.eE+-]+\s*s), total: ([0-9.eE+-]+\s*s)',
        category=MetricCategory.COMPILED_DAG,
    ),
]

# Category display names and order
CATEGORY_DISPLAY = {
    MetricCategory.TIMELINE: "📊 TIMELINE METRICS (Migration & KV Operations)",
    MetricCategory.STOP_TIME: "⏱️ STOP TIME METRICS (Service Interruption)",
    MetricCategory.MODEL_LOCK: "🔒 MODEL LOCK METRICS",
    MetricCategory.FORWARDING: "⏩ FORWARDING METRICS",
    MetricCategory.COMMUNICATION: "📡 COMMUNICATION METRICS",
    MetricCategory.MEMORY: "💾 MEMORY ALLOCATION METRICS",
    MetricCategory.COMPILED_DAG: "🔄 COMPILED DAG METRICS",
    MetricCategory.OTHER: "📋 OTHER METRICS",
}

CATEGORY_ORDER = [
    MetricCategory.TIMELINE,
    MetricCategory.STOP_TIME,
    MetricCategory.MODEL_LOCK,
    MetricCategory.FORWARDING,
    MetricCategory.COMMUNICATION,
    MetricCategory.MEMORY,
    MetricCategory.COMPILED_DAG,
    MetricCategory.OTHER,
]


class LogMetricsAnalyzer:
    """Analyzer for extracting and analyzing metrics from log files."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.metrics: Dict[str, Any] = {}
        self.metric_configs: Dict[str, MetricConfig] = {}
        
        # Initialize storage for each metric
        for config in METRIC_CONFIGS:
            self.metric_configs[config.name] = config
            if config.metric_type == MetricType.SIMPLE_LIST or config.metric_type == MetricType.SPECIAL:
                self.metrics[config.name] = []
            elif config.metric_type == MetricType.KEYED_DICT:
                self.metrics[config.name] = defaultdict(list)
    
    def parse_time_with_unit(self, time_str: str) -> Tuple[Optional[float], str]:
        """
        Parse time string with unit and convert to microseconds for comparison.
        Returns: (time_in_microseconds, original_string)
        """
        time_str = time_str.strip()
        
        # Match patterns like "123µs", "45ms", "1.23 seconds", "1.4e-05s" (scientific notation)
        match = re.match(r'([0-9.eE+-]+)\s*(µs|ms|seconds?|s)', time_str)
        if not match:
            return None, time_str
        
        value = float(match.group(1))
        unit = match.group(2)
        
        # Convert to microseconds
        if unit in ['µs']:
            time_us = value
        elif unit == 'ms':
            time_us = value * 1000
        elif unit in ['seconds', 'second', 's']:
            time_us = value * 1000000
        else:
            time_us = value
        
        return time_us, time_str
    
    def analyze_file(self):
        """Parse the log file and extract all metrics."""
        print(f"Analyzing log file: {self.log_file}")
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Process each metric config
                for config in METRIC_CONFIGS:
                    match = re.search(config.log_pattern, line)
                    if match:
                        # Check if there are any groups to extract
                        if match.lastindex is None or match.lastindex < config.time_group:
                            # Pattern matched but no capturing group for time - use marker
                            time_value = "matched"
                        else:
                            # Extract time value
                            time_value = match.group(config.time_group)
                        
                        # Store based on metric type
                        if config.metric_type == MetricType.SIMPLE_LIST or config.metric_type == MetricType.SPECIAL:
                            self.metrics[config.name].append(time_value)
                        elif config.metric_type == MetricType.KEYED_DICT:
                            if match.lastindex and match.lastindex >= config.key_group:
                                key = int(match.group(config.key_group))
                                self.metrics[config.name][key].append(time_value)
        
        print(f"Finished parsing log file.\n")
    
    def print_distribution(self, metric_name: str, display_name: str, values: List[str], top_n: int = 20):
        """Print distribution statistics for a metric."""
        if not values:
            print(f"  No data found for {display_name}")
            return
        
        print(f"\n{'='*80}")
        print(f"Metric: {display_name}")
        print(f"Total count: {len(values)}")
        print(f"{'='*80}")
        
        # Format time with appropriate units (moved here so it's always available)
        def format_time(us):
            if us < 1000:
                return f"{us:.2f}µs"
            elif us < 1000000:
                return f"{us/1000:.2f}ms"
            else:
                return f"{us/1000000:.2f}s"
        
        # Count occurrences and collect all time values in microseconds
        counter = Counter(values)
        all_times_us = []
        
        # Sort by value (converting to microseconds for proper numerical sorting)
        sorted_items = []
        for value, count in counter.items():
            time_us, original = self.parse_time_with_unit(value)
            if time_us is not None:
                sorted_items.append((time_us, original, count))
                # Add to all_times_us list (repeat by count for proper statistics)
                all_times_us.extend([time_us] * count)
            else:
                sorted_items.append((0, original, count))
        
        sorted_items.sort(key=lambda x: x[0], reverse=True)
        
        # Calculate and print statistics (average and median)
        if all_times_us:
            avg_us = statistics.mean(all_times_us)
            median_us = statistics.median(all_times_us)
            min_us = min(all_times_us)
            max_us = max(all_times_us)
            
            print(f"\nStatistics:")
            print(f"  Average: {format_time(avg_us)}")
            print(f"  Median:  {format_time(median_us)}")
            print(f"  Min:     {format_time(min_us)}")
            print(f"  Max:     {format_time(max_us)}")
        
        # Print top N
        print(f"\nTop {min(top_n, len(sorted_items))} longest times:")
        print(f"{'Rank':<6} {'Time':<15} {'Count':<10} {'Percentage':<12}")
        print("-" * 50)
        for i, (time_us, original, count) in enumerate(sorted_items[:top_n], 1):
            percentage = (count / len(values)) * 100
            # Format time nicely instead of showing original raw string
            formatted_time = format_time(time_us) if time_us is not None else original
            print(f"{i:<6} {formatted_time:<15} {count:<10} {percentage:>6.2f}%")
        
        # Print compact distribution summary (most common values)
        print(f"\nDistribution Summary (Top {min(20, len(sorted_items))} most frequent):")
        # Sort by count (frequency) instead of time
        sorted_by_freq = sorted(sorted_items, key=lambda x: x[2], reverse=True)[:20]
        print(f"{'Time':<15} {'Count':<10} {'Percentage':<12}")
        print("-" * 40)
        for time_us, original, count in sorted_by_freq:
            percentage = (count / len(values)) * 100
            # Format time nicely instead of showing original raw string
            formatted_time = format_time(time_us) if time_us is not None else original
            print(f"{formatted_time:<15} {count:<10} {percentage:>6.2f}%")
        
        # Print statistics summary
        total_unique_values = len(sorted_items)
        if total_unique_values > 20:
            print(f"\n... and {total_unique_values - 20} more unique values (showing top 20 by frequency)")
    
    def print_layer_distribution(self, metric_name: str, display_name: str, layer_data: Dict[int, List[str]], top_n: int = 10):
        """Print distribution statistics for per-layer metrics (aggregated across all layers)."""
        if not layer_data:
            print(f"  No data found for {display_name}")
            return
        
        # Aggregate all layer data into a single list
        all_values = []
        for layer_id, values in layer_data.items():
            all_values.extend(values)
        
        # Now use the standard distribution print
        self.print_distribution(metric_name, display_name, all_values, top_n)
    
    def print_special_metric(self, metric_name: str, display_name: str, values: List[str]):
        """Print special metrics (like migration times) as a simple list."""
        if not values:
            return
        
        print(f"\n{'='*80}")
        print(f"{display_name.upper()}")
        print(f"{'='*80}")
        print(f"Total events: {len(values)}")
        for i, time in enumerate(values, 1):
            print(f"  Event {i}: {time}")
        print()
    
    def generate_report(self, top_n: int = 20):
        """Generate complete analysis report grouped by category."""
        print("\n" + "="*80)
        print("VLLM LOG METRICS ANALYSIS REPORT")
        print("="*80)
        
        # Group metrics by category
        metrics_by_category: Dict[MetricCategory, List[MetricConfig]] = defaultdict(list)
        for config in METRIC_CONFIGS:
            if config.show_in_report:
                metrics_by_category[config.category].append(config)
        
        # Process metrics in category order
        for category in CATEGORY_ORDER:
            configs = metrics_by_category.get(category, [])
            
            # Check if any metrics in this category have data
            has_data = False
            for config in configs:
                metric_data = self.metrics.get(config.name)
                if metric_data:
                    has_data = True
                    break
            
            if not has_data:
                continue
            
            # Print category header
            print("\n" + "#"*80)
            print(f"# {CATEGORY_DISPLAY.get(category, category.value)}")
            print("#"*80)
            
            # Process each metric in this category
            for config in configs:
                metric_data = self.metrics.get(config.name)
                if not metric_data:
                    continue
                
                # Print based on metric type
                if config.metric_type == MetricType.SPECIAL:
                    self.print_special_metric(config.name, config.display_name, metric_data)
                elif config.metric_type == MetricType.KEYED_DICT:
                    self.print_layer_distribution(config.name, config.display_name, metric_data, top_n)
                else:  # SIMPLE_LIST
                    self.print_distribution(config.name, config.display_name, metric_data, top_n)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze performance metrics from vLLM log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a log file with default settings (top 20)
  python analyze_log_metrics_v2.py /path/to/logfile.log
  
  # Show top 50 longest times
  python analyze_log_metrics_v2.py /path/to/logfile.log --top 50
        """
    )
    
    parser.add_argument('log_file', help='Path to the log file to analyze')
    parser.add_argument('--top', '-t', type=int, default=20,
                       help='Number of top entries to show (default: 20)')
    
    args = parser.parse_args()
    
    try:
        analyzer = LogMetricsAnalyzer(args.log_file)
        analyzer.analyze_file()
        analyzer.generate_report(top_n=args.top)
    except FileNotFoundError:
        print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
