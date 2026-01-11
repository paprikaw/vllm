#!/usr/bin/env python3
"""
Log Metrics Analyzer
Analyzes various performance metrics from vLLM log files.
"""

import re
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import sys


class LogMetricsAnalyzer:
    """Analyzer for extracting and analyzing metrics from log files."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.migration_times = []  # Store migration process times
        self.engine_locking_times = []  # Store engine locking times
        self.compact_kv_cache_times = []  # Store compact kv cache times
        self.bind_kv_cache_times = []  # Store bind kv cache times
        self.memory_stress_tester_times = []  # Store MemoryStressTester allocation times
        self.compiled_dag_write_lock_times = []  # Store CompiledDAG WRITE lock acquisition times
        self.compiled_dag_write_complete_times = []  # Store CompiledDAG WRITE completion times
        self.compiled_dag_read_lock_times = []  # Store CompiledDAG READ lock acquisition times
        self.compiled_dag_read_complete_times = []  # Store CompiledDAG READ completion times
        self.dynamic_channel_recv_times = []  # Store DynamicChannel tensor receive times
        self.weight_loading_lock_times = []  # Store lock acquisition times before loading weights
        self.recv_buffer_alloc_times = []  # Store recv buffer allocation times
        self.dynamic_channel_recv_with_fbgate_times = []  # Store DynamicChannel tensor(s) received times with fbgate
        self.tensor_metadata_recv_times = []  # Store tensor metadata receive times
        self.nccl_recv_buffer_alloc_times = []  # Store RAY.NCCLGROUP recv buffer allocation times
        self.nccl_recv_data_times = []  # Store RAY.NCCLGROUP received data from comm times
        self.nccl_cuda_sync_times = []  # Store RAY.NCCLGROUP cuda synchronize times
        self.nccl_send_queued_times = []  # Store RAY.NCCLGROUP send queued to peer times
        self.allocate_kv_times = []
        self.weight_loading_times = []  # Store weight loading times for individual weights
        self.metrics = {
            'getting_model_lock': [],
            'getting_forward_lock': [],
            'engine_step_time': [],
            'after_layer_forwarding': defaultdict(list),  # layer_id -> times
            'after_forwarding': [],
            'communication_time': [],
            'attention_forward': []
        }
    
    def parse_time_with_unit(self, time_str: str) -> Tuple[float, str]:
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
                # 0. Migration process time (capture first)
                if '[timeline]: migration process time taken:' in line:
                    match = re.search(r'migration process time taken: ([0-9.]+[a-zµ]+)', line)
                    if match:
                        self.migration_times.append(match.group(1))
                
                # 0a. Engine locking time
                if '[timeline]: engine locking time:' in line:
                    match = re.search(r'engine locking time: ([0-9.]+[a-zµ]+)', line)
                    if match:
                        self.engine_locking_times.append(match.group(1))
                
                # 0b. Compact kv cache time
                if '[timeline]: compact kv cache take:' in line:
                    match = re.search(r'compact kv cache take: ([0-9.]+[a-zµ]+)', line)
                    if match:
                        self.compact_kv_cache_times.append(match.group(1))
                
                # 0c. Bind kv cache time
                if '[timeline]: bind single kv cache for layer' in line:
                    match = re.search(r'bind single kv cache for layer .+ take ([0-9.]+[a-zµ]+)', line)
                    if match:
                        self.bind_kv_cache_times.append(match.group(1))
                
                # 1. getting model lock taking
                if 'getting model lock taking' in line:
                    match = re.search(r'getting model lock taking (\S+)', line)
                    if match:
                        self.metrics['getting_model_lock'].append(match.group(1))
                
                # 2. getting forward lock taking
                if 'getting forward lock taking' in line:
                    match = re.search(r'getting forward lock taking (\S+)', line)
                    if match:
                        self.metrics['getting_forward_lock'].append(match.group(1))
                
                # 3. process the engine step, time
                if 'process the engine step, time:' in line:
                    match = re.search(r'process the engine step, time: ([0-9.]+) seconds', line)
                    if match:
                        self.metrics['engine_step_time'].append(f"{match.group(1)}s")
                
                # 4. after Layer forwarding took (with layer number)
                if 'after Layer forwarding took' in line:
                    match = re.search(r'after Layer forwarding took ([0-9.]+[a-zµ]+), layer (\d+)', line)
                    if match:
                        time_str = match.group(1)
                        layer_id = int(match.group(2))
                        self.metrics['after_layer_forwarding'][layer_id].append(time_str)
                
                # 5. after forwarding took (general)
                if 'after forwarding took' in line and 'after Layer forwarding took' not in line:
                    match = re.search(r'after forwarding took ([0-9.]+[a-zµ]+)', line)
                    if match:
                        self.metrics['after_forwarding'].append(match.group(1))
                
                # 6. Communication time from upstream
                if 'Communication time from upstream:' in line:
                    match = re.search(r'Communication time from upstream: ([0-9.]+) ms', line)
                    if match:
                        self.metrics['communication_time'].append(f"{match.group(1)}ms")
                
                # 7. attention forward took
                if 'attention forward took ' in line:
                    match = re.search(r'attention forward took ([0-9.]+[a-zµ]+)', line)
                    if match:
                        self.metrics['attention_forward'].append(match.group(1))
                
                # 8. MemoryStressTester allocation time
                if 'MemoryStressTester: cycle #' in line:
                    match = re.search(r'alloc_time=([0-9.]+)ms', line)
                    if match:
                        self.memory_stress_tester_times.append(f"{match.group(1)}ms")

                # 8a. Async KV allocation time
                if 'async kv allocation: ' in line:
                    match = re.search(r'alloc_time=([0-9.]+)ms', line)
                    if match:
                        self.allocate_kv_times.append(f"{match.group(1)}ms")

                # 9. CompiledDAG WRITE lock acquisition time
                if '[CompiledDAG WRITE]' in line and 'time acuiring lock:' in line:
                    match = re.search(r'time acuiring lock: ([0-9.]+) seconds', line)
                    if match:
                        self.compiled_dag_write_lock_times.append(f"{match.group(1)}s")
                
                # 10. CompiledDAG WRITE completion time
                if '[CompiledDAG WRITE]' in line and 'Write completed, time taken:' in line:
                    match = re.search(r'time taken: ([0-9.]+) seconds', line)
                    if match:
                        self.compiled_dag_write_complete_times.append(f"{match.group(1)}s")
                
                # 11. CompiledDAG READ lock acquisition time
                if '[CompiledDAG READ]' in line and 'time acquiring lock:' in line:
                    match = re.search(r'time acquiring lock: ([0-9.]+) seconds', line)
                    if match:
                        self.compiled_dag_read_lock_times.append(f"{match.group(1)}s")
                
                # 12. CompiledDAG READ completion time
                if '[CompiledDAG READ]' in line and 'Read completed, time taken:' in line:
                    match = re.search(r'time taken: ([0-9.]+) seconds', line)
                    if match:
                        self.compiled_dag_read_complete_times.append(f"{match.group(1)}s")
                
                # 13. DynamicChannel tensor receive time
                if '[DynamicChannel] Tensor read completed in' in line:
                    match = re.search(r'Tensor read completed in ([0-9.]+) seconds', line)
                    if match:
                        self.dynamic_channel_recv_times.append(f"{match.group(1)}s")
                
                # 14. Got lock before loading weight
                if 'Got lock before loading weight for' in line and 'taking:' in line:
                    match = re.search(r'taking: ([0-9.]+[a-zµ]+)', line)
                    if match:
                        self.weight_loading_lock_times.append(match.group(1))
                
                # 15. Recv buffer allocation time
                if 'Recv buffer allocated in' in line:
                    match = re.search(r'allocated in ([0-9.eE+-]+) seconds', line)
                    if match:
                        self.recv_buffer_alloc_times.append(f"{match.group(1)}s")
                
                # 16. DynamicChannel tensor(s) received with fbgate
                if '[DynamicChannel]' in line and 'tensor(s) received in' in line and '(with fbgate)' in line:
                    match = re.search(r'received in ([0-9.]+) seconds', line)
                    if match:
                        self.dynamic_channel_recv_with_fbgate_times.append(f"{match.group(1)}s")
                
                # 17. Received tensor metadata time
                if 'received tensor metadata in' in line:
                    match = re.search(r'metadata in ([0-9.eE+-]+) seconds', line)
                    if match:
                        self.tensor_metadata_recv_times.append(f"{match.group(1)}s")
                
                # 18. RAY.NCCLGROUP Recv buffer allocated
                if '[RAY.NCCLGROUP] Recv buffer allocated in' in line:
                    match = re.search(r'allocated in ([0-9.eE+-]+) seconds', line)
                    if match:
                        self.nccl_recv_buffer_alloc_times.append(f"{match.group(1)}s")
                
                # 19. RAY.NCCLGROUP received data from comm
                if '[RAY.NCCLGROUP] received data from comm taking' in line:
                    match = re.search(r'taking ([0-9.eE+-]+) seconds', line)
                    if match:
                        self.nccl_recv_data_times.append(f"{match.group(1)}s")
                
                # 20. RAY.NCCLGROUP cuda synchronize
                if '[RAY.NCCLGROUP] cuda synchronize taking' in line:
                    match = re.search(r'taking ([0-9.eE+-]+) seconds', line)
                    if match:
                        self.nccl_cuda_sync_times.append(f"{match.group(1)}s")
                
                # 21. RAY.NCCLGROUP send queued to peer
                if '[RAY.NCCLGROUP] send queued to peer' in line:
                    match = re.search(r'in ([0-9.eE+-]+)s, total: ([0-9.eE+-]+)s', line)
                    if match:
                        self.nccl_send_queued_times.append(f"{match.group(1)}s (total: {match.group(2)}s)")
                
                # 22. Weight loading time for individual weights
                if '[Weight Loading] Loaded weight for' in line and 'took' in line:
                    match = re.search(r'Loaded weight for .+ took ([0-9.]+[a-zµ]+)', line)
                    if match:
                        self.weight_loading_times.append(match.group(1))        
        print(f"Finished parsing log file.\n")
    
    def print_distribution(self, metric_name: str, values: List[str], top_n: int = 20):
        """Print distribution statistics for a metric."""
        if not values:
            print(f"  No data found for {metric_name}")
            return
        
        print(f"\n{'='*80}")
        print(f"Metric: {metric_name}")
        print(f"Total count: {len(values)}")
        print(f"{'='*80}")
        
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
            import statistics
            avg_us = statistics.mean(all_times_us)
            median_us = statistics.median(all_times_us)
            min_us = min(all_times_us)
            max_us = max(all_times_us)
            
            # Format with appropriate units
            def format_time(us):
                if us < 1000:
                    return f"{us:.2f}µs"
                elif us < 1000000:
                    return f"{us/1000:.2f}ms"
                else:
                    return f"{us/1000000:.2f}s"
            
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
            print(f"{i:<6} {original:<15} {count:<10} {percentage:>6.2f}%")
        
        # Print compact distribution summary (most common values)
        print(f"\nDistribution Summary (Top {min(20, len(sorted_items))} most frequent):")
        # Sort by count (frequency) instead of time
        sorted_by_freq = sorted(sorted_items, key=lambda x: x[2], reverse=True)[:20]
        print(f"{'Time':<15} {'Count':<10} {'Percentage':<12}")
        print("-" * 40)
        for time_us, original, count in sorted_by_freq:
            percentage = (count / len(values)) * 100
            print(f"{original:<15} {count:<10} {percentage:>6.2f}%")
        
        # Print statistics summary
        total_unique_values = len(sorted_items)
        if total_unique_values > 20:
            print(f"\n... and {total_unique_values - 20} more unique values (showing top 20 by frequency)")
    
    def print_layer_distribution(self, metric_name: str, layer_data: Dict[int, List[str]], top_n: int = 10):
        """Print distribution statistics for per-layer metrics (aggregated across all layers)."""
        if not layer_data:
            print(f"  No data found for {metric_name}")
            return
        
        # Aggregate all layer data into a single list
        all_values = []
        for layer_id, values in layer_data.items():
            all_values.extend(values)
        
        # Now use the standard distribution print
        self.print_distribution(f"{metric_name} (All Layers)", all_values, top_n)
    
    def generate_report(self, top_n: int = 20):
        """Generate complete analysis report."""
        print("\n" + "="*80)
        print("VLLM LOG METRICS ANALYSIS REPORT")
        print("="*80)
        
        # Print migration times first (if any)
        if self.migration_times:
            print("\n" + "="*80)
            print("MIGRATION PROCESS TIMES")
            print("="*80)
            print(f"Total migrations detected: {len(self.migration_times)}")
            for i, time in enumerate(self.migration_times, 1):
                print(f"  Migration {i}: {time}")
            print()
        
        # Print engine locking times
        if self.engine_locking_times:
            print("\n" + "="*80)
            print("ENGINE LOCKING TIMES (During Migration)")
            print("="*80)
            print(f"Total events: {len(self.engine_locking_times)}")
            for i, time in enumerate(self.engine_locking_times, 1):
                print(f"  Event {i}: {time}")
            print()
        
        # Print compact kv cache times
        if self.compact_kv_cache_times:
            print("\n" + "="*80)
            print("COMPACT KV CACHE TIMES (During Migration)")
            print("="*80)
            print(f"Total events: {len(self.compact_kv_cache_times)}")
            for i, time in enumerate(self.compact_kv_cache_times, 1):
                print(f"  Event {i}: {time}")
            print()
        
        # Print bind kv cache statistics
        if self.bind_kv_cache_times:
            print("\n" + "="*80)
            print("BIND KV CACHE TIMES (During Migration)")
            print("="*80)
            print(f"Total bind operations: {len(self.bind_kv_cache_times)}")
            # Parse and calculate statistics
            bind_times_us = []
            for time_str in self.bind_kv_cache_times:
                time_us, _ = self.parse_time_with_unit(time_str)
                if time_us is not None:
                    bind_times_us.append(time_us)
            
            if bind_times_us:
                import statistics
                avg_us = statistics.mean(bind_times_us)
                median_us = statistics.median(bind_times_us)
                min_us = min(bind_times_us)
                max_us = max(bind_times_us)
                
                # Format with appropriate units
                def format_time(us):
                    if us < 1000:
                        return f"{us:.2f}µs"
                    elif us < 1000000:
                        return f"{us/1000:.2f}ms"
                    else:
                        return f"{us/1000000:.2f}s"
                
                print(f"  Average: {format_time(avg_us)}")
                print(f"  Median:  {format_time(median_us)}")
                print(f"  Min:     {format_time(min_us)}")
                print(f"  Max:     {format_time(max_us)}")
            print()
        
        # Print MemoryStressTester statistics
        if self.memory_stress_tester_times:
            self.print_distribution("MemoryStressTester Allocation Time",
                                  self.memory_stress_tester_times, top_n)
        
        if self.allocate_kv_times:
            self.print_distribution("Asynchronous KV Allocation Time",
                                  self.allocate_kv_times, top_n) 
        # Print CompiledDAG WRITE lock acquisition times
        if self.compiled_dag_write_lock_times:
            self.print_distribution("CompiledDAG WRITE Forward Lock Acquisition Time",
                                  self.compiled_dag_write_lock_times, top_n)
        
        # Print CompiledDAG WRITE completion times
        if self.compiled_dag_write_complete_times:
            self.print_distribution("CompiledDAG WRITE Completion Time",
                                  self.compiled_dag_write_complete_times, top_n)
        
        # Print CompiledDAG READ lock acquisition times
        if self.compiled_dag_read_lock_times:
            self.print_distribution("CompiledDAG READ Forward Lock Acquisition Time",
                                  self.compiled_dag_read_lock_times, top_n)
        
        # Print CompiledDAG READ completion times
        if self.compiled_dag_read_complete_times:
            self.print_distribution("CompiledDAG READ Completion Time",
                                  self.compiled_dag_read_complete_times, top_n)
        
        # Print DynamicChannel tensor receive times
        if self.dynamic_channel_recv_times:
            self.print_distribution("DynamicChannel Tensors Received Time",
                                  self.dynamic_channel_recv_times, top_n)
        
        # Print weight loading lock acquisition times
        if self.weight_loading_lock_times:
            self.print_distribution("Weight Loading Lock Acquisition Time",
                                  self.weight_loading_lock_times, top_n)
        
        # Print recv buffer allocation times
        if self.recv_buffer_alloc_times:
            self.print_distribution("Recv Buffer Allocation Time",
                                  self.recv_buffer_alloc_times, top_n)
        
        # Print DynamicChannel tensor(s) received with fbgate times
        if self.dynamic_channel_recv_with_fbgate_times:
            self.print_distribution("DynamicChannel Tensor(s) Received Time (with fbgate)",
                                  self.dynamic_channel_recv_with_fbgate_times, top_n)
        
        # Print tensor metadata receive times
        if self.tensor_metadata_recv_times:
            self.print_distribution("Tensor Metadata Receive Time",
                                  self.tensor_metadata_recv_times, top_n)
        
        # Print RAY.NCCLGROUP recv buffer allocation times
        if self.nccl_recv_buffer_alloc_times:
            self.print_distribution("RAY.NCCLGROUP Recv Buffer Allocation Time",
                                  self.nccl_recv_buffer_alloc_times, top_n)
        
        # Print RAY.NCCLGROUP received data from comm times
        if self.nccl_recv_data_times:
            self.print_distribution("RAY.NCCLGROUP Received Data From Comm Time",
                                  self.nccl_recv_data_times, top_n)
        
        # Print RAY.NCCLGROUP cuda synchronize times
        if self.nccl_cuda_sync_times:
            self.print_distribution("RAY.NCCLGROUP CUDA Synchronize Time",
                                  self.nccl_cuda_sync_times, top_n)
        
        # Print RAY.NCCLGROUP send queued to peer times
        if self.nccl_send_queued_times:
            self.print_distribution("RAY.NCCLGROUP Send Queued To Peer Time",
                                  self.nccl_send_queued_times, top_n)
        
        # Print weight loading times
        if self.weight_loading_times:
            self.print_distribution("[Weight Loading] Weight Loading Time (Individual Weights)",
                                  self.weight_loading_times, top_n)
        
        # 1. Getting model lock
        if self.metrics['getting_model_lock']:
            self.print_distribution("Getting Model Lock Taking", 
                                  self.metrics['getting_model_lock'], top_n)
        
        # 2. Getting forward lock
        if self.metrics['getting_forward_lock']:
            self.print_distribution("Getting Forward Lock Taking", 
                                  self.metrics['getting_forward_lock'], top_n)
        
        # 3. Engine step time
        if self.metrics['engine_step_time']:
            self.print_distribution("Process Engine Step Time", 
                                  self.metrics['engine_step_time'], top_n)
        
        # 4. After layer forwarding (per layer)
        if self.metrics['after_layer_forwarding']:
            self.print_layer_distribution("After Layer Forwarding", 
                                        self.metrics['after_layer_forwarding'])
        
        # 5. After forwarding
        if self.metrics['after_forwarding']:
            self.print_distribution("After Forwarding Took", 
                                  self.metrics['after_forwarding'], top_n)
        
        # 6. Communication time
        if self.metrics['communication_time']:
            self.print_distribution("Communication Time from Upstream", 
                                  self.metrics['communication_time'], top_n)
        
        # 7. attention forward took
        if self.metrics['attention_forward']:
            self.print_distribution("Attention Forward Took", 
                                  self.metrics['attention_forward'], top_n)
        
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
  python analyze_log_metrics.py /path/to/logfile.log
  
  # Show top 50 longest times
  python analyze_log_metrics.py /path/to/logfile.log --top 50
  
  # Analyze specific metrics only
  python analyze_log_metrics.py /path/to/logfile.log --metrics forward_lock communication
        """
    )
    
    parser.add_argument('log_file', help='Path to the log file to analyze')
    parser.add_argument('--top', '-t', type=int, default=20,
                       help='Number of top entries to show (default: 20)')
    parser.add_argument('--metrics', '-m', nargs='+',
                       choices=['model_lock', 'forward_lock', 'engine_step', 
                               'layer_forward', 'forwarding', 'communication', 'reshape'],
                       help='Specific metrics to analyze (default: all)')
    
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
        sys.exit(1)


if __name__ == "__main__":
    main()
