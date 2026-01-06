#!/usr/bin/env python3
"""
TTFT Breakdown Analyzer
Analyzes TTFT composition from [ttft_trace] logs to identify bottlenecks.
"""

import re
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TTFTEvent:
    """Represents a single TTFT trace event."""
    timestamp: str
    request_id: str
    event_type: str
    details: Dict[str, float]
    raw_line: str


class TTFTBreakdownAnalyzer:
    """Analyze TTFT breakdown from server logs."""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.events: Dict[str, List[TTFTEvent]] = defaultdict(list)
        self.request_timelines: Dict[str, Dict] = {}
        
    def parse_log(self):
        """Parse log file and extract [ttft_trace] events."""
        print(f"Parsing log file: {self.log_file}")
        
        with open(self.log_file, 'r') as f:
            for line in f:
                if '[ttft_trace]' not in line:
                    continue
                    
                event = self._parse_ttft_line(line)
                if event:
                    self.events[event.request_id].append(event)
        
        print(f"Found trace events for {len(self.events)} requests")
        print()
    
    def _parse_ttft_line(self, line: str) -> Optional[TTFTEvent]:
        """Parse a single [ttft_trace] log line."""
        # Extract timestamp - support both formats
        ts_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if not ts_match:
            # Try alternative format: INFO 01-02 16:50:02
            ts_match = re.search(r'INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if not ts_match:
            return None
        timestamp = ts_match.group(1)
        
        # Extract request ID - handles both "Request ID:" and "Waiting request ID:"
        req_match = re.search(r'(?:Request|request) ([^\s:]+):', line)
        if not req_match:
            # Try alternative: "Scheduler:" lines don't have request ID
            if 'Scheduler:' in line:
                return None
            return None
        request_id = req_match.group(1)
        
        details = {}
        
        # Parse different event types
        if 'Enqueued at' in line:
            event_type = 'ENQUEUE'
            # Extract: arrival_time, wait_in_queue
            arrival_match = re.search(r'arrival_time=([\d.]+)', line)
            wait_match = re.search(r'wait_in_queue=([\d.]+)ms', line)
            if arrival_match:
                details['arrival_time'] = float(arrival_match.group(1))
            if wait_match:
                details['wait_in_queue_ms'] = float(wait_match.group(1))
                
        elif 'Added to EngineCore' in line:
            event_type = 'ADD_TO_CORE'
            time_match = re.search(r'in ([\d.]+)ms', line)
            if time_match:
                details['add_time_ms'] = float(time_match.group(1))
                
        elif 'Selected for PREFILL' in line:
            event_type = 'SELECTED_PREFILL'
            wait_match = re.search(r'waited ([\d.]+)ms', line)
            tokens_match = re.search(r'num_tokens=(\d+)', line)
            if wait_match:
                details['wait_time_ms'] = float(wait_match.group(1))
            if tokens_match:
                details['num_tokens'] = int(tokens_match.group(1))
                
        elif 'FIRST TOKEN generated' in line:
            event_type = 'FIRST_TOKEN'
            ttft_match = re.search(r'TTFT=([\d.]+)ms', line)
            if ttft_match:
                details['ttft_ms'] = float(ttft_match.group(1))
                
        else:
            return None
        
        return TTFTEvent(
            timestamp=timestamp,
            request_id=request_id,
            event_type=event_type,
            details=details,
            raw_line=line.strip()
        )
    
    def build_request_timelines(self):
        """Build complete timelines for each request."""
        for req_id, events in self.events.items():
            timeline = {
                'enqueue': None,
                'add_to_core': None,
                'selected_prefill': None,
                'first_token': None,
            }
            
            for event in events:
                if event.event_type == 'ENQUEUE':
                    timeline['enqueue'] = event
                elif event.event_type == 'ADD_TO_CORE':
                    timeline['add_to_core'] = event
                elif event.event_type == 'SELECTED_PREFILL':
                    timeline['selected_prefill'] = event
                elif event.event_type == 'FIRST_TOKEN':
                    timeline['first_token'] = event
            
            self.request_timelines[req_id] = timeline
    
    def calculate_ttft_breakdown(self, req_id: str) -> Optional[Dict]:
        """Calculate TTFT breakdown for a specific request."""
        timeline = self.request_timelines.get(req_id)
        if not timeline:
            return None
        
        breakdown = {}
        
        # Extract times
        if timeline['enqueue']:
            breakdown['wait_in_queue_ms'] = timeline['enqueue'].details.get('wait_in_queue_ms', 0)
        
        if timeline['add_to_core']:
            breakdown['add_to_core_ms'] = timeline['add_to_core'].details.get('add_time_ms', 0)
        
        if timeline['selected_prefill']:
            breakdown['wait_for_scheduling_ms'] = timeline['selected_prefill'].details.get('wait_time_ms', 0)
        
        if timeline['first_token']:
            breakdown['total_ttft_ms'] = timeline['first_token'].details.get('ttft_ms', 0)
        
        # Calculate execution time (total - waiting times)
        if 'total_ttft_ms' in breakdown:
            known_wait = (
                breakdown.get('wait_in_queue_ms', 0) +
                breakdown.get('add_to_core_ms', 0)
            )
            breakdown['execution_ms'] = breakdown['total_ttft_ms'] - known_wait
        
        return breakdown
    
    def analyze_high_ttft_requests(self, threshold_ms: float = 1000) -> List[Tuple[str, Dict]]:
        """Analyze requests with high TTFT."""
        high_ttft = []
        
        for req_id, timeline in self.request_timelines.items():
            if timeline['first_token']:
                ttft_ms = timeline['first_token'].details.get('ttft_ms', 0)
                if ttft_ms > threshold_ms:
                    breakdown = self.calculate_ttft_breakdown(req_id)
                    if breakdown:
                        high_ttft.append((req_id, breakdown))
        
        return sorted(high_ttft, key=lambda x: x[1].get('total_ttft_ms', 0), reverse=True)
    
    def print_request_timeline(self, req_id: str):
        """Print detailed timeline for a specific request."""
        timeline = self.request_timelines.get(req_id)
        if not timeline:
            print(f"No timeline found for request {req_id}")
            return
        
        print(f"\n{'='*80}")
        print(f"Request Timeline: {req_id}")
        print(f"{'='*80}")
        
        for stage, event in timeline.items():
            if event:
                print(f"\n{stage.upper()}:")
                print(f"  Timestamp: {event.timestamp}")
                print(f"  Details: {event.details}")
        
        breakdown = self.calculate_ttft_breakdown(req_id)
        if breakdown:
            print(f"\nTTFT Breakdown:")
            print(f"  Total TTFT:           {breakdown.get('total_ttft_ms', 0):.2f} ms")
            print(f"  Wait in Queue:        {breakdown.get('wait_in_queue_ms', 0):.2f} ms")
            print(f"  Add to EngineCore:    {breakdown.get('add_to_core_ms', 0):.2f} ms")
            print(f"  Execution Time:       {breakdown.get('execution_ms', 0):.2f} ms")
            
            total = breakdown.get('total_ttft_ms', 1)
            print(f"\nPercentage Breakdown:")
            print(f"  Queue Wait:     {breakdown.get('wait_in_queue_ms', 0)/total*100:6.2f}%")
            print(f"  Add to Core:    {breakdown.get('add_to_core_ms', 0)/total*100:6.2f}%")
            print(f"  Execution:      {breakdown.get('execution_ms', 0)/total*100:6.2f}%")
        
        print(f"{'='*80}\n")
    
    def print_summary_report(self, threshold_ms: float = 1000):
        """Print summary report of TTFT composition."""
        print("=" * 80)
        print(f"TTFT COMPOSITION ANALYSIS")
        print(f"Log file: {self.log_file}")
        print("=" * 80)
        print()
        
        print(f"Total requests traced: {len(self.request_timelines)}")
        print()
        
        # Analyze high TTFT requests
        high_ttft = self.analyze_high_ttft_requests(threshold_ms)
        
        if not high_ttft:
            print(f"No requests with TTFT > {threshold_ms}ms found")
            return
        
        print(f"Requests with TTFT > {threshold_ms}ms: {len(high_ttft)}")
        print()
        
        # Calculate aggregate statistics
        all_wait_in_queue = []
        all_execution = []
        all_total = []
        
        for req_id, breakdown in high_ttft:
            all_wait_in_queue.append(breakdown.get('wait_in_queue_ms', 0))
            all_execution.append(breakdown.get('execution_ms', 0))
            all_total.append(breakdown.get('total_ttft_ms', 0))
        
        import numpy as np
        
        print("Aggregate Statistics for High-TTFT Requests:")
        print(f"  {'Component':<20} {'Mean':<12} {'Median':<12} {'Max':<12}")
        print("  " + "-" * 56)
        print(f"  {'Wait in Queue':<20} {np.mean(all_wait_in_queue):<12.2f} "
              f"{np.median(all_wait_in_queue):<12.2f} {np.max(all_wait_in_queue):<12.2f}")
        print(f"  {'Execution Time':<20} {np.mean(all_execution):<12.2f} "
              f"{np.median(all_execution):<12.2f} {np.max(all_execution):<12.2f}")
        print(f"  {'Total TTFT':<20} {np.mean(all_total):<12.2f} "
              f"{np.median(all_total):<12.2f} {np.max(all_total):<12.2f}")
        print()
        
        # Show percentage breakdown
        avg_total = np.mean(all_total)
        print("Average Percentage Breakdown:")
        print(f"  Queue Wait:   {np.mean(all_wait_in_queue)/avg_total*100:6.2f}%")
        print(f"  Execution:    {np.mean(all_execution)/avg_total*100:6.2f}%")
        print()
        
        # Show top 10 worst requests
        print("Top 10 Highest TTFT Requests:")
        print(f"  {'Request ID':<30} {'Total TTFT':<12} {'Queue Wait':<12} {'Execution':<12}")
        print("  " + "-" * 66)
        
        for req_id, breakdown in high_ttft[:10]:
            print(f"  {req_id:<30} {breakdown.get('total_ttft_ms', 0):<12.2f} "
                  f"{breakdown.get('wait_in_queue_ms', 0):<12.2f} "
                  f"{breakdown.get('execution_ms', 0):<12.2f}")
        
        print()
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze TTFT composition from server logs')
    parser.add_argument('log_file', help='Path to server log file with [ttft_trace] entries')
    parser.add_argument('--threshold', type=float, default=1000,
                        help='TTFT threshold in ms for high-latency analysis (default: 1000)')
    parser.add_argument('--request-id', type=str,
                        help='Show detailed timeline for specific request ID')
    
    args = parser.parse_args()
    
    analyzer = TTFTBreakdownAnalyzer(args.log_file)
    analyzer.parse_log()
    analyzer.build_request_timelines()
    
    if args.request_id:
        analyzer.print_request_timeline(args.request_id)
    else:
        analyzer.print_summary_report(args.threshold)


if __name__ == '__main__':
    main()
