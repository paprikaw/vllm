#!/usr/bin/env python3
"""
Log Metric Extractor Tool

Extracts numeric values from log files based on a text pattern and computes statistics.
The tool assumes the value immediately follows the pattern text.

Usage:
    python extract_metric.py <log_file> "<pattern>" [options]

Examples:
    # Extract "kv ptr_tables update took" values
    python extract_metric.py server.log "kv ptr_tables update took"
    
    # Extract from multiple files using glob
    python extract_metric.py "logs/**/*.log" "prepare_inputs took" --recursive
    
    # Show individual values
    python extract_metric.py server.log "forward took" --show-values
    
    # Output as JSON
    python extract_metric.py server.log "kv ptr_tables update took" --json
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional


# Time unit conversion to microseconds
TIME_UNITS = {
    'ns': 0.001,
    'µs': 1,
    'us': 1,
    'μs': 1,
    'ms': 1000,
    's': 1000000,
    'sec': 1000000,
    'second': 1000000,
    'seconds': 1000000,
    'min': 60000000,
    'minute': 60000000,
    'minutes': 60000000,
}

# Size unit conversion to bytes
SIZE_UNITS = {
    'b': 1,
    'bytes': 1,
    'kb': 1024,
    'mb': 1024**2,
    'gb': 1024**3,
    'tb': 1024**4,
}


def parse_value_with_unit(value_str: str) -> Tuple[float, str, str]:
    """
    Parse a value string like "314µs", "2.5ms", "1.2GB" into (numeric_value, unit, category).
    Returns (value, unit, category) where category is 'time', 'size', or 'unknown'.
    """
    # Pattern: number (optional decimal) followed by optional unit
    pattern = r'^(\d+(?:\.\d+)?)\s*([a-zA-Zµμ]*)'
    match = re.match(pattern, value_str.strip())
    
    if not match:
        raise ValueError(f"Cannot parse value: {value_str}")
    
    value = float(match.group(1))
    unit = match.group(2).lower() if match.group(2) else ''
    
    # Handle special unicode characters
    if 'µ' in value_str or 'μ' in value_str:
        unit = 'µs' if 's' in unit else unit
    
    # Determine category and normalize
    if unit in TIME_UNITS or unit == 'µs':
        return value, unit if unit else 'µs', 'time'
    elif unit in SIZE_UNITS:
        return value, unit, 'size'
    else:
        return value, unit, 'unknown'


def normalize_value(value: float, unit: str, category: str) -> float:
    """Normalize value to base unit (µs for time, bytes for size)."""
    if category == 'time':
        if unit == 'µs' or unit == 'μs':
            return value
        return value * TIME_UNITS.get(unit, 1)
    elif category == 'size':
        return value * SIZE_UNITS.get(unit, 1)
    return value


def extract_values_from_file(filepath: str, pattern: str) -> List[Tuple[float, str, str, int]]:
    """
    Extract values following the pattern from a file.
    Returns list of (normalized_value, original_unit, category, line_number).
    """
    results = []
    
    # Escape special regex characters in pattern but keep it as literal text
    escaped_pattern = re.escape(pattern)
    # Pattern: look for the text followed by whitespace and then a value with optional unit
    value_pattern = escaped_pattern + r'\s+(\d+(?:\.\d+)?[a-zA-Zµμ]*)'
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                match = re.search(value_pattern, line)
                if match:
                    value_str = match.group(1)
                    try:
                        value, unit, category = parse_value_with_unit(value_str)
                        normalized = normalize_value(value, unit, category)
                        results.append((normalized, unit, category, line_num))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Warning: Error reading {filepath}: {e}", file=sys.stderr)
    
    return results


def compute_statistics(values: List[float]) -> dict:
    """Compute statistics for a list of values."""
    if not values:
        return {}
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    return {
        'count': n,
        'avg': sum(values) / n,
        'min': min(values),
        'max': max(values),
        'median': sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2,
        'p95': sorted_values[int(n * 0.95)] if n >= 20 else None,
        'p99': sorted_values[int(n * 0.99)] if n >= 100 else None,
        'std': (sum((x - sum(values) / n) ** 2 for x in values) / n) ** 0.5 if n > 1 else 0,
    }


def format_value(value: float, category: str) -> str:
    """Format value with appropriate unit."""
    if category == 'time':
        if value < 1000:
            return f"{value:.2f}µs"
        elif value < 1000000:
            return f"{value/1000:.2f}ms"
        else:
            return f"{value/1000000:.2f}s"
    elif category == 'size':
        if value < 1024:
            return f"{value:.2f}B"
        elif value < 1024**2:
            return f"{value/1024:.2f}KB"
        elif value < 1024**3:
            return f"{value/1024**2:.2f}MB"
        else:
            return f"{value/1024**3:.2f}GB"
    else:
        return f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description='Extract and analyze numeric values from log files based on text pattern.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('log_file', help='Log file path or glob pattern (use quotes for patterns)')
    parser.add_argument('pattern', help='Text pattern to search for (value should follow this text)')
    parser.add_argument('-r', '--recursive', action='store_true', help='Search recursively for glob patterns')
    parser.add_argument('-v', '--show-values', action='store_true', help='Show individual values')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--top', type=int, default=10, help='Number of top/bottom values to show (default: 10)')
    parser.add_argument('--histogram', action='store_true', help='Show histogram of values')
    
    args = parser.parse_args()
    
    # Expand glob pattern
    if '*' in args.log_file or '?' in args.log_file:
        files = glob.glob(args.log_file, recursive=args.recursive)
    else:
        files = [args.log_file]
    
    if not files:
        print(f"No files found matching: {args.log_file}", file=sys.stderr)
        sys.exit(1)
    
    # Extract values from all files
    all_results = []
    for filepath in files:
        if Path(filepath).is_file():
            results = extract_values_from_file(filepath, args.pattern)
            for value, unit, category, line_num in results:
                all_results.append({
                    'file': filepath,
                    'line': line_num,
                    'value': value,
                    'unit': unit,
                    'category': category
                })
    
    if not all_results:
        print(f"No matches found for pattern: '{args.pattern}'", file=sys.stderr)
        sys.exit(1)
    
    # Get category from first result (assume all same category)
    category = all_results[0]['category']
    values = [r['value'] for r in all_results]
    stats = compute_statistics(values)
    
    if args.json:
        output = {
            'pattern': args.pattern,
            'files_searched': len(files),
            'statistics': stats,
            'unit': 'µs' if category == 'time' else 'bytes' if category == 'size' else 'raw',
        }
        if args.show_values:
            output['values'] = all_results
        print(json.dumps(output, indent=2))
    else:
        print(f"Pattern: '{args.pattern}'")
        print(f"Files searched: {len(files)}")
        print(f"Matches found: {stats['count']}")
        print()
        print("Statistics:")
        print(f"  Count:  {stats['count']}")
        print(f"  Avg:    {format_value(stats['avg'], category)}")
        print(f"  Min:    {format_value(stats['min'], category)}")
        print(f"  Max:    {format_value(stats['max'], category)}")
        print(f"  Median: {format_value(stats['median'], category)}")
        if stats.get('p95'):
            print(f"  P95:    {format_value(stats['p95'], category)}")
        if stats.get('p99'):
            print(f"  P99:    {format_value(stats['p99'], category)}")
        print(f"  StdDev: {format_value(stats['std'], category)}")
        
        if args.show_values:
            print()
            print(f"Top {args.top} highest values:")
            sorted_results = sorted(all_results, key=lambda x: x['value'], reverse=True)
            for i, r in enumerate(sorted_results[:args.top], 1):
                print(f"  {i}. {format_value(r['value'], category)} ({r['file']}:{r['line']})")
            
            print()
            print(f"Top {args.top} lowest values:")
            for i, r in enumerate(sorted_results[-args.top:], 1):
                print(f"  {i}. {format_value(r['value'], category)} ({r['file']}:{r['line']})")
        
        if args.histogram:
            print()
            print("Histogram:")
            # Simple text histogram
            num_bins = 10
            min_val, max_val = min(values), max(values)
            if min_val == max_val:
                print(f"  All values are {format_value(min_val, category)}")
            else:
                bin_width = (max_val - min_val) / num_bins
                bins = [0] * num_bins
                for v in values:
                    bin_idx = min(int((v - min_val) / bin_width), num_bins - 1)
                    bins[bin_idx] += 1
                
                max_count = max(bins)
                bar_width = 40
                for i, count in enumerate(bins):
                    bin_start = min_val + i * bin_width
                    bin_end = bin_start + bin_width
                    bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
                    print(f"  {format_value(bin_start, category):>12} - {format_value(bin_end, category):>12} | {'█' * bar_len} ({count})")


if __name__ == '__main__':
    main()
