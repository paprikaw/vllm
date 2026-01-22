#!/usr/bin/env python3
"""
Analyze single request trace logs from three GPU configurations.
Compares hidden states, logits, and sampled tokens across:
- A100 Single Node
- L40 Single Node  
- A100 -> L40 Cross Node
"""

import re
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import glob

LOG_DIR = "/home/bxb1/vllm_workbench/vllm/logs/agent/single_req_trace"

def parse_token_hs(line: str) -> Optional[Dict]:
    """Parse [PP_TOKEN_HS], [PP_SEND_HS], or [PP_RECV_HS] lines"""
    patterns = [
        r'\[PP_TOKEN_HS\] rank=(\d+) req=(\w+) tok_idx=(\d+) norm=([\d.]+) first8=\[([^\]]+)\] last8=\[([^\]]+)\]',
        r'\[PP_SEND_HS\] rank=(\d+) req=(\w+) tok_idx=(\d+) norm=([\d.]+) first8=\[([^\]]+)\] last8=\[([^\]]+)\]',
        r'\[PP_RECV_HS\] rank=(\d+) req=(\w+) tok_idx=(\d+) norm=([\d.]+) first8=\[([^\]]+)\] last8=\[([^\]]+)\]',
    ]
    
    for pattern in patterns:
        m = re.search(pattern, line)
        if m:
            tag = "TOKEN_HS" if "TOKEN_HS" in line else ("SEND_HS" if "SEND_HS" in line else "RECV_HS")
            return {
                'tag': tag,
                'rank': int(m.group(1)),
                'req_id': m.group(2),
                'tok_idx': int(m.group(3)),
                'norm': float(m.group(4)),
                'first8': [float(x.strip().strip("'")) for x in m.group(5).split(',')],
                'last8': [float(x.strip().strip("'")) for x in m.group(6).split(',')],
            }
    return None

def parse_token_logits(line: str) -> Optional[Dict]:
    """Parse [PP_TOKEN_LOGITS] lines"""
    pattern = r'\[PP_TOKEN_LOGITS\] rank=(\d+) req=(\w+) tok_idx=(\d+) top5_tokens=\[([^\]]+)\] top5_logits=\[([^\]]+)\]'
    m = re.search(pattern, line)
    if m:
        return {
            'rank': int(m.group(1)),
            'req_id': m.group(2),
            'tok_idx': int(m.group(3)),
            'top5_tokens': [int(x.strip()) for x in m.group(4).split(',')],
            'top5_logits': [float(x.strip().strip("'")) for x in m.group(5).split(',')],
        }
    return None

def parse_sample_detail(line: str) -> Optional[Dict]:
    """Parse [PP_SAMPLE_DETAIL] lines"""
    pattern = r'\[PP_SAMPLE_DETAIL\] rank=(\d+) req=(\w+) token=(\d+) top3=\[([^\]]*)\]'
    m = re.search(pattern, line)
    if m:
        top3_str = m.group(4)
        top3 = [int(x.strip()) for x in top3_str.split(',')] if top3_str.strip() else []
        return {
            'rank': int(m.group(1)),
            'req_id': m.group(2),
            'sampled_token': int(m.group(3)),
            'top3': top3,
        }
    return None

def parse_sample_input(line: str) -> Optional[Dict]:
    """Parse [PP_SAMPLE_INPUT] lines"""
    pattern = r'\[PP_SAMPLE_INPUT\] rank=(\d+) sample_hidden_states: min=([\-\d.]+), max=([\-\d.]+), mean=([\-\d.]+), std=([\-\d.]+), absmax=([\-\d.]+)'
    m = re.search(pattern, line)
    if m:
        return {
            'rank': int(m.group(1)),
            'hs_min': float(m.group(2)),
            'hs_max': float(m.group(3)),
            'hs_mean': float(m.group(4)),
            'hs_std': float(m.group(5)),
            'hs_absmax': float(m.group(6)),
        }
    return None

def analyze_log(log_path: str) -> Dict:
    """Analyze a single log file"""
    result = {
        'token_hs': [],  # Per-token hidden states
        'send_hs': [],   # Hidden states being sent
        'recv_hs': [],   # Hidden states received
        'token_logits': [],  # Per-token logits
        'sample_details': [],  # Sampled tokens
        'sample_inputs': [],  # Overall sample input stats
    }
    
    if not os.path.exists(log_path):
        print(f"  Warning: Log file not found: {log_path}")
        return result
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse token hidden states
            parsed = parse_token_hs(line)
            if parsed:
                if parsed['tag'] == 'TOKEN_HS':
                    result['token_hs'].append(parsed)
                elif parsed['tag'] == 'SEND_HS':
                    result['send_hs'].append(parsed)
                elif parsed['tag'] == 'RECV_HS':
                    result['recv_hs'].append(parsed)
                continue
            
            # Parse token logits
            parsed = parse_token_logits(line)
            if parsed:
                result['token_logits'].append(parsed)
                continue
            
            # Parse sample details
            parsed = parse_sample_detail(line)
            if parsed:
                result['sample_details'].append(parsed)
                continue
            
            # Parse sample inputs
            parsed = parse_sample_input(line)
            if parsed:
                result['sample_inputs'].append(parsed)
    
    return result

def compare_token_hs(test_name: str, data1: List[Dict], data2: List[Dict], name1: str, name2: str):
    """Compare token hidden states between two tests"""
    print(f"\n  Comparing {name1} vs {name2} hidden states:")
    
    # Group by tok_idx
    hs1_by_idx = {d['tok_idx']: d for d in data1}
    hs2_by_idx = {d['tok_idx']: d for d in data2}
    
    common_indices = set(hs1_by_idx.keys()) & set(hs2_by_idx.keys())
    
    for idx in sorted(common_indices)[:5]:  # Check first 5 tokens
        h1 = hs1_by_idx[idx]
        h2 = hs2_by_idx[idx]
        
        norm_diff = abs(h1['norm'] - h2['norm'])
        first8_diff = sum(abs(a - b) for a, b in zip(h1['first8'], h2['first8'])) / 8
        last8_diff = sum(abs(a - b) for a, b in zip(h1['last8'], h2['last8'])) / 8
        
        status = "✓" if norm_diff < 0.01 and first8_diff < 0.01 else "✗"
        print(f"    Token {idx}: norm_diff={norm_diff:.6f}, first8_diff={first8_diff:.6f}, last8_diff={last8_diff:.6f} {status}")

def compare_logits(data1: List[Dict], data2: List[Dict], name1: str, name2: str):
    """Compare logits between two tests"""
    print(f"\n  Comparing {name1} vs {name2} logits:")
    
    log1_by_idx = {d['tok_idx']: d for d in data1}
    log2_by_idx = {d['tok_idx']: d for d in data2}
    
    common_indices = set(log1_by_idx.keys()) & set(log2_by_idx.keys())
    
    for idx in sorted(common_indices)[:5]:
        l1 = log1_by_idx[idx]
        l2 = log2_by_idx[idx]
        
        top1_match = l1['top5_tokens'][0] == l2['top5_tokens'][0]
        top5_overlap = len(set(l1['top5_tokens']) & set(l2['top5_tokens']))
        
        logits_diff = sum(abs(a - b) for a, b in zip(l1['top5_logits'], l2['top5_logits'])) / 5
        
        status = "✓" if top1_match else "✗"
        print(f"    Token {idx}: top1_match={top1_match} {status}, top5_overlap={top5_overlap}/5, logits_diff={logits_diff:.4f}")
        print(f"      {name1}: top5={l1['top5_tokens']}")
        print(f"      {name2}: top5={l2['top5_tokens']}")

def compare_sampled_tokens(data1: List[Dict], data2: List[Dict], name1: str, name2: str):
    """Compare sampled tokens between two tests"""
    print(f"\n  Comparing {name1} vs {name2} sampled tokens:")
    
    tokens1 = [d['sampled_token'] for d in data1]
    tokens2 = [d['sampled_token'] for d in data2]
    
    min_len = min(len(tokens1), len(tokens2))
    matches = sum(1 for i in range(min_len) if tokens1[i] == tokens2[i])
    
    print(f"    Total tokens: {name1}={len(tokens1)}, {name2}={len(tokens2)}")
    print(f"    Matching tokens: {matches}/{min_len} ({100*matches/min_len:.1f}%)")
    
    if min_len > 0 and matches < min_len:
        print(f"    First mismatch at index:")
        for i in range(min_len):
            if tokens1[i] != tokens2[i]:
                print(f"      Index {i}: {name1}={tokens1[i]}, {name2}={tokens2[i]}")
                break
    
    print(f"    First 10 tokens:")
    print(f"      {name1}: {tokens1[:10]}")
    print(f"      {name2}: {tokens2[:10]}")

def main():
    print("="*70)
    print("Single Request Trace Analysis")
    print("="*70)
    
    tests = {
        'A100 Single': 'a100_single',
        'L40 Single': 'l40_single',
        'Cross Node': 'cross_node',
    }
    
    results = {}
    
    # Load all logs
    for name, dir_name in tests.items():
        log_pattern = f"{LOG_DIR}/{dir_name}/project-single_req_trace_*/server-*.log"
        log_files = glob.glob(log_pattern)
        
        print(f"\n{name}:")
        print(f"  Looking for: {log_pattern}")
        
        if not log_files:
            print(f"  No log files found!")
            results[name] = None
            continue
        
        log_file = log_files[0]  # Use first match
        print(f"  Analyzing: {log_file}")
        results[name] = analyze_log(log_file)
        
        data = results[name]
        print(f"  Found:")
        print(f"    - {len(data['token_hs'])} token hidden states (rank 1)")
        print(f"    - {len(data['send_hs'])} sent hidden states (rank 0)")
        print(f"    - {len(data['recv_hs'])} received hidden states (rank 1)")
        print(f"    - {len(data['token_logits'])} token logits")
        print(f"    - {len(data['sample_details'])} sampled tokens")
    
    # Compare results
    print("\n" + "="*70)
    print("Comparison Analysis")
    print("="*70)
    
    # Compare A100 vs L40 (should be similar)
    if results.get('A100 Single') and results.get('L40 Single'):
        print("\n[1] A100 vs L40 Single Node (baseline comparison)")
        
        # Compare sent hidden states (rank 0 output)
        compare_token_hs("Sent HS", 
                        results['A100 Single']['send_hs'],
                        results['L40 Single']['send_hs'],
                        "A100", "L40")
        
        # Compare logits (rank 1)
        compare_logits(results['A100 Single']['token_logits'],
                      results['L40 Single']['token_logits'],
                      "A100", "L40")
        
        # Compare sampled tokens
        compare_sampled_tokens(results['A100 Single']['sample_details'],
                              results['L40 Single']['sample_details'],
                              "A100", "L40")
    
    # Compare A100 vs Cross Node
    if results.get('A100 Single') and results.get('Cross Node'):
        print("\n[2] A100 Single vs Cross Node (detect cross-node issues)")
        
        compare_token_hs("Sent HS",
                        results['A100 Single']['send_hs'],
                        results['Cross Node']['send_hs'],
                        "A100", "Cross")
        
        compare_logits(results['A100 Single']['token_logits'],
                      results['Cross Node']['token_logits'],
                      "A100", "Cross")
        
        compare_sampled_tokens(results['A100 Single']['sample_details'],
                              results['Cross Node']['sample_details'],
                              "A100", "Cross")
    
    # Specific cross-node analysis: compare sent vs received
    if results.get('Cross Node'):
        print("\n[3] Cross Node: Sent vs Received Hidden States")
        data = results['Cross Node']
        
        if data['send_hs'] and data['recv_hs']:
            print("  Comparing what rank 0 sends vs what rank 1 receives:")
            
            send_by_idx = {d['tok_idx']: d for d in data['send_hs']}
            recv_by_idx = {d['tok_idx']: d for d in data['recv_hs']}
            
            common = set(send_by_idx.keys()) & set(recv_by_idx.keys())
            
            for idx in sorted(common)[:5]:
                s = send_by_idx[idx]
                r = recv_by_idx[idx]
                
                norm_diff = abs(s['norm'] - r['norm'])
                first8_diff = sum(abs(a - b) for a, b in zip(s['first8'], r['first8'])) / 8
                
                status = "✓" if norm_diff < 0.001 else "✗ MISMATCH!"
                print(f"    Token {idx}: norm_diff={norm_diff:.6f}, first8_diff={first8_diff:.6f} {status}")
                if norm_diff >= 0.001:
                    print(f"      Sent:     norm={s['norm']:.4f}, first8={s['first8'][:4]}")
                    print(f"      Received: norm={r['norm']:.4f}, first8={r['first8'][:4]}")
    
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)

if __name__ == "__main__":
    main()
