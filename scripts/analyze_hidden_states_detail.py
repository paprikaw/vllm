#!/usr/bin/env python3
"""
Detailed analysis of hidden states differences between A100 and L40 GPUs.
This script compares the actual numerical values to understand why results differ.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

def parse_hidden_states(log_path):
    """Parse hidden states from log file."""
    data = {
        'send': [],    # PP_SEND_HS from rank 0
        'recv': [],    # PP_RECV_HS from rank 1
        'sample': [],  # PP_TOKEN_HS from rank 1
        'logits': [],  # PP_TOKEN_LOGITS
        'lm_head': [],
        'sample_input': [],
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse PP_SEND_HS
            m = re.search(r'\[PP_SEND_HS\] rank=(\d+).*tok_idx=(\d+) norm=([\d.]+) first8=\[(.*?)\] last8=\[(.*?)\]', line)
            if m:
                rank, tok_idx, norm, first8, last8 = m.groups()
                first8_vals = [float(x.strip().strip("'\"")) for x in first8.split(',')]
                last8_vals = [float(x.strip().strip("'\"")) for x in last8.split(',')]
                data['send'].append({
                    'rank': int(rank),
                    'tok_idx': int(tok_idx),
                    'norm': float(norm),
                    'first8': first8_vals,
                    'last8': last8_vals,
                })
                continue
            
            # Parse PP_RECV_HS
            m = re.search(r'\[PP_RECV_HS\] rank=(\d+).*tok_idx=(\d+) norm=([\d.]+) first8=\[(.*?)\] last8=\[(.*?)\]', line)
            if m:
                rank, tok_idx, norm, first8, last8 = m.groups()
                first8_vals = [float(x.strip().strip("'\"")) for x in first8.split(',')]
                last8_vals = [float(x.strip().strip("'\"")) for x in last8.split(',')]
                data['recv'].append({
                    'rank': int(rank),
                    'tok_idx': int(tok_idx),
                    'norm': float(norm),
                    'first8': first8_vals,
                    'last8': last8_vals,
                })
                continue
            
            # Parse PP_TOKEN_HS (sample hidden states)
            m = re.search(r'\[PP_TOKEN_HS\] rank=(\d+).*tok_idx=(\d+) norm=([\d.]+) first8=\[(.*?)\] last8=\[(.*?)\]', line)
            if m:
                rank, tok_idx, norm, first8, last8 = m.groups()
                first8_vals = [float(x.strip().strip("'\"")) for x in first8.split(',')]
                last8_vals = [float(x.strip().strip("'\"")) for x in last8.split(',')]
                data['sample'].append({
                    'rank': int(rank),
                    'tok_idx': int(tok_idx),
                    'norm': float(norm),
                    'first8': first8_vals,
                    'last8': last8_vals,
                })
                continue
            
            # Parse PP_TOKEN_LOGITS
            m = re.search(r'\[PP_TOKEN_LOGITS\] rank=(\d+).*tok_idx=(\d+) top5_tokens=\[(.*?)\] top5_logits=\[(.*?)\]', line)
            if m:
                rank, tok_idx, top5_tokens, top5_logits = m.groups()
                top5_t = [int(x.strip()) for x in top5_tokens.split(',')]
                top5_l = [float(x.strip().strip("'\"")) for x in top5_logits.split(',')]
                data['logits'].append({
                    'rank': int(rank),
                    'tok_idx': int(tok_idx),
                    'top5_tokens': top5_t,
                    'top5_logits': top5_l,
                })
                continue
            
            # Parse PP_LM_HEAD
            m = re.search(r'\[PP_LM_HEAD\] rank=(\d+).*shape=(\S+).*min=([-\d.]+), max=([-\d.]+), mean=([-\d.]+)', line)
            if m:
                rank, shape, min_v, max_v, mean_v = m.groups()
                data['lm_head'].append({
                    'rank': int(rank),
                    'shape': shape,
                    'min': float(min_v),
                    'max': float(max_v),
                    'mean': float(mean_v),
                })
                continue
            
            # Parse PP_SAMPLE_INPUT
            m = re.search(r'\[PP_SAMPLE_INPUT\] rank=(\d+).*min=([-\d.]+), max=([-\d.]+), mean=([-\d.]+), std=([-\d.]+)', line)
            if m:
                rank, min_v, max_v, mean_v, std_v = m.groups()
                data['sample_input'].append({
                    'rank': int(rank),
                    'min': float(min_v),
                    'max': float(max_v),
                    'mean': float(mean_v),
                    'std': float(std_v),
                })
                continue
    
    return data


def main():
    base_dir = Path("/home/bxb1/vllm_workbench/vllm/logs/agent/single_req_trace")
    
    # Find log files
    a100_log = list(base_dir.glob("a100_single/project-*/server_raw-*.log"))
    l40_log = list(base_dir.glob("l40_single/project-*/server_raw-*.log"))
    cross_log = list(base_dir.glob("cross_node/project-*/server_raw-*.log"))
    
    if not a100_log or not l40_log or not cross_log:
        print("Missing log files!")
        return
    
    print("=" * 80)
    print("DETAILED HIDDEN STATES ANALYSIS")
    print("=" * 80)
    
    # Parse all logs
    a100_data = parse_hidden_states(a100_log[0])
    l40_data = parse_hidden_states(l40_log[0])
    cross_data = parse_hidden_states(cross_log[0])
    
    print(f"\nA100 log: {a100_log[0]}")
    print(f"L40 log: {l40_log[0]}")
    print(f"Cross log: {cross_log[0]}")
    
    # Check data counts
    print(f"\n[A100] send={len(a100_data['send'])}, recv={len(a100_data['recv'])}, "
          f"sample={len(a100_data['sample'])}, logits={len(a100_data['logits'])}")
    print(f"[L40]  send={len(l40_data['send'])}, recv={len(l40_data['recv'])}, "
          f"sample={len(l40_data['sample'])}, logits={len(l40_data['logits'])}")
    print(f"[Cross] send={len(cross_data['send'])}, recv={len(cross_data['recv'])}, "
          f"sample={len(cross_data['sample'])}, logits={len(cross_data['logits'])}")
    
    # Group by tok_idx for easier comparison
    def group_by_tok_idx(items):
        result = defaultdict(list)
        for item in items:
            result[item['tok_idx']].append(item)
        return result
    
    a100_send_by_idx = group_by_tok_idx(a100_data['send'])
    l40_send_by_idx = group_by_tok_idx(l40_data['send'])
    cross_send_by_idx = group_by_tok_idx(cross_data['send'])
    
    # Analyze token 0 in detail - this is the prefill phase
    print("\n" + "=" * 80)
    print("TOKEN 0 ANALYSIS (PREFILL PHASE)")
    print("=" * 80)
    
    print("\n[1] Sent Hidden States from Rank 0 (after layers 0-31):")
    print("-" * 60)
    
    for test_name, send_data in [("A100", a100_send_by_idx), ("L40", l40_send_by_idx), ("Cross", cross_send_by_idx)]:
        if 0 in send_data:
            for i, item in enumerate(send_data[0]):
                print(f"  {test_name} (occurrence {i+1}): norm={item['norm']:.4f}")
                print(f"    first8: {item['first8']}")
                print(f"    last8: {item['last8']}")
    
    # Compare LM head weights
    print("\n" + "=" * 80)
    print("LM HEAD WEIGHTS COMPARISON")
    print("=" * 80)
    
    for test_name, data in [("A100", a100_data), ("L40", l40_data), ("Cross", cross_data)]:
        if data['lm_head']:
            item = data['lm_head'][0]
            print(f"  {test_name}: shape={item['shape']}, min={item['min']:.6f}, max={item['max']:.6f}, mean={item['mean']:.8f}")
    
    # Compare sample_input (hidden states going into lm_head)
    print("\n" + "=" * 80)
    print("SAMPLE HIDDEN STATES (INPUT TO LM_HEAD)")
    print("=" * 80)
    
    for test_name, data in [("A100", a100_data), ("L40", l40_data), ("Cross", cross_data)]:
        if data['sample_input']:
            print(f"\n  {test_name}:")
            for i, item in enumerate(data['sample_input'][:3]):  # First 3 occurrences
                print(f"    [{i}] min={item['min']:.4f}, max={item['max']:.4f}, mean={item['mean']:.6f}, std={item['std']:.4f}")
    
    # Compare first token logits in detail
    print("\n" + "=" * 80)
    print("FIRST TOKEN LOGITS COMPARISON")
    print("=" * 80)
    
    a100_logits_by_idx = group_by_tok_idx(a100_data['logits'])
    l40_logits_by_idx = group_by_tok_idx(l40_data['logits'])
    cross_logits_by_idx = group_by_tok_idx(cross_data['logits'])
    
    print("\n[Token 0 - First generated token]")
    for test_name, logits_data in [("A100", a100_logits_by_idx), ("L40", l40_logits_by_idx), ("Cross", cross_logits_by_idx)]:
        if 0 in logits_data:
            for i, item in enumerate(logits_data[0][:2]):  # First 2 occurrences
                print(f"  {test_name} ({i}): top5_tokens={item['top5_tokens']}")
                print(f"           top5_logits={item['top5_logits']}")
    
    # Check for obvious issues
    print("\n" + "=" * 80)
    print("DIAGNOSTIC CHECKS")
    print("=" * 80)
    
    # Check 1: Are there multiple occurrences of same tok_idx? (indicates multiple forward passes)
    for test_name, send_data in [("A100", a100_send_by_idx), ("L40", l40_send_by_idx), ("Cross", cross_send_by_idx)]:
        multi_occur = {idx: len(items) for idx, items in send_data.items() if len(items) > 1}
        if multi_occur:
            print(f"\n  {test_name}: Multiple occurrences of same tok_idx in SEND: {multi_occur}")
            # Check if these are consistent
            for idx, count in list(multi_occur.items())[:3]:
                items = send_data[idx]
                norms = [item['norm'] for item in items]
                if len(set(norms)) > 1:
                    print(f"    WARNING: tok_idx={idx} has DIFFERENT norms across occurrences: {norms}")
    
    # Check 2: Do A100 and L40 have same hidden state values at tok_idx 0?
    print("\n  Comparing A100 vs L40 tok_idx=0 first occurrence:")
    if 0 in a100_send_by_idx and 0 in l40_send_by_idx:
        a100_item = a100_send_by_idx[0][0]
        l40_item = l40_send_by_idx[0][0]
        norm_diff = abs(a100_item['norm'] - l40_item['norm'])
        first8_diff = sum(abs(a - b) for a, b in zip(a100_item['first8'], l40_item['first8'])) / 8
        print(f"    A100 norm: {a100_item['norm']:.4f}, L40 norm: {l40_item['norm']:.4f}, diff: {norm_diff:.4f}")
        print(f"    A100 first8: {a100_item['first8']}")
        print(f"    L40  first8: {l40_item['first8']}")
        print(f"    Mean absolute diff in first8: {first8_diff:.6f}")
    
    # Check 3: Analyze the large norm values (second occurrence)
    print("\n  Investigating large norm values (occurrence 2):")
    for test_name, send_data in [("A100", a100_send_by_idx), ("L40", l40_send_by_idx)]:
        if 0 in send_data and len(send_data[0]) > 1:
            item = send_data[0][1]  # Second occurrence
            print(f"    {test_name} tok_idx=0 (2nd): norm={item['norm']:.4f}")
            print(f"      first8: {item['first8']}")
            # This large norm suggests it might be pre-RMSNorm or something else
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
