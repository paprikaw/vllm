#!/usr/bin/env python3
"""
Build script for KV Cache Allocator C++ Extension

This script compiles the kv_cache_allocator_optimized extension and
installs it so it can be imported directly without runtime compilation.

Usage:
    python build_kv_allocator.py [--clean] [--verbose]
"""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.cpp_extension import load, CppExtension, CUDAExtension


def clean_cache(verbose=False):
    """Clean the torch extensions cache"""
    cache_dir = Path.home() / '.cache' / 'torch_extensions' / f'py{sys.version_info.major}{sys.version_info.minor}_cu{torch.version.cuda.replace(".", "")}'
    extension_cache = cache_dir / 'kv_cache_allocator_optimized'
    
    if extension_cache.exists():
        if verbose:
            print(f"Cleaning cache: {extension_cache}")
        shutil.rmtree(extension_cache)
        print("✓ Cache cleaned")
    else:
        if verbose:
            print("Cache directory not found, nothing to clean")


def build_extension(verbose=False):
    """Build the C++ extension"""
    source_file = PROJECT_ROOT / 'csrc' / 'kv_cache_allocator_optimized.cpp'
    
    if not source_file.exists():
        print(f"ERROR: Source file not found: {source_file}")
        return False
    
    print(f"Building kv_cache_allocator_optimized from {source_file}")
    print("=" * 70)
    
    try:
        module = load(
            name='kv_cache_allocator_optimized',
            sources=[str(source_file)],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            extra_cflags=['-O3'],
            verbose=verbose,
            with_cuda=True
        )
        
        print("\n" + "=" * 70)
        print("✓ Build successful!")
        
        # Get the path to the compiled extension
        import kv_cache_allocator_optimized
        module_path = kv_cache_allocator_optimized.__file__
        print(f"Extension location: {module_path}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Build failed: {e}")
        return False


def test_extension():
    """Test the compiled extension"""
    print("\nTesting extension...")
    print("=" * 70)
    
    try:
        import kv_cache_allocator_optimized
        
        # Test available functions
        print("Available functions:")
        for name in dir(kv_cache_allocator_optimized):
            if not name.startswith('_'):
                print(f"  - {name}")
        
        print("\nRunning basic tests...")
        
        # Test on cuda:0
        device = torch.device('cuda:0')
        current_stream = torch.cuda.current_stream(device)
        k_ptrs, v_ptrs, time_ms = kv_cache_allocator_optimized.allocate_with_cuda_async(
            size=100,
            block_shape=[16, 8, 128],
            dtype=torch.float16,
            device=device,
            stream_ptr=current_stream.cuda_stream
        )
        
        print(f"  ✓ cuda:0 test: {len(k_ptrs)} pointers allocated in {time_ms:.2f}ms")
        assert len(k_ptrs) == 100
        assert len(v_ptrs) == 100
        assert all(ptr != 0 for ptr in k_ptrs), "Found null pointers in k_ptrs"
        assert all(ptr != 0 for ptr in v_ptrs), "Found null pointers in v_ptrs"
        print(f"  ✓ Pointer validation passed")
        
        # Test on cuda:1 if available
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:1')
            current_stream = torch.cuda.current_stream(device)
            k1_ptrs, v1_ptrs, time_ms1 = kv_cache_allocator_optimized.allocate_with_cuda_async(
                size=100,
                block_shape=[16, 8, 128],
                dtype=torch.float16,
                device=device,
                stream_ptr=current_stream.cuda_stream
            )
            print(f"  ✓ cuda:1 test: {len(k1_ptrs)} pointers allocated in {time_ms1:.2f}ms")
        
        # Test free_cuda_async
        print("\nTesting free_cuda_async...")
        kv_cache_allocator_optimized.free_cuda_async(k_ptrs, 0, current_stream.cuda_stream)
        kv_cache_allocator_optimized.free_cuda_async(v_ptrs, 0, current_stream.cuda_stream)
        torch.cuda.synchronize()
        print(f"  ✓ Memory freed successfully")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build kv_cache_allocator_optimized C++ extension'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean cache before building'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output during compilation'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Only run tests, do not rebuild'
    )
    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip tests after building'
    )
    
    args = parser.parse_args()
    
    print("KV Cache Allocator Builder")
    print("=" * 70)
    
    # Clean cache if requested
    if args.clean:
        clean_cache(args.verbose)
    
    # Build extension
    if not args.test_only:
        success = build_extension(args.verbose)
        if not success:
            sys.exit(1)
    
    # Test extension
    if not args.no_test:
        success = test_extension()
        if not success:
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Done! Extension is ready to use.")
    print("\nTo use in Python:")
    print("  import kv_cache_allocator_optimized")
    print("  k, v, time = kv_cache_allocator_optimized.allocate_with_cuda_async(...)")


if __name__ == '__main__':
    main()
