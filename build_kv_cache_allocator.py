#!/usr/bin/env python3
"""
Build script for kv_cache_allocator extension only

This script builds only the kv_cache_allocator extension without rebuilding
the entire vLLM package. Useful for development and testing.

Usage:
    python build_kv_cache_allocator.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the project root directory
    root_dir = Path(__file__).parent.absolute()
    
    print("=" * 60)
    print("Building kv_cache_allocator extension")
    print("=" * 60)
    print(f"Project root: {root_dir}")
    
    # Create build directory
    build_dir = root_dir / "build_kv_alloc"
    build_dir.mkdir(exist_ok=True)
    
    print(f"Build directory: {build_dir}")
    
    # Get Python executable
    python_executable = sys.executable
    print(f"Python executable: {python_executable}")
    
    # CMake configure
    print("\n" + "=" * 60)
    print("Step 1: CMake Configure")
    print("=" * 60)
    
    cmake_args = [
        "cmake",
        f"-DVLLM_PYTHON_EXECUTABLE={python_executable}",
        f"-DCMAKE_INSTALL_PREFIX={root_dir}",
        "-G", "Ninja",
        ".."
    ]
    
    try:
        result = subprocess.run(
            cmake_args,
            cwd=build_dir,
            check=True,
            capture_output=False
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ CMake configure failed with exit code {e.returncode}")
        return 1
    
    # Build only kv_cache_allocator target
    print("\n" + "=" * 60)
    print("Step 2: Build kv_cache_allocator")
    print("=" * 60)
    
    build_args = [
        "cmake",
        "--build", ".",
        "--target", "kv_cache_allocator"
    ]
    
    try:
        result = subprocess.run(
            build_args,
            cwd=build_dir,
            check=True,
            capture_output=False
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed with exit code {e.returncode}")
        return 1
    
    # Install the built extension
    print("\n" + "=" * 60)
    print("Step 3: Install kv_cache_allocator")
    print("=" * 60)
    
    install_args = [
        "cmake",
        "--install", ".",
        "--component", "kv_cache_allocator"
    ]
    
    try:
        result = subprocess.run(
            install_args,
            cwd=build_dir,
            check=True,
            capture_output=False
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Install failed with exit code {e.returncode}")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Build completed successfully!")
    print("=" * 60)
    print("\nYou can now test the extension with:")
    print(f"  python test_kv_allocator.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
