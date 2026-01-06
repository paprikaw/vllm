"""
Setup script for optimized KV cache allocator
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='kv_allocator',
    ext_modules=[
        CUDAExtension(
            name='kv_allocator',
            sources=['/home/bxb1/vllm_workbench/vllm/csrc/kv_cache_allocator_optimized.cpp'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
