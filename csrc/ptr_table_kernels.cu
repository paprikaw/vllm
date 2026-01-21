/*
 * CUDA kernel for efficient ptr_table update from block_table.
 * 
 * This kernel fuses the following operations:
 * 1. Flatten and convert block_table to long
 * 2. Clamp indices to valid range
 * 3. Create valid mask (block_id >= 0)
 * 4. Gather from stacked_ptr_tensors for all layers
 * 5. Apply mask and reshape
 * 
 * Input:
 *   - block_table: (batch_size, max_blocks_per_req), int32
 *   - stacked_ptr_tensors: (num_layers, num_blocks), int64 (representing uint64 pointers)
 * 
 * Output:
 *   - ptr_table: (num_layers, batch_size, max_blocks_per_req), int64
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"

namespace vllm {

// Kernel: Each thread handles one (layer, batch, block_idx) element
template <typename scalar_t>
__global__ void update_ptr_table_kernel(
    scalar_t* __restrict__ ptr_table,           // Output: (num_layers, ptr_table_batch_stride, max_blocks_per_req)
    const int32_t* __restrict__ block_table,    // Input: (block_table_batch_size, max_blocks_per_req)
    const scalar_t* __restrict__ ptr_tensors,   // Input: (num_layers, num_blocks)
    const int num_layers,
    const int batch_size,                       // Number of batches to process
    const int max_blocks_per_req,
    const int num_blocks,                       // Number of blocks in ptr_tensors
    const int ptr_table_batch_stride,           // Actual batch dimension of ptr_table
    const int block_table_batch_stride          // Actual batch dimension of block_table
) {
    // Calculate global thread index
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = num_layers * batch_size * max_blocks_per_req;
    
    if (idx >= total_elements) return;
    
    // Decompose linear index into (layer, batch, block_idx)
    const int block_idx = idx % max_blocks_per_req;
    const int batch_idx = (idx / max_blocks_per_req) % batch_size;
    const int layer_idx = idx / (max_blocks_per_req * batch_size);
    
    // Read block_id from block_table (use block_table_batch_stride for correct indexing)
    const int block_table_offset = batch_idx * max_blocks_per_req + block_idx;
    const int32_t block_id = block_table[block_table_offset];
    
    // Output offset in ptr_table (use ptr_table_batch_stride for correct indexing)
    const int ptr_table_offset = layer_idx * ptr_table_batch_stride * max_blocks_per_req + 
                                  batch_idx * max_blocks_per_req + 
                                  block_idx;
    
    if (block_id >= 0) {
        // Clamp block_id to valid range and gather
        const int clamped_block_id = min(block_id, num_blocks - 1);
        const int ptr_tensors_offset = layer_idx * num_blocks + clamped_block_id;
        ptr_table[ptr_table_offset] = ptr_tensors[ptr_tensors_offset];
    } else {
        // Invalid block_id, set to 0
        ptr_table[ptr_table_offset] = static_cast<scalar_t>(0);
    }
}

// Optimized kernel with shared memory for ptr_tensors (when num_blocks is small enough)
template <typename scalar_t, int BLOCK_DIM = 256>
__global__ void update_ptr_table_kernel_smem(
    scalar_t* __restrict__ ptr_table,           // Output: (num_layers, ptr_table_batch_stride, max_blocks_per_req)
    const int32_t* __restrict__ block_table,    // Input: (block_table_batch_size, max_blocks_per_req)
    const scalar_t* __restrict__ ptr_tensors,   // Input: (num_layers, num_blocks)
    const int num_layers,
    const int batch_size,                       // Number of batches to process
    const int max_blocks_per_req,
    const int num_blocks,
    const int ptr_table_batch_stride            // Actual batch dimension of ptr_table
) {
    // Each block handles one layer
    const int layer_idx = blockIdx.y;
    if (layer_idx >= num_layers) return;
    
    // Thread index within the layer's work
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements_per_layer = batch_size * max_blocks_per_req;
    
    if (tid >= elements_per_layer) return;
    
    const int block_idx = tid % max_blocks_per_req;
    const int batch_idx = tid / max_blocks_per_req;
    
    // Read block_id from block_table
    const int32_t block_id = block_table[batch_idx * max_blocks_per_req + block_idx];
    
    // Output offset (use ptr_table_batch_stride for correct indexing)
    const int ptr_table_offset = layer_idx * ptr_table_batch_stride * max_blocks_per_req + 
                                  batch_idx * max_blocks_per_req + block_idx;
    
    if (block_id >= 0) {
        const int clamped_block_id = min(block_id, num_blocks - 1);
        ptr_table[ptr_table_offset] = ptr_tensors[layer_idx * num_blocks + clamped_block_id];
    } else {
        ptr_table[ptr_table_offset] = static_cast<scalar_t>(0);
    }
}

void update_ptr_table_from_block_table(
    torch::Tensor& ptr_table,           // Output: (num_layers, batch_size, max_blocks_per_req), int64
    const torch::Tensor& block_table,   // Input: (batch_size, max_blocks_per_req), int32
    const torch::Tensor& ptr_tensors,   // Input: (num_layers, num_blocks), int64
    int64_t num_layers,
    int64_t batch_size
) {
    TORCH_CHECK(ptr_table.is_cuda(), "ptr_table must be a CUDA tensor");
    TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
    TORCH_CHECK(ptr_tensors.is_cuda(), "ptr_tensors must be a CUDA tensor");
    
    TORCH_CHECK(ptr_table.dtype() == torch::kInt64, "ptr_table must be int64");
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(ptr_tensors.dtype() == torch::kInt64, "ptr_tensors must be int64");
    
    TORCH_CHECK(ptr_table.is_contiguous(), "ptr_table must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(ptr_tensors.is_contiguous(), "ptr_tensors must be contiguous");
    
    const int max_blocks_per_req = block_table.size(1);
    const int num_blocks = ptr_tensors.size(1);
    
    // Get actual tensor dimensions for stride calculation
    const int ptr_table_batch_stride = ptr_table.size(1);
    const int block_table_batch_stride = block_table.size(0);
    
    TORCH_CHECK(num_layers <= ptr_tensors.size(0), 
                "num_layers exceeds ptr_tensors first dimension");
    TORCH_CHECK(batch_size <= block_table.size(0),
                "batch_size exceeds block_table first dimension");
    TORCH_CHECK(ptr_table.size(0) >= num_layers &&
                ptr_table.size(1) >= batch_size &&
                ptr_table.size(2) >= max_blocks_per_req,
                "ptr_table dimensions are insufficient");
    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(ptr_table));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const int total_elements = num_layers * batch_size * max_blocks_per_req;
    
    if (total_elements == 0) return;
    
    // Choose kernel based on workload
    constexpr int BLOCK_SIZE = 256;
    
    if (num_layers <= 128) {
        // Use 2D grid: (blocks_per_layer, num_layers)
        const int elements_per_layer = batch_size * max_blocks_per_req;
        const int blocks_per_layer = (elements_per_layer + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 grid(blocks_per_layer, num_layers);
        dim3 block(BLOCK_SIZE);
        
        update_ptr_table_kernel_smem<int64_t, BLOCK_SIZE><<<grid, block, 0, stream>>>(
            ptr_table.data_ptr<int64_t>(),
            block_table.data_ptr<int32_t>(),
            ptr_tensors.data_ptr<int64_t>(),
            num_layers,
            batch_size,
            max_blocks_per_req,
            num_blocks,
            ptr_table_batch_stride
        );
    } else {
        // Use 1D grid for large num_layers
        const int num_blocks_kernel = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        update_ptr_table_kernel<int64_t><<<num_blocks_kernel, BLOCK_SIZE, 0, stream>>>(
            ptr_table.data_ptr<int64_t>(),
            block_table.data_ptr<int32_t>(),
            ptr_tensors.data_ptr<int64_t>(),
            num_layers,
            batch_size,
            max_blocks_per_req,
            num_blocks,
            ptr_table_batch_stride,
            block_table_batch_stride
        );
    }
}

} // namespace vllm
