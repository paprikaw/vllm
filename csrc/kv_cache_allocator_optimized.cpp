/*
 * KV Cache Allocator - Optimized Implementation
 * 
 * This module provides optimized KV cache allocation methods while maintaining
 * the required list-of-tensors interface (independent memory allocation).
 * 
 * Methods:
 * 1. allocate_python_style    - Baseline C++ implementation of Python approach
 * 2. allocate_optimized       - Optimized with no intermediate sync (RECOMMENDED)
 * 3. allocate_with_warmup     - With cache warmup for repeated allocations
 * 4. allocate_raw_cuda        - Direct cudaMalloc (for comparison)
 * 
 * All methods return: (k_cache: List[Tensor], v_cache: List[Tensor], time_ms: float)
 */

// CRITICAL: Python 3.12 compatibility - define macros BEFORE including pybind11
#define PY_SSIZE_T_CLEAN
#include <Python.h>


// Now include torch/pybind11 headers
#include <cstdint>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

// ============================================================================
// Helper Functions
// ============================================================================

inline double get_elapsed_ms(
    const std::chrono::high_resolution_clock::time_point& start,
    const std::chrono::high_resolution_clock::time_point& end
) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// CUDA error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            throw std::runtime_error(                                        \
                std::string("CUDA error in ") + __FILE__ + ":" +             \
                std::to_string(__LINE__) + " - " +                           \
                cudaGetErrorString(err));                                    \
        }                                                                    \
    } while (0)

// ============================================================================
// Method 1: Python-style allocation (baseline)
// ============================================================================

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, double>
allocate_python_style(
    int64_t size,
    const std::vector<int64_t>& block_shape,
    torch::ScalarType dtype,
    torch::Device device
) {
    std::vector<torch::Tensor> k_cache;
    std::vector<torch::Tensor> v_cache;
    
    k_cache.reserve(size);
    v_cache.reserve(size);
    
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(device)
        .requires_grad(false);
    
    torch::NoGradGuard no_grad;
    
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int64_t i = 0; i < size; ++i) {
        k_cache.push_back(torch::empty(block_shape, options));
        v_cache.push_back(torch::empty(block_shape, options));
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::make_tuple(k_cache, v_cache, get_elapsed_ms(start, end));
}



// ============================================================================
// Method 5: cudaMallocAsync (CUDA 11.2+, stream-ordered allocation)
// Uses CUDA's stream-ordered memory allocator for potentially better performance
// ============================================================================

std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t, double>
allocate_with_cuda_async(
    int64_t size,
    const std::vector<int64_t>& block_shape,
    torch::ScalarType dtype,
    torch::Device device,
    int64_t stream_ptr
) {
    // Set the correct CUDA device
    int device_id = device.is_cuda() ? device.index() : 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Calculate tensor size in bytes
    int64_t numel = 1;
    for (auto dim : block_shape) {
        numel *= dim;
    }
    // Get the correct element size based on dtype
    size_t element_size = torch::elementSize(dtype);
    size_t bytes_per_tensor = numel * element_size;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use the stream provided by the caller
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    // Check available memory before allocation
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // size_t total_required = size * bytes_per_tensor * 2;  // K and V
    // if (free_mem < total_required) {
    //     throw std::runtime_error(
    //         "Insufficient GPU memory for KV cache allocation. "
    //         "Required: " + std::to_string(total_required / (1024*1024)) + " MB, "
    //         "Available: " + std::to_string(free_mem / (1024*1024)) + " MB");
    // }
    
    // Allocate all pointers using cudaMallocAsync on the specified stream
    std::vector<int64_t> k_ptrs(size);
    std::vector<int64_t> v_ptrs(size);
    
    for (int64_t i = 0; i < size; ++i) {
        void* k_ptr;
        void* v_ptr;
        CUDA_CHECK(cudaMallocAsync(&k_ptr, bytes_per_tensor, stream));
        CUDA_CHECK(cudaMallocAsync(&v_ptr, bytes_per_tensor, stream));
        k_ptrs[i] = reinterpret_cast<int64_t>(k_ptr);
        v_ptrs[i] = reinterpret_cast<int64_t>(v_ptr);
    }

    // Allocate device memory for pointer arrays
    void** k_ptrs_dev;
    void** v_ptrs_dev;
    CUDA_CHECK(cudaMallocAsync(&k_ptrs_dev, size * sizeof(void*), stream));
    CUDA_CHECK(cudaMemcpyAsync(k_ptrs_dev, k_ptrs.data(),
               size * sizeof(void*),
               cudaMemcpyHostToDevice, stream));
    
    CUDA_CHECK(cudaMallocAsync(&v_ptrs_dev, size * sizeof(void*), stream));
    CUDA_CHECK(cudaMemcpyAsync(v_ptrs_dev, v_ptrs.data(),
               size * sizeof(void*),
               cudaMemcpyHostToDevice, stream));
    // Synchronize stream to ensure all allocations are complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::make_tuple(k_ptrs, v_ptrs, 
                          reinterpret_cast<int64_t>(k_ptrs_dev), 
                          reinterpret_cast<int64_t>(v_ptrs_dev), 
                          get_elapsed_ms(start, end));
}

// Prepare and cache KV pointers on GPU for flexi attention
std::tuple<int64_t, int64_t> prepare_flexi_kv_ptrs(
    const std::vector<int64_t>& k_list,
    const std::vector<int64_t>& v_list
) {
    void** k_ptrs_dev = nullptr;
    void** v_ptrs_dev = nullptr;
    
    size_t k_size = k_list.size() * sizeof(void*);
    size_t v_size = v_list.size() * sizeof(void*);
    
    // Allocate and copy key pointers
    CUDA_CHECK(cudaMalloc(&k_ptrs_dev, k_size));
    CUDA_CHECK(cudaMemcpy(k_ptrs_dev, k_list.data(), k_size, cudaMemcpyHostToDevice));
    
    // Allocate and copy value pointers
    CUDA_CHECK(cudaMalloc(&v_ptrs_dev, v_size));
    CUDA_CHECK(cudaMemcpy(v_ptrs_dev, v_list.data(), v_size, cudaMemcpyHostToDevice));
    
    // Return as int64 to be Python-compatible
    return std::make_tuple(reinterpret_cast<int64_t>(k_ptrs_dev), 
                          reinterpret_cast<int64_t>(v_ptrs_dev));
}


void free_page_list(
    const int64_t ptr,
    int device_id,
    int64_t stream_ptr
) {
    // Set the correct CUDA device
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Use the stream provided by the caller
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    void *ptr_dev = reinterpret_cast<void*>(ptr); 

    // Free each pointer using cudaFreeAsync
    if (ptr_dev != nullptr) {
        CUDA_CHECK(cudaFreeAsync(ptr_dev, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}
// ============================================================================
// Free memory allocated by cudaMallocAsync
// ============================================================================

void free_cache(
    const std::vector<int64_t>& ptrs,
    int device_id,
    int64_t stream_ptr
) {
    if (ptrs.empty()) {
        return;
    }
    
    // Set the correct CUDA device
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Use the stream provided by the caller
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    // Free each pointer using cudaFreeAsync
    for (int64_t ptr_int : ptrs) {
        void* ptr = reinterpret_cast<void*>(ptr_int);
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFreeAsync(ptr, stream));
        }
    }
    
    // Synchronize the stream to ensure memory is actually released
    // This is important because cudaFreeAsync only schedules the free operation
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ============================================================================
// Trim CUDA Memory Pool - Force release retained memory
// ============================================================================

void trim_memory_pool(int device_id, size_t min_bytes_to_keep) {
    CUDA_CHECK(cudaSetDevice(device_id));
    
    cudaMemPool_t mempool;
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device_id));
    
    // Trim the pool to release retained memory back to the OS
    // min_bytes_to_keep: minimum bytes the pool should keep (0 = release all)
    CUDA_CHECK(cudaMemPoolTrimTo(mempool, min_bytes_to_keep));
}

// ============================================================================
// Python Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Optimized KV cache allocator with multiple strategies";
    
    m.def("allocate_python_style", 
          &allocate_python_style,
          "Baseline C++ implementation mimicking Python approach",
          py::arg("size"),
          py::arg("block_shape"),
          py::arg("dtype"),
          py::arg("device"));
    
    m.def("allocate_with_cuda_async", 
          &allocate_with_cuda_async,
          "cudaMallocAsync (CUDA 11.2+, stream-ordered) - minimal PyTorch overhead",
          py::arg("size"),
          py::arg("block_shape"),
          py::arg("dtype"),
          py::arg("device"),
          py::arg("stream_ptr"));
    
    m.def("free_cache",
          &free_cache,
          "Free memory allocated by cudaMallocAsync",
          py::arg("ptrs"),
          py::arg("device_id"),
          py::arg("stream_ptr"));

    m.def("prepare_flexi_kv_ptrs",
          &prepare_flexi_kv_ptrs,
          "Prepare and cache KV pointers on GPU for flexi attention",
          py::arg("k_list"),
          py::arg("v_list"));

    m.def("free_page_list",
          &free_page_list,
          "Free page list memory allocated by cudaMallocAsync",
          py::arg("ptrs"),
          py::arg("device_id"),
          py::arg("stream_ptr"));

    m.def("trim_memory_pool",
          &trim_memory_pool,
          "Trim CUDA memory pool to release retained memory back to OS",
          py::arg("device_id"),
          py::arg("min_bytes_to_keep") = 0);
}
