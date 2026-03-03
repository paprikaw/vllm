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
#include <cuda.h>  // For VMM APIs (cuMemCreate, cuMemMap, etc.)
#include <vector>
#include <chrono>

// CUDA Driver API error checking macro
#define CU_CHECK(call)                                                       \
    do {                                                                     \
        CUresult err = call;                                                 \
        if (err != CUDA_SUCCESS) {                                           \
            const char* errStr;                                              \
            cuGetErrorString(err, &errStr);                                  \
            throw std::runtime_error(                                        \
                std::string("CUDA Driver error in ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + " - " + errStr);                  \
        }                                                                    \
    } while (0)

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

// ============================================================================
// VMM API allocation (CUDA Driver API - supports fine-grained 2MB release)
// This enables non-contiguous memory release at 2MB granularity
// ============================================================================

std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t, 
           int64_t, std::vector<int64_t>, std::vector<int64_t>, double>
allocate_with_cuda_vmm(
    int64_t size,
    const std::vector<int64_t>& block_shape,
    torch::ScalarType dtype,
    torch::Device device
) {
    // Set the correct CUDA device
    int device_id = device.is_cuda() ? device.index() : 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Initialize CUDA Driver API
    CU_CHECK(cuInit(0));
    
    // Get CUDA device handle
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id));
    
    // Calculate tensor size in bytes
    int64_t numel = 1;
    for (auto dim : block_shape) {
        numel *= dim;
    }
    size_t element_size = torch::elementSize(dtype);
    size_t bytes_per_tensor = numel * element_size;
    
    // VMM requires 2MB alignment (GPU minimum allocation granularity)
    const size_t VMM_GRANULARITY = 2 * 1024 * 1024;  // 2MB
    size_t aligned_bytes = ((bytes_per_tensor + VMM_GRANULARITY - 1) / VMM_GRANULARITY) * VMM_GRANULARITY;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Setup allocation properties
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    
    // Setup access descriptor
    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    // Reserve virtual address space for all K and V tensors
    size_t total_k_size = size * aligned_bytes;
    size_t total_v_size = size * aligned_bytes;
    
    CUdeviceptr k_va_base, v_va_base;
    CU_CHECK(cuMemAddressReserve(&k_va_base, total_k_size, VMM_GRANULARITY, 0, 0));
    CU_CHECK(cuMemAddressReserve(&v_va_base, total_v_size, VMM_GRANULARITY, 0, 0));
    
    // Allocate physical memory handles and map for each tensor
    std::vector<int64_t> k_ptrs(size);
    std::vector<int64_t> v_ptrs(size);
    std::vector<int64_t> k_handles(size);
    std::vector<int64_t> v_handles(size);
    
    for (int64_t i = 0; i < size; ++i) {
        CUmemGenericAllocationHandle k_handle, v_handle;
        
        // Allocate physical memory
        CU_CHECK(cuMemCreate(&k_handle, aligned_bytes, &prop, 0));
        CU_CHECK(cuMemCreate(&v_handle, aligned_bytes, &prop, 0));
        
        // Calculate virtual addresses
        CUdeviceptr k_va = k_va_base + i * aligned_bytes;
        CUdeviceptr v_va = v_va_base + i * aligned_bytes;
        
        // Map physical to virtual
        CU_CHECK(cuMemMap(k_va, aligned_bytes, 0, k_handle, 0));
        CU_CHECK(cuMemMap(v_va, aligned_bytes, 0, v_handle, 0));
        
        // Store pointers and handles
        k_ptrs[i] = static_cast<int64_t>(k_va);
        v_ptrs[i] = static_cast<int64_t>(v_va);
        k_handles[i] = static_cast<int64_t>(k_handle);
        v_handles[i] = static_cast<int64_t>(v_handle);
    }
    
    // Set access permissions for the entire ranges
    CU_CHECK(cuMemSetAccess(k_va_base, total_k_size, &access_desc, 1));
    CU_CHECK(cuMemSetAccess(v_va_base, total_v_size, &access_desc, 1));
    
    // Prepare GPU pointer arrays for flexi attention
    void** k_ptrs_dev;
    void** v_ptrs_dev;
    CUDA_CHECK(cudaMalloc(&k_ptrs_dev, size * sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(k_ptrs_dev, k_ptrs.data(), size * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&v_ptrs_dev, size * sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(v_ptrs_dev, v_ptrs.data(), size * sizeof(void*), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Returns: k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev, aligned_bytes, k_handles, v_handles, time_ms
    return std::make_tuple(
        k_ptrs, v_ptrs,
        reinterpret_cast<int64_t>(k_ptrs_dev),
        reinterpret_cast<int64_t>(v_ptrs_dev),
        static_cast<int64_t>(aligned_bytes),
        k_handles, v_handles,
        get_elapsed_ms(start, end)
    );
}

// ============================================================================
// VMM Combined API - K and V share the same 2MB physical page
// Each handle maps to both K and V (K at offset 0, V at offset bytes_per_tensor)
// This avoids memory waste when K/V blocks are smaller than 2MB
// ============================================================================

std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t, int64_t, 
           int64_t, int64_t, std::vector<int64_t>, double>
allocate_with_cuda_vmm_combined(
    int64_t size,
    const std::vector<int64_t>& block_shape,
    torch::ScalarType dtype,
    torch::Device device
) {
    // Set the correct CUDA device
    int device_id = device.is_cuda() ? device.index() : 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Initialize CUDA Driver API
    CU_CHECK(cuInit(0));
    
    // Get CUDA device handle
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id));
    
    // Calculate tensor size in bytes (for one K or V block)
    int64_t numel = 1;
    for (auto dim : block_shape) {
        numel *= dim;
    }
    size_t element_size = torch::elementSize(dtype);
    size_t bytes_per_tensor = numel * element_size;  // Size of one K or V block
    
    // Combined size: K + V in one allocation
    size_t combined_bytes = 2 * bytes_per_tensor;
    
    // VMM requires 2MB alignment (GPU minimum allocation granularity)
    const size_t VMM_GRANULARITY = 2 * 1024 * 1024;  // 2MB
    size_t aligned_combined_bytes = ((combined_bytes + VMM_GRANULARITY - 1) / VMM_GRANULARITY) * VMM_GRANULARITY;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Setup allocation properties
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    
    // Setup access descriptor
    CUmemAccessDesc access_desc = {};
    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id = device_id;
    access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    
    // Reserve virtual address space for all combined KV tensors
    size_t total_size = size * aligned_combined_bytes;
    
    CUdeviceptr va_base;
    CU_CHECK(cuMemAddressReserve(&va_base, total_size, VMM_GRANULARITY, 0, 0));
    
    // Allocate physical memory handles and map for each combined KV block
    std::vector<int64_t> k_ptrs(size);
    std::vector<int64_t> v_ptrs(size);
    std::vector<int64_t> kv_handles(size);  // One handle per combined KV block
    
    for (int64_t i = 0; i < size; ++i) {
        CUmemGenericAllocationHandle kv_handle;
        
        // Allocate physical memory for combined K+V
        CU_CHECK(cuMemCreate(&kv_handle, aligned_combined_bytes, &prop, 0));
        
        // Calculate virtual address for this combined block
        CUdeviceptr kv_va = va_base + i * aligned_combined_bytes;
        
        // Map physical to virtual
        CU_CHECK(cuMemMap(kv_va, aligned_combined_bytes, 0, kv_handle, 0));
        
        // K is at the start, V is offset by bytes_per_tensor
        k_ptrs[i] = static_cast<int64_t>(kv_va);
        v_ptrs[i] = static_cast<int64_t>(kv_va + bytes_per_tensor);
        kv_handles[i] = static_cast<int64_t>(kv_handle);
    }
    
    // Set access permissions for the entire range
    CU_CHECK(cuMemSetAccess(va_base, total_size, &access_desc, 1));
    
    // Prepare GPU pointer arrays for flexi attention
    void** k_ptrs_dev;
    void** v_ptrs_dev;
    CUDA_CHECK(cudaMalloc(&k_ptrs_dev, size * sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(k_ptrs_dev, k_ptrs.data(), size * sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&v_ptrs_dev, size * sizeof(void*)));
    CUDA_CHECK(cudaMemcpy(v_ptrs_dev, v_ptrs.data(), size * sizeof(void*), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Returns: k_ptrs, v_ptrs, k_ptrs_dev, v_ptrs_dev, aligned_combined_bytes, bytes_per_tensor, kv_handles, time_ms
    return std::make_tuple(
        k_ptrs, v_ptrs,
        reinterpret_cast<int64_t>(k_ptrs_dev),
        reinterpret_cast<int64_t>(v_ptrs_dev),
        static_cast<int64_t>(aligned_combined_bytes),
        static_cast<int64_t>(bytes_per_tensor),
        kv_handles,
        get_elapsed_ms(start, end)
    );
}

// ============================================================================
// Free VMM combined allocated memory (K and V share the same physical page)
// ============================================================================

void free_vmm_blocks_combined(
    const std::vector<int64_t>& k_ptrs,  // K pointers (start of combined block)
    const std::vector<int64_t>& handles,
    int64_t aligned_combined_bytes,
    int device_id
) {
    if (k_ptrs.empty()) return;
    
    // Validate that k_ptrs and handles have the same length
    if (k_ptrs.size() != handles.size()) {
        throw std::runtime_error(
            "free_vmm_blocks_combined: k_ptrs.size() (" + std::to_string(k_ptrs.size()) + 
            ") != handles.size() (" + std::to_string(handles.size()) + ")"
        );
    }
    
    CUDA_CHECK(cudaSetDevice(device_id));
    
    for (size_t i = 0; i < k_ptrs.size(); ++i) {
        // K pointer is at the start of the combined block
        CUdeviceptr va = static_cast<CUdeviceptr>(k_ptrs[i]);
        CUmemGenericAllocationHandle handle = static_cast<CUmemGenericAllocationHandle>(handles[i]);
        
        // Unmap the entire combined block and release physical memory
        CU_CHECK(cuMemUnmap(va, aligned_combined_bytes));
        CU_CHECK(cuMemRelease(handle));
    }
}

// ============================================================================
// Free VMM allocated memory (supports non-contiguous release)
// ============================================================================

void free_vmm_blocks(
    const std::vector<int64_t>& ptrs,
    const std::vector<int64_t>& handles,
    int64_t aligned_bytes,
    int device_id
) {
    if (ptrs.empty()) return;
    
    CUDA_CHECK(cudaSetDevice(device_id));
    
    for (size_t i = 0; i < ptrs.size(); ++i) {
        CUdeviceptr va = static_cast<CUdeviceptr>(ptrs[i]);
        CUmemGenericAllocationHandle handle = static_cast<CUmemGenericAllocationHandle>(handles[i]);
        
        // Unmap and release physical memory
        CU_CHECK(cuMemUnmap(va, aligned_bytes));
        CU_CHECK(cuMemRelease(handle));
    }
}

// Free VMM virtual address range
void free_vmm_va_range(
    int64_t va_base,
    int64_t total_size,
    int device_id
) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CU_CHECK(cuMemAddressFree(static_cast<CUdeviceptr>(va_base), total_size));
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

    m.def("allocate_with_cuda_vmm",
          &allocate_with_cuda_vmm,
          "VMM API allocation (supports fine-grained 2MB release)",
          py::arg("size"),
          py::arg("block_shape"),
          py::arg("dtype"),
          py::arg("device"));

    m.def("allocate_with_cuda_vmm_combined",
          &allocate_with_cuda_vmm_combined,
          "VMM API allocation with K and V combined in same physical page (saves memory)",
          py::arg("size"),
          py::arg("block_shape"),
          py::arg("dtype"),
          py::arg("device"));

    m.def("free_vmm_blocks",
          &free_vmm_blocks,
          "Free VMM allocated blocks (supports non-contiguous release)",
          py::arg("ptrs"),
          py::arg("handles"),
          py::arg("aligned_bytes"),
          py::arg("device_id"));

    m.def("free_vmm_blocks_combined",
          &free_vmm_blocks_combined,
          "Free VMM combined blocks (K and V share same physical page)",
          py::arg("k_ptrs"),
          py::arg("handles"),
          py::arg("aligned_combined_bytes"),
          py::arg("device_id"));

    m.def("free_vmm_va_range",
          &free_vmm_va_range,
          "Free VMM virtual address range",
          py::arg("va_base"),
          py::arg("total_size"),
          py::arg("device_id"));
}
