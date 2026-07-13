// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <mutex>
#include <torch/extension.h>
#include <unordered_map>
#include <utility>

#include "allocator.hpp"
#include "constants.hpp"
#include "cuda_utils.hpp"
#include "ftensor.hpp"
#include "page.hpp"
#include "torch_utils.hpp"

namespace kvcached {
// Global configurable page size
size_t kPageSize = 2 * 1024 * 1024; // Default 2MB

std::unordered_map<int64_t, std::unique_ptr<FTensorAllocator>>
    FTensorAllocator::g_allocators_;
std::mutex FTensorAllocator::g_allocator_mutex_;
torch::Device FTensorAllocator::g_device_(torch::kCPU);
bool FTensorAllocator::g_contiguous_layout_ = false;

static inline std::shared_ptr<Page> make_shared_page(const torch::Device &dev,
                                                     page_id_t page_id,
                                                     size_t page_size = 0) {
  if (dev.is_cuda()) {
    return std::make_shared<GPUPage>(page_id, dev.index(), page_size);
  } else if (dev.is_cpu()) {
    return std::make_shared<CPUPage>(page_id, page_size);
  }
  ASSERT(false, "Unsupported device type.");
  return nullptr;
}

static inline size_t get_v_base_offset(const torch::Tensor &tensor) {
  size_t num_eles = tensor.numel() * tensor.element_size();
  ASSERT(num_eles % (2 * kPageSize) == 0,
         "Invalid tensor size: %zu, must be a multiple of 2 * page size %zu",
         num_eles, 2 * kPageSize);
  return num_eles / 2;
}

FTensorAllocator::FTensorAllocator(const torch::Device &device,
                                   bool contiguous_layout)
    : dev_(device), num_layers_(0), layer_group_granularity_(1),
      contiguous_layout_(contiguous_layout), layer_group_layout_(false),
      unified_pool_(false), kv_tensor_size_per_layer_(0) {
  if (dev_.is_cuda()) {
    init_cuda_();
  }
}

FTensorAllocator::~FTensorAllocator() { destroy(); }

void FTensorAllocator::destroy() {
  std::lock_guard<std::mutex> lock(mtx_);
  ftensors_.clear();
  contiguous_kv_tensor_.reset();
  zero_page_.reset();
  mapped_offset_refcounts_.clear();
}

void FTensorAllocator::init(const std::string &dev_str, size_t page_size,
                            bool contiguous_layout) {
  std::lock_guard<std::mutex> lock(g_allocator_mutex_);
  if (!g_allocators_.empty()) {
    LOGGER(ERROR, "FTensorAllocator has been initialized. Re-initializing...");
    g_allocators_.clear();
  }

  // Set global page size if provided (0 means use default)
  if (page_size > 0) {
    // Validate that page_size is a multiple of 2MB
    size_t base_size = 2 * 1024 * 1024; // 2MB
    if (page_size % base_size != 0) {
      LOGGER(
          ERROR,
          "Invalid page size: %zu, must be a multiple of 2MB (2097152 bytes)",
          page_size);
      abort();
    }
    kPageSize = page_size;
  }

  auto device = torch::Device(dev_str);
  g_device_ = device;
  g_contiguous_layout_ = contiguous_layout;
  g_allocators_[0] =
      std::make_unique<FTensorAllocator>(device, contiguous_layout);
}

FTensorAllocator *FTensorAllocator::global_allocator(int64_t group_id) {
  std::lock_guard<std::mutex> lock(g_allocator_mutex_);
  auto it = g_allocators_.find(group_id);
  if (it == g_allocators_.end()) {
    // Lazily create a new allocator for this group,
    // using the device/layout from init().
    assert(!g_allocators_.empty() &&
           "FTensorAllocator::init() must be called first");
    g_allocators_[group_id] =
        std::make_unique<FTensorAllocator>(g_device_, g_contiguous_layout_);
    return g_allocators_[group_id].get();
  }
  return it->second.get();
}

void FTensorAllocator::shutdown() {
  std::lock_guard<std::mutex> lock(g_allocator_mutex_);
  g_allocators_.clear();
}

std::vector<torch::Tensor> FTensorAllocator::create_kv_tensors(
    size_t size, torch::Dtype dtype, const std::string &dev_str,
    int64_t num_layers, int64_t num_kv_buffers, bool unified_pool,
    bool layer_group_layout, int64_t layer_group_granularity,
    size_t contiguous_page_size, size_t contiguous_total_size) {
  std::lock_guard<std::mutex> lock(mtx_);

  assert(num_layers_ == 0 || num_layers_ == num_layers);
  num_layers_ = num_layers;
  layer_group_layout_ = layer_group_layout;
  if (layer_group_layout_) {
    if (layer_group_granularity <= 0) {
      layer_group_granularity = 1;
    }
    if (layer_group_granularity > num_layers) {
      layer_group_granularity = num_layers;
    }
    ASSERT(num_layers % layer_group_granularity == 0,
           "num_layers (%ld) must be divisible by layer_group_granularity (%ld)",
           num_layers, layer_group_granularity);
  } else {
    layer_group_granularity = 1;
  }
  layer_group_granularity_ = layer_group_granularity;
  unified_pool_ = unified_pool;
  // Ensure size is aligned to page size.
  size_t aligned_size = size;
  if (size % kPageSize != 0) {
    aligned_size = ((size + kPageSize - 1) / kPageSize) * kPageSize;
    LOGGER(WARNING, "Size %zu is not aligned to page size %zu, aligning to %zu",
           size, kPageSize, aligned_size);
  }
  kv_tensor_size_per_layer_ = aligned_size;

  if (contiguous_layout_ && !layer_group_layout_) {
    // Upstream contiguous layout: one FTensor and one compound page covers
    // every layer for the same page id. kPageSize is the user-facing K+V
    // total physical block size; split K/V layouts derive per-buffer slices
    // internally instead of exposing separate physical-block units.
    ASSERT(num_kv_buffers > 0, "num_kv_buffers must be positive.");
    ASSERT(kPageSize % static_cast<size_t>(num_kv_buffers) == 0,
           "K+V physical block size %zu must be divisible by num_kv_buffers "
           "%ld",
           kPageSize, num_kv_buffers);
    size_t per_buffer_page_size =
        kPageSize / static_cast<size_t>(num_kv_buffers);
    size_t compound_page_size =
        per_buffer_page_size * num_layers * num_kv_buffers;
    zero_page_ = make_shared_page(dev_, ZERO_PAGE_ID, compound_page_size);
    return create_kv_tensors_contiguous_(aligned_size * num_layers, dtype,
                                         dev_str, compound_page_size);
  } else if (layer_group_layout_) {
    // Layer-stacked layout: one FTensor with groups of layers packed into
    // smaller compound pages. This is intentionally distinct from the
    // upstream contiguous-layout meaning above.
    size_t page_size = contiguous_page_size > 0
                           ? contiguous_page_size
                           : kPageSize * layer_group_granularity_;
    size_t total_size =
        contiguous_total_size > 0 ? contiguous_total_size : aligned_size * num_layers;
    ASSERT(page_size % kPageSize == 0,
           "contiguous page size %zu must be a multiple of base page size %zu",
           page_size, kPageSize);
    ASSERT(total_size % page_size == 0,
           "contiguous tensor size %zu must be aligned to page size %zu",
           total_size, page_size);
    zero_page_ = make_shared_page(dev_, ZERO_PAGE_ID, page_size);
    return create_kv_tensors_contiguous_(total_size, dtype, dev_str, page_size);
  } else {
    ASSERT(num_kv_buffers > 0, "num_kv_buffers must be positive.");
    ASSERT(kPageSize % static_cast<size_t>(num_kv_buffers) == 0,
           "K+V physical block size %zu must be divisible by num_kv_buffers "
           "%ld",
           kPageSize, num_kv_buffers);
    size_t per_buffer_page_size =
        unified_pool ? kPageSize
                     : kPageSize / static_cast<size_t>(num_kv_buffers);
    zero_page_ = make_shared_page(dev_, ZERO_PAGE_ID, per_buffer_page_size);
    return create_kv_tensors_per_layer_(kv_prefix, aligned_size, dtype, dev_str,
                                        num_layers, per_buffer_page_size);
  }
}

bool FTensorAllocator::kv_tensors_created() {
  std::lock_guard<std::mutex> lock(mtx_);
  return num_layers_ > 0;
}

bool FTensorAllocator::map_to_kv_tensors(const std::vector<offset_t> &offsets) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (num_layers_ == 0) {
    LOGGER(ERROR, "try to map to KV tensors when KV tensors are not created");
    return false;
  }

  std::vector<std::pair<FTensor *, offset_t>> mapped_offsets;
  auto decrement_mapped_refcount = [&](offset_t offset) {
    auto it = mapped_offset_refcounts_.find(offset);
    if (it == mapped_offset_refcounts_.end()) {
      return;
    }
    if (--it->second <= 0) {
      mapped_offset_refcounts_.erase(it);
    }
  };
  auto rollback_mapped_offsets = [&]() {
    for (auto it = mapped_offsets.rbegin(); it != mapped_offsets.rend(); ++it) {
      if (!it->first->unmap(it->second)) {
        LOGGER(ERROR,
               "FTensorAllocator failed to roll back mapped KV tensor "
               "offset=%ld",
               it->second);
      }
      decrement_mapped_refcount(it->second);
    }
  };
  auto map_or_fail = [&](FTensor *ftensor, offset_t offset) {
    try {
      if (ftensor->map(offset)) {
        mapped_offsets.emplace_back(ftensor, offset);
        mapped_offset_refcounts_[offset]++;
        return true;
      }
      LOGGER(ERROR, "FTensorAllocator failed to map KV tensor offset=%ld",
             offset);
      rollback_mapped_offsets();
      return false;
    } catch (...) {
      rollback_mapped_offsets();
      throw;
    }
  };

  if (contiguous_layout_ || layer_group_layout_) {
    // In contiguous layout, use the single contiguous tensor for mapping
    // Each offset maps a block that contains all layers
    auto ftensor = contiguous_kv_tensor_.get();
    auto tensor = ftensor->get_tensor();

    for (auto offset : offsets) {
      // Map K and V regions for this block (covers all layers)
      if (!map_or_fail(ftensor, offset)) {
        return false;
      }
    }
  } else if (unified_pool_) {
    // Unified pool: K and V share a single block-interleaved FTensor per
    // layer. Each page id maps exactly one VMM page at pid * page_size.
    for (int64_t i = 0; i < num_layers_; i++) {
      auto kv_name = std::string(kv_prefix) + std::to_string(i);
      auto ftensor = ftensors_[kv_name].get();
      for (auto offset : offsets) {
        if (!map_or_fail(ftensor, offset)) {
          return false;
        }
      }
    }
  } else {
    // Original per-layer mapping
    for (int64_t i = 0; i < num_layers_; i++) {
      auto kv_name = std::string(kv_prefix) + std::to_string(i);
      auto ftensor = ftensors_[kv_name].get();
      /**
       * NOTE: we assume the K tensor and the V tensor are stacked at the 1st
       * dim. This is used for calculating the offset of the V tensor.
       * FIXME: (YIFAN) we may support other KV cache layouts later.
       */
      auto tensor = ftensor->get_tensor();
      auto v_base_offset = get_v_base_offset(tensor);
      for (auto offset : offsets) {
        auto koffset = offset;
        auto voffset = offset + v_base_offset;
        if (!map_or_fail(ftensor, koffset) ||
            !map_or_fail(ftensor, voffset)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool FTensorAllocator::unmap_from_kv_tensors(
    const std::vector<offset_t> &offsets) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (num_layers_ == 0) {
    LOGGER(ERROR,
           "try to unmap from KV tensors when KV tensors are not created");
    return false;
  }

  if (contiguous_layout_ || layer_group_layout_) {
    // In contiguous layout, unmap using the single contiguous tensor
    auto ftensor = contiguous_kv_tensor_.get();
    auto tensor = ftensor->get_tensor();

    for (auto offset : offsets) {
      // Unmap K and V regions for this block (covers all layers)
      if (!ftensor->unmap(offset)) {
        LOGGER(ERROR, "FTensorAllocator failed to unmap KV tensor offset=%ld",
               offset);
        return false;
      }
      auto it = mapped_offset_refcounts_.find(offset);
      if (it != mapped_offset_refcounts_.end()) {
        if (--it->second <= 0) {
          mapped_offset_refcounts_.erase(it);
        }
      }
    }
  } else if (unified_pool_) {
    // Unified pool: single unmap per pid.
    for (int64_t i = 0; i < num_layers_; i++) {
      auto kv_name = std::string(kv_prefix) + std::to_string(i);
      auto ftensor = ftensors_[kv_name].get();
      for (auto offset : offsets) {
        if (!ftensor->unmap(offset)) {
          LOGGER(ERROR,
                 "FTensorAllocator failed to unmap KV tensor offset=%ld",
                 offset);
          return false;
        }
        auto it = mapped_offset_refcounts_.find(offset);
        if (it != mapped_offset_refcounts_.end()) {
          if (--it->second <= 0) {
            mapped_offset_refcounts_.erase(it);
          }
        }
      }
    }
  } else {
    // Original per-layer unmapping
    for (int64_t i = 0; i < num_layers_; i++) {
      auto kv_name = std::string(kv_prefix) + std::to_string(i);
      auto ftensor = ftensors_[kv_name].get();
      /**
       * NOTE: we assume the K tensor and the V tensor are stacked at the 1st
       * dim. This is used for calculating the offset of the V tensor.
       * FIXME: (YIFAN) we may support other KV cache layouts later.
       */
      auto tensor = ftensor->get_tensor();
      auto v_base_offset = get_v_base_offset(tensor);
      for (auto offset : offsets) {
        if (!ftensor->unmap(offset)) {
          LOGGER(ERROR,
                 "FTensorAllocator failed to unmap K tensor offset=%ld",
                 offset);
          return false;
        }
        auto k_it = mapped_offset_refcounts_.find(offset);
        if (k_it != mapped_offset_refcounts_.end()) {
          if (--k_it->second <= 0) {
            mapped_offset_refcounts_.erase(k_it);
          }
        }
        if (!ftensor->unmap(offset + v_base_offset)) {
          LOGGER(ERROR,
                 "FTensorAllocator failed to unmap V tensor offset=%ld",
                 offset + v_base_offset);
          return false;
        }
        auto v_it = mapped_offset_refcounts_.find(offset + v_base_offset);
        if (v_it != mapped_offset_refcounts_.end()) {
          if (--v_it->second <= 0) {
            mapped_offset_refcounts_.erase(v_it);
          }
        }
      }
    }
  }
  return true;
}

std::vector<offset_t>
FTensorAllocator::debug_unmapped_offsets(const std::vector<offset_t> &offsets) {
  std::lock_guard<std::mutex> lock(mtx_);
  std::vector<offset_t> unmapped;
  unmapped.reserve(offsets.size());
  for (auto offset : offsets) {
    auto it = mapped_offset_refcounts_.find(offset);
    if (it == mapped_offset_refcounts_.end() || it->second <= 0) {
      unmapped.push_back(offset);
    }
  }
  return unmapped;
}

std::string FTensorAllocator::get_anon_tensor_name_() {
  static constexpr std::string_view prefix = "anon_tensor_";
  static std::atomic<int> counter(0);
  return std::string(prefix) + std::to_string(counter++);
}

std::vector<torch::Tensor> FTensorAllocator::create_kv_tensors_per_layer_(
    std::string_view prefix, size_t size, torch::Dtype dtype,
    const std::string &dev_str, int64_t num_layers, size_t page_size) {
  std::vector<torch::Tensor> ftensors;
  for (int64_t i = 0; i < num_layers; i++) {
    auto name = std::string(prefix) + std::to_string(i);
    auto tensor = create_ftensor_(size, dtype, dev_str, name, page_size);
    ftensors.push_back(tensor);
  }
  return ftensors;
}

std::vector<torch::Tensor> FTensorAllocator::create_kv_tensors_contiguous_(
    size_t total_kv_size, torch::Dtype dtype, const std::string &dev_str,
    size_t page_size) {
  // Create the single contiguous KV tensor (contains K and V for all layers)
  auto contiguous_name = std::string(kv_prefix) + "contiguous";
  contiguous_kv_tensor_ =
      std::make_unique<FTensor>(contiguous_name, total_kv_size, dtype, dev_,
                                zero_page_, page_size);

  // Get the contiguous tensor
  auto contiguous_tensor = contiguous_kv_tensor_->get_tensor();
  return {contiguous_tensor};
}

/** this function is not thread-safe */
torch::Tensor FTensorAllocator::create_ftensor_(size_t size, torch::Dtype dtype,
                                                const std::string &dev_str,
                                                std::string name,
                                                size_t page_size) {
  if (name.empty())
    name = get_anon_tensor_name_();

  if (ftensors_.find(name) != ftensors_.end()) {
    auto tensor = ftensors_[name].get()->get_tensor();
    assert(tensor.numel() * tensor.element_size() == size);
    assert(tensor.device() == torch::Device(dev_str));
    return tensor;
  }

  // Create a new FTensor
  ftensors_[name] =
      std::make_unique<FTensor>(name, size, dtype, dev_, zero_page_, page_size);
  return ftensors_[name]->get_tensor();
}

/** this function is not thread-safe */
void FTensorAllocator::free_ftensor_(torch::Tensor &ftensor) {
  auto name = ftensor.name();
  if (ftensors_.find(name) == ftensors_.end()) {
    return;
  }
  ftensors_.erase(name);
}

void FTensorAllocator::init_cuda_() {
  CHECK_RT(cudaFree(0));

  CUdevice dev;
  CHECK_DRV(cuCtxGetDevice(&dev));

  int supportsVMM = 0;
  CHECK_DRV(cuDeviceGetAttribute(
      &supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
      dev));
  // LOGE("Supports VMM: %d", supportsVMM);

  CUcontext context;
  CHECK_DRV(cuCtxGetCurrent(&context));

  CUmemAllocationProp prop{
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev,
          },
  };

  size_t chunk_sz = 0;
  CHECK_DRV(cuMemGetAllocationGranularity(&chunk_sz, &prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  ASSERT(kPageSize % chunk_sz == 0,
         "Invalid page size: %lu must be a multiple of CUDA granularity %lu\n",
         kPageSize, chunk_sz);
}

} // namespace kvcached
