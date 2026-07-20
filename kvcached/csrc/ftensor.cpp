// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <cstring>
#include <sys/mman.h>

#include "constants.hpp"
#include "cuda_utils.hpp"
#include "ftensor.hpp"
#include "page.hpp"

namespace kvcached {

static std::atomic<size_t> g_vaddr_allocated_offset = 0;

static inline generic_ptr_t alloc_virtual_mem(const torch::Device &dev,
                                              size_t size) {
  size_t alignment_2mb = 2 * 1024 * 1024;
  ASSERT(size % alignment_2mb == 0,
         "alloc size not aligned."); // Ensure alignment.

  generic_ptr_t vaddr;
  size_t offset = g_vaddr_allocated_offset.fetch_add(size);
  if (dev.is_cuda()) {
    CHECK_DRV(cuMemAddressReserve(reinterpret_cast<CUdeviceptr *>(&vaddr), size,
                                  alignment_2mb, kStartAddr + offset, 0ULL));
  } else {
    vaddr = mmap(reinterpret_cast<void *>(kStartAddr + offset), size,
                 PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT(vaddr != MAP_FAILED, "mmap failed.");
  }
  // LOGE("Allocated virtual memory at %p", vaddr);
  return vaddr;
}

static inline std::unique_ptr<Page> make_unique_page(const torch::Device &dev,
                                                     page_id_t page_id,
                                                     size_t page_size = 0) {
  if (dev.is_cuda()) {
    return std::make_unique<GPUPage>(page_id, dev.index(), page_size);
  } else if (dev.is_cpu()) {
    return std::make_unique<CPUPage>(page_id, page_size);
  }
  ASSERT(false, "Unsupported device type.");
  return nullptr;
}

FTensor::FTensor(const std::string &name, size_t size, torch::Dtype dtype,
                 torch::Device dev, std::shared_ptr<Page> zero_page,
                 size_t page_size)
    : name_(name), vaddr_(nullptr), size_(size),
      page_size_(page_size > 0 ? page_size : kPageSize), dtype_(dtype),
      dev_(dev), zero_page_(zero_page) {
  vaddr_ = alloc_virtual_mem(dev_, size_);
  init_with_zero_();

  auto num_elems = static_cast<int64_t>(size / torch::elementSize(dtype_));
  auto options =
      torch::TensorOptions().dtype(dtype_).device(dev_).requires_grad(false);
  tensor_ =
      torch::from_blob(reinterpret_cast<void *>(vaddr_), {num_elems}, options);
}

FTensor::~FTensor() {
  if (vaddr_) {
    // Sparse KV tensors intentionally leave unused virtual pages unmapped.
    // Unmap only the pages that currently own physical memory; cuMemUnmap on
    // the entire sparse reservation fails as soon as it encounters a hole.
    for (const auto &[page_id, page] : mapping_) {
      (void)page;
      auto mapped_addr = static_cast<CUdeviceptr>(
          reinterpret_cast<uintptr_t>(vaddr_) + page_id * page_size_);
      CUresult res = cuMemUnmap(mapped_addr, page_size_);
      if (res != CUDA_SUCCESS) {
        const char *err = nullptr;
        (void)cuGetErrorString(res, &err);
        LOGGER(ERROR,
               "cuMemUnmap during FTensor cleanup failed for page %ld: %s",
               page_id, err ? err : "unknown");
      }
    }
    if (zero_page_mapped_) {
      CUresult res =
          cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr_), page_size_);
      if (res != CUDA_SUCCESS) {
        const char *err = nullptr;
        (void)cuGetErrorString(res, &err);
        LOGGER(ERROR,
               "cuMemUnmap during FTensor anchor cleanup failed: %s",
               err ? err : "unknown");
      }
      zero_page_mapped_ = false;
    }
    mapping_.clear(); // Release physical handles after their mappings are gone.

    CUresult res =
        cuMemAddressFree(reinterpret_cast<CUdeviceptr>(vaddr_), size_);
    if (res != CUDA_SUCCESS) {
      const char *err = nullptr;
      (void)cuGetErrorString(res, &err);
      LOGGER(ERROR, "cuMemAddressFree during FTensor cleanup failed: %s",
             err ? err : "unknown");
    }
  }
  mapping_.clear();
  zero_page_.reset();
}

bool FTensor::map(offset_t offset) {
  assert(offset % page_size_ == 0); // Ensure alignment.

  page_id_t page_id = offset / page_size_;
  if (mapping_.find(page_id) != mapping_.end()) {
    LOGGER(ERROR, "Page %ld is already mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  auto page = make_unique_page(dev_, page_id, page_size_);
  if (page_id == 0 && zero_page_mapped_) {
    CHECK_DRV(cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr), page_size_));
    zero_page_mapped_ = false;
  }
  page->map(vaddr);
  if (dev_.is_cuda()) {
    CHECK_RT(cudaMemset(vaddr, 0, page_size_));
  } else {
    std::memset(vaddr, 0, page_size_);
  }
  mapping_.emplace(page_id, std::move(page));
  return true;
}

bool FTensor::unmap(offset_t offset) {
  assert(offset % page_size_ == 0); // Ensure alignment.

  page_id_t page_id = offset / page_size_;
  if (mapping_.find(page_id) == mapping_.end()) {
    LOGGER(ERROR, "Page %ld is not mapped.", page_id);
    return false;
  }

  auto vaddr = reinterpret_cast<generic_ptr_t>(
      reinterpret_cast<uintptr_t>(vaddr_) + offset);
  CHECK_DRV(cuMemUnmap(reinterpret_cast<CUdeviceptr>(vaddr), page_size_));

  mapping_.erase(page_id);
  return true;
}

bool FTensor::map_(Page *page, offset_t offset, bool set_access) {
  assert(offset % page_size_ == 0); // Ensure alignment.
  assert(page);
  auto vaddr =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vaddr_) + offset);
  return page->map(vaddr, set_access);
}

bool FTensor::set_access_(generic_ptr_t addr, size_t size) {
  CUmemAccessDesc accessDesc_{
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev_.index(),
          },
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };
  CHECK_DRV(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(addr), size,
                           &accessDesc_, 1));
  return true;
}

bool FTensor::init_with_zero_() {
  assert(reinterpret_cast<uintptr_t>(vaddr_) % page_size_ ==
         0);                       // Ensure alignment.
  assert(size_ % page_size_ == 0); // Ensure alignment.

  // Keep the address range sparse. torch::from_blob needs the base pointer to
  // resolve to the target CUDA device, so retain one zero-page anchor at offset
  // zero. Mapping the shared zero page over the full reservation would still
  // create one CUDA VMM mapping and page-table entry per page, which can OOM at
  // startup for large reconfigurable KV capacities. Every other page is mapped
  // and zeroed only when PageAllocator hands it to the scheduler.
  zero_page_mapped_ = map_(zero_page_.get(), 0, /* set_access = */ true);
  return zero_page_mapped_;
}

} // namespace kvcached
