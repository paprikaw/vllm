// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "constants.hpp"
#include "cuda_utils.hpp"
#include "page.hpp"

namespace kvcached {

GPUPage::GPUPage(page_id_t page_id, int dev_idx, size_t page_size)
    : page_id_(page_id), dev_(dev_idx),
      page_size_(page_size > 0 ? page_size : kPageSize), handle_(0) {
  // CHECK_DRV(cuCtxGetDevice(&dev_));

  CUmemAllocationProp prop = {
      .type = CU_MEM_ALLOCATION_TYPE_PINNED,
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev_,
      },
  };
  CUresult res = cuMemCreate(&handle_, page_size_, &prop, 0);
  if (res != CUDA_SUCCESS) {
    const char *err = nullptr;
    (void)cuGetErrorString(res, &err);
    throw std::runtime_error("cuMemCreate failed for GPUPage page_id=" +
                             std::to_string(page_id_) +
                             " page_size=" +
                             std::to_string(page_size_) +
                             " cuda_error=" +
                             (err ? err : "unknown"));
  }
}

GPUPage::~GPUPage() {
  if (handle_ == 0) {
    return;
  }
  CUresult res = cuMemRelease(handle_);
  if (res != CUDA_SUCCESS) {
    const char *err = nullptr;
    (void)cuGetErrorString(res, &err);
    LOGGER(ERROR, "cuMemRelease during GPUPage cleanup failed: %s",
           err ? err : "unknown");
  }
}

bool GPUPage::map(generic_ptr_t vaddr, bool set_access) {
  CUmemAccessDesc accessDesc_{
      .location =
          {
              .type = CU_MEM_LOCATION_TYPE_DEVICE,
              .id = dev_,
          },
      .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
  };
  CHECK_DRV(cuMemMap(reinterpret_cast<CUdeviceptr>(vaddr), page_size_, 0,
                     handle_, 0));
  if (set_access)
    CHECK_DRV(cuMemSetAccess(reinterpret_cast<CUdeviceptr>(vaddr), page_size_,
                             &accessDesc_, 1));
  return true;
}

// TODO: finish CPUPage impl.
CPUPage::CPUPage(page_id_t page_id, size_t page_size)
    : page_id_(page_id), page_size_(page_size > 0 ? page_size : kPageSize),
      mapped_addr_(nullptr) {}

CPUPage::~CPUPage() {}

bool CPUPage::map(void *vaddr, bool set_access) {
  mapped_addr_ = vaddr;
  return true;
}

} // namespace kvcached
