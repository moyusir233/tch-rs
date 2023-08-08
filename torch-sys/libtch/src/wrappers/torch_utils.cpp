#include "wrappers/torch_utils.h"

#include <torch/cuda.h>

void empty_cache() {
    c10::cuda::CUDACachingAllocator::emptyCache();
}

int8_t current_device() {
    return c10::cuda::current_device();
}
