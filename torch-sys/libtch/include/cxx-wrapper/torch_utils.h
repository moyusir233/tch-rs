#ifndef TCH_TORCH_UTILS_H
#define TCH_TORCH_UTILS_H

#include <c10/cuda/CUDACachingAllocator.h>

void empty_cache();

int8_t current_device();

#endif //TCH_TORCH_UTILS_H
