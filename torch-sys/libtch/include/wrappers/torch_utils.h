#ifndef TCH_TORCH_UTILS_H
#define TCH_TORCH_UTILS_H

#include <c10/cuda/CUDACachingAllocator.h>

void empty_cache();

int8_t current_device();

// 用于进行一些必要的初始化工作,参考:torch/csrc/Module.cpp
void init_torch_module();

#endif // TCH_TORCH_UTILS_H
