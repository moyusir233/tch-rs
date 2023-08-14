#include "wrappers/torch_utils.h"

#include <torch/cuda.h>

#include "wrappers/utils.h"
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/util/Logging.h>

// 用于进行一些必要的初始化工作,参考:torch/csrc/Module.cpp
void init_torch_module() {
  // 初始化libtorch库的日志功能
  c10::initLogging();

  at::internal::lazy_init_num_threads();

  at::init();
}

void empty_cache() { c10::cuda::CUDACachingAllocator::emptyCache(); }

int8_t current_device() { return c10::cuda::current_device(); }
