#ifndef TCH_TORCH_CUDA_EVENT_H
#define TCH_TORCH_CUDA_EVENT_H

#include <ATen/cuda/CUDAEvent.h>
#include <memory>

namespace at {
    namespace cuda {
        std::unique_ptr<at::cuda::CUDAEvent> new_cuda_event();

        std::unique_ptr<at::cuda::CUDAEvent> new_cuda_event_with_flags(size_t flags);

    }
}

#endif //TCH_TORCH_CUDA_EVENT_H
