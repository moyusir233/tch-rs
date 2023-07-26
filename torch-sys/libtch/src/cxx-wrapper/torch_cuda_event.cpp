#include "torch_cuda_event.h"

#include <memory>

namespace at {
    namespace cuda {
        std::unique_ptr<at::cuda::CUDAEvent> new_cuda_event() {
            return std::make_unique<at::cuda::CUDAEvent>();
        }

        std::unique_ptr<at::cuda::CUDAEvent> new_cuda_event_with_flags(size_t flags) {
            return std::make_unique<at::cuda::CUDAEvent>(flags);
        }
    }
}
