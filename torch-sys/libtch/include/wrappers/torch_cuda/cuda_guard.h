#ifndef TCH_TORCH_CUDA_GUARD_H
#define TCH_TORCH_CUDA_GUARD_H

#include <c10/cuda/CUDAGuard.h>
#include <memory>

namespace c10 {
    namespace cuda {
        // CUDAGuard
        std::unique_ptr<CUDAGuard> new_cuda_guard(int8_t device_index);

        int8_t original_device(const CUDAGuard &guard);

        int8_t current_device(const CUDAGuard &guard);

        // OptionalCUDAGuard
        std::unique_ptr<OptionalCUDAGuard> new_uninit_optional_cuda_guard();

        std::unique_ptr<OptionalCUDAGuard> new_optional_cuda_guard(int8_t device_index);

        int8_t original_device(const OptionalCUDAGuard &guard);

        int8_t current_device(const OptionalCUDAGuard &guard);

        // CUDAStreamGuard
        std::unique_ptr<CUDAStreamGuard> new_cuda_stream_guard(const CUDAStream &stream);

        void reset_stream(CUDAStreamGuard &guard, const CUDAStream &stream);

        std::unique_ptr<CUDAStream> original_stream(const CUDAStreamGuard &guard);

        std::unique_ptr<CUDAStream> current_stream(const CUDAStreamGuard &guard);

        int8_t original_device(const CUDAStreamGuard &guard);

        int8_t current_device(const CUDAStreamGuard &guard);

        // OptionalCUDAStreamGuard
        std::unique_ptr<OptionalCUDAStreamGuard> new_uninit_optional_cuda_stream_guard();

        std::unique_ptr<OptionalCUDAStreamGuard> new_optional_cuda_stream_guard(const CUDAStream &stream);

        void reset_stream(OptionalCUDAStreamGuard &guard, const CUDAStream &stream);

        std::unique_ptr<CUDAStream> original_stream(const OptionalCUDAStreamGuard &guard);

        std::unique_ptr<CUDAStream> current_stream(const OptionalCUDAStreamGuard &guard);

        // CUDAMultiStreamGuard
        std::unique_ptr<CUDAMultiStreamGuard>
        new_cuda_multi_stream_guard(const CUDAStream *streams,size_t length);
    }
}

#endif //TCH_TORCH_CUDA_GUARD_H
