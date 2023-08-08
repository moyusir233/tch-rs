#ifndef TCH_TORCH_CUDA_STREAM_H
#define TCH_TORCH_CUDA_STREAM_H

#include <c10/cuda/CUDAStream.h>
#include <memory>


namespace c10 {
    namespace cuda {
        bool equal(const CUDAStream &self, const CUDAStream &other);

        bool not_equal(const CUDAStream &self, const CUDAStream &other);

        int64_t _priority(const CUDAStream &self);

        std::unique_ptr<CUDAStream> _getStreamFromPool(bool is_high_priority, int8_t device_index);

        std::unique_ptr<CUDAStream> _getDefaultCUDAStream(int8_t device_index);

        std::unique_ptr<CUDAStream> _getCurrentCUDAStream(int8_t device_index);
    }
}

#endif //TCH_TORCH_CUDA_STREAM_H
