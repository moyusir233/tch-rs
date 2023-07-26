#include "torch_cuda_stream.h"

#include <memory>

namespace c10 {
    namespace cuda {

        bool equal(const CUDAStream &self, const CUDAStream &other) {
            return self == other;
        }

        bool not_equal(const CUDAStream &self, const CUDAStream &other) {
            return self != other;
        }

        int64_t _priority(const CUDAStream &self) {
            return self.priority();
        }

        std::unique_ptr<CUDAStream> _getStreamFromPool(bool is_high_priority, int8_t device_index) {
            auto stream = getStreamFromPool(is_high_priority, device_index);
            return std::make_unique<CUDAStream>(stream);
        }

        std::unique_ptr<CUDAStream> _getDefaultCUDAStream(int8_t device_index) {
            auto stream = getDefaultCUDAStream(device_index);
            return std::make_unique<CUDAStream>(stream);
        }

        std::unique_ptr<CUDAStream> _getCurrentCUDAStream(int8_t device_index) {
            auto stream = getCurrentCUDAStream(device_index);
            return std::make_unique<CUDAStream>(stream);
        }

    }
}
