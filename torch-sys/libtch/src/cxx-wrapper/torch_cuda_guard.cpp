#include "torch_cuda_guard.h"

namespace c10 {
    namespace cuda {

        std::unique_ptr<CUDAGuard> new_cuda_guard(int8_t device_index) {
            return std::make_unique<CUDAGuard>(device_index);
        }

        int8_t original_device(const CUDAGuard &guard) {
            return guard.original_device().index();
        }

        int8_t current_device(const CUDAGuard &guard) {
            return guard.current_device().index();
        }

        std::unique_ptr<OptionalCUDAGuard> new_uninit_optional_cuda_guard() {
            return std::make_unique<OptionalCUDAGuard>();
        }

        std::unique_ptr<OptionalCUDAGuard> new_optional_cuda_guard(int8_t device_index) {
            return std::make_unique<OptionalCUDAGuard>(device_index);
        }

        int8_t original_device(const OptionalCUDAGuard &guard) {
            auto device = guard.original_device();
            return device.has_value() ? device->index() : -1;
        }

        int8_t current_device(const OptionalCUDAGuard &guard) {
            auto device = guard.current_device();
            return device.has_value() ? device->index() : -1;
        }

        std::unique_ptr<CUDAStreamGuard> new_cuda_stream_guard(const CUDAStream &stream) {
            return std::make_unique<CUDAStreamGuard>(stream);
        }

        void reset_stream(CUDAStreamGuard &guard, const CUDAStream &stream) {
            guard.reset_stream(stream);
        }

        std::unique_ptr<CUDAStream> original_stream(const CUDAStreamGuard &guard) {
            return std::make_unique<CUDAStream>(guard.original_stream());
        }

        std::unique_ptr<CUDAStream> current_stream(const CUDAStreamGuard &guard) {
            return std::make_unique<CUDAStream>(guard.current_stream());
        }

        int8_t original_device(const CUDAStreamGuard &guard) {
            return guard.original_device().index();
        }

        int8_t current_device(const CUDAStreamGuard &guard) {
            return guard.current_device().index();
        }

        std::unique_ptr<OptionalCUDAStreamGuard> new_uninit_optional_cuda_stream_guard() {
            return std::make_unique<OptionalCUDAStreamGuard>();
        }

        std::unique_ptr<OptionalCUDAStreamGuard> new_optional_cuda_stream_guard(const CUDAStream &stream) {
            return std::make_unique<OptionalCUDAStreamGuard>(stream.unwrap());
        }

        void reset_stream(OptionalCUDAStreamGuard &guard, const CUDAStream &stream) {
            guard.reset_stream(stream);
        }

        std::unique_ptr<CUDAStream> original_stream(const OptionalCUDAStreamGuard &guard) {
            auto stream = guard.original_stream();
            return stream.has_value() ? std::make_unique<CUDAStream>(stream.value()) : nullptr;
        }

        std::unique_ptr<CUDAStream> current_stream(const OptionalCUDAStreamGuard &guard) {
            auto stream = guard.current_stream();
            return stream.has_value() ? std::make_unique<CUDAStream>(stream.value()) : nullptr;
        }

        std::unique_ptr<CUDAMultiStreamGuard> new_cuda_multi_stream_guard(const CUDAStream *streams, size_t length) {
            auto array_ref = c10::ArrayRef<CUDAStream>(streams, length);
            return std::make_unique<CUDAMultiStreamGuard>(array_ref);
        }

    }
}
