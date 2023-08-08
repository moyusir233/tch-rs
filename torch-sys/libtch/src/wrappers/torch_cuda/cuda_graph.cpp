#include "wrappers/torch_cuda/cuda_graph.h"
#include "wrappers/torch_cuda/cuda_graph/cxx_wrappers.rs.h"

namespace at {
    namespace cuda {
        std::unique_ptr<CUDAGraph> new_cuda_graph() {
            return std::make_unique<CUDAGraph>();
        }

        MemPoolId create_graph_pool_id() {
            auto id = graph_pool_handle();
            return {.first = id.first, .second = id.second};
        }

        void begin_capture(CUDAGraph &cuda_graph, MemPoolId pool) {
            cuda_graph.capture_begin({pool.first, pool.second});
        }

        MemPoolId get_pool_id(CUDAGraph &cuda_graph) {
            auto id = cuda_graph.pool();
            return {.first = id.first, .second = id.second};
        }
    }

}

