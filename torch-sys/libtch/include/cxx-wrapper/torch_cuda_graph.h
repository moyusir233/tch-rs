#ifndef TCH_TORCH_CUDA_GRAPH_H
#define TCH_TORCH_CUDA_GRAPH_H

#include <ATen/cuda/CUDAGraph.h>
#include <memory>

struct MemPoolId;

namespace at {
    namespace cuda {
        std::unique_ptr<CUDAGraph> new_cuda_graph();

        MemPoolId create_graph_pool_id();

        void begin_capture(CUDAGraph &cuda_graph, MemPoolId pool);

        MemPoolId get_pool_id(CUDAGraph &cuda_graph);
    }
}

#endif //TCH_TORCH_CUDA_GRAPH_H
