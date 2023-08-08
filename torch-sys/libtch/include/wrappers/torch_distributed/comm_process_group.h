#ifndef TCH_TORCH_PROCESS_GROUP_NCCL_H
#define TCH_TORCH_PROCESS_GROUP_NCCL_H

#include <memory>

#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>

#include "wrappers/utils.h"
#include "wrappers/torch_distributed/comm_store.h"


namespace c10d {
    struct ProcessGroupNCCLOptions {
        std::int64_t timeout = 300000;
        bool is_high_priority_stream = false;

        explicit operator c10::intrusive_ptr<ProcessGroupNCCL::Options>() const {
            auto opt = ProcessGroupNCCL::Options::create(is_high_priority_stream);
            opt->timeout = std::chrono::milliseconds(timeout);
            return opt;
        }
    };

    CREATE_CONTAINER_CLASS(_ArcProcessGroupNCCL, ProcessGroupNCCL)

    class ArcProcessGroupNCCL : public _ArcProcessGroupNCCL {
    public:
        explicit ArcProcessGroupNCCL(
                const ArcPrefixStore &store,
                std::int32_t rank,
                std::int32_t size,
                ProcessGroupNCCLOptions options
        ) : _ArcProcessGroupNCCL(c10::make_intrusive<ProcessGroupNCCL>(
                store.inner, rank, size,
                (c10::intrusive_ptr<ProcessGroupNCCL::Options>) options
        )) {}

        explicit ArcProcessGroupNCCL(
                const ArcTCPStore &store,
                std::int32_t rank,
                std::int32_t size,
                ProcessGroupNCCLOptions options
        ) : _ArcProcessGroupNCCL(c10::make_intrusive<ProcessGroupNCCL>(
                store.inner, rank, size,
                (c10::intrusive_ptr<ProcessGroupNCCL::Options>) options
        )) {}

        void set_sequence_number_for_group() {
            inner->setSequenceNumberForGroup();
        }
    };
}

#endif // TCH_TORCH_PROCESS_GROUP_NCCL_H
