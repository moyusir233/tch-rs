#ifndef TCH_TORCH_PROCESS_GROUP_NCCL_H
#define TCH_TORCH_PROCESS_GROUP_NCCL_H

#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>

#include <memory>

namespace c10d {
    struct ProcessGroupNCCLOptions : ProcessGroupNCCL::Options {
        explicit ProcessGroupNCCLOptions(
                std::int64_t timeout,
                bool is_high_priority_stream
        ) : ProcessGroupNCCL::Options(is_high_priority_stream) {
            this->timeout = std::chrono::milliseconds(timeout);
        }
    };

    std::unique_ptr<TCPStoreOptions> new_tcp_store_options(
            std::uint16_t port,
            bool is_server,
            std::size_t num_workers,
            bool wait_workers,
            std::int64_t timeout,
            bool multi_tenant
    );

    std::unique_ptr<TCPStore> new_tcp_store(
            const char *host,
            const TCPStoreOptions &opts);

    void set_tcp_store_timeout(TCPStore &tcp_store, std::int64_t timeout);

    std::unique_ptr<PrefixStore>
    new_prefix_store_with_tcp_store(const char *prefix, std::unique_ptr<TCPStore> tcp_store);

    std::unique_ptr<PrefixStore>
    new_nested_prefix_store(const char *prefix, std::unique_ptr<PrefixStore> prefix_store);

    void set_prefix_store_timeout(PrefixStore &prefix_store, std::int64_t timeout);

    std::unique_ptr<ProcessGroupNCCLOptions> new_process_group_nccl_options(
            std::int64_t timeout,
            bool is_high_priority_stream);

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_tcp_store(
            std::unique_ptr<TCPStore> store,
            std::int32_t rank,
            std::int32_t size,
            std::unique_ptr<ProcessGroupNCCLOptions> options);

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_prefix_store(
            std::unique_ptr<PrefixStore> store,
            std::int32_t rank,
            std::int32_t size,
            std::unique_ptr<ProcessGroupNCCLOptions> options);
}

#endif // TCH_TORCH_PROCESS_GROUP_NCCL_H
