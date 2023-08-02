#ifndef TCH_TORCH_PROCESS_GROUP_NCCL_H
#define TCH_TORCH_PROCESS_GROUP_NCCL_H

#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include "torch_comm_store.rs.h"
#include "torch_comm_process_group.rs.h"

#include <memory>

namespace c10d {
    c10::intrusive_ptr<ProcessGroupNCCL::Options> from_nccl_options(
            ProcessGroupNCCLOptions &opts
    );

    TCPStoreOptions from_my_tcp_store_options(const MyTCPStoreOptions &opts);

    std::unique_ptr<TCPStore> new_tcp_store(
            const char *host,
            const MyTCPStoreOptions &opts);

    void set_tcp_store_timeout(TCPStore &tcp_store, std::int64_t timeout);

    std::unique_ptr<PrefixStore>
    new_prefix_store_with_tcp_store(const char *prefix, std::unique_ptr<TCPStore> tcp_store);

    std::unique_ptr<PrefixStore>
    new_nested_prefix_store(const char *prefix, std::unique_ptr<PrefixStore> prefix_store);

    void set_prefix_store_timeout(PrefixStore &prefix_store, std::int64_t timeout);

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_tcp_store(
            std::unique_ptr<TCPStore> store,
            std::int32_t rank,
            std::int32_t size,
            ProcessGroupNCCLOptions options);

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_prefix_store(
            std::unique_ptr<PrefixStore> store,
            std::int32_t rank,
            std::int32_t size,
            ProcessGroupNCCLOptions options);
}

#endif // TCH_TORCH_PROCESS_GROUP_NCCL_H
