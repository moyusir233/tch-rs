#include "torch_comm_process_group.h"

namespace c10d
{

    std::unique_ptr<TCPStoreOptions> new_tcp_store_options(
        std::uint16_t port,
        bool is_server,
        std::size_t num_workers,
        bool wait_workers,
        std::int64_t timeout)
    {
    }

    std::unique_ptr<TCPStore> new_tcp_store(
        rust::String host,
        const TCPStoreOptions &opts)
    {
    }

    void set_tcp_store_timeout(TCPStore &tcp_store, std::int64_t timeout)
    {
    }

    std::unique_ptr<PrefixStore> new_prefix_store_with_tcp_store(rust::String prefix, std::unique_ptr<TCPStore> tcp_store)
    {
    }

    std::unique_ptr<PrefixStore> new_nested_prefix_store(rust::String prefix, std::unique_ptr<PrefixStore> tcp_store)
    {
    }

    void set_tcp_store_timeout(PrefixStore &prefix_store, std::int64_t timeout) {}

    std::unique_ptr<ProcessGroupNCCLOptions> new_process_group_nccl_options(
        rust::String backend,
        std::int64_t timeout,
        bool is_high_priority_stream)
    {
    }

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_tcp_store(
        std::unique_ptr<TCPStore> store,
        std::int32_t rank,
        std::int32_t size,
        std::unique_ptr<ProcessGroupNCCLOptions> options) {}

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_prefix_store(
        std::unique_ptr<PrefixStore> store,
        std::int32_t rank,
        std::int32_t size,
        std::unique_ptr<ProcessGroupNCCLOptions> options) {}
}
