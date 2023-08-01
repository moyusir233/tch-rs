#include "torch_comm_process_group.h"


namespace c10d {
    std::unique_ptr<TCPStoreOptions> new_tcp_store_options(
            std::uint16_t port,
            bool is_server,
            std::size_t num_workers,
            bool wait_workers,
            std::int64_t timeout,
            bool multi_tenant
    ) {
        return std::unique_ptr<TCPStoreOptions>(
                new TCPStoreOptions
                        ({
                                 .port = port,
                                 .isServer = is_server,
                                 .numWorkers=c10::optional<std::size_t>(num_workers),
                                 .waitWorkers=wait_workers,
                                 .timeout=std::chrono::milliseconds(timeout),
                                 .multiTenant=multi_tenant,
                         })
        );
    }

    std::unique_ptr<TCPStore> new_tcp_store(
            const char *host,
            const TCPStoreOptions &opts) {
        return std::make_unique<TCPStore>(std::string(host), opts);
    }

    void set_tcp_store_timeout(TCPStore &tcp_store, std::int64_t timeout) {
        tcp_store.setTimeout(std::chrono::milliseconds(timeout));
    }

    std::unique_ptr<PrefixStore>
    new_prefix_store_with_tcp_store(const char *prefix, std::unique_ptr<TCPStore> tcp_store) {
        return std::make_unique<PrefixStore>(std::string(prefix), c10::intrusive_ptr<TCPStore>(std::move(tcp_store)));
    }

    std::unique_ptr<PrefixStore>
    new_nested_prefix_store(const char *prefix, std::unique_ptr<PrefixStore> prefix_store) {
        return std::make_unique<PrefixStore>(std::string(prefix),
                                             c10::intrusive_ptr<PrefixStore>(std::move(prefix_store)));
    }

    void set_prefix_store_timeout(PrefixStore &prefix_store, std::int64_t timeout) {
        prefix_store.setTimeout(std::chrono::milliseconds(timeout));
    }

    std::unique_ptr<ProcessGroupNCCLOptions> new_process_group_nccl_options(
            std::int64_t timeout,
            bool is_high_priority_stream) {

        return std::make_unique<ProcessGroupNCCLOptions>(
                timeout, is_high_priority_stream
        );
    }

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_tcp_store(
            std::unique_ptr<TCPStore> store,
            std::int32_t rank,
            std::int32_t size,
            std::unique_ptr<ProcessGroupNCCLOptions> options) {
        return std::make_unique<ProcessGroupNCCL>(
                c10::intrusive_ptr<Store>(std::move(store)),
                rank,
                size,
                c10::intrusive_ptr<ProcessGroupNCCL::Options>(std::move(options))
        );
    }

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_prefix_store(
            std::unique_ptr<PrefixStore> store,
            std::int32_t rank,
            std::int32_t size,
            std::unique_ptr<ProcessGroupNCCLOptions> options) {
        return std::make_unique<ProcessGroupNCCL>(
                c10::intrusive_ptr<Store>(std::move(store)),
                rank,
                size,
                c10::intrusive_ptr<ProcessGroupNCCL::Options>(std::move(options))
        );
    }
}
