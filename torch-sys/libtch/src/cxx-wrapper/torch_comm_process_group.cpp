#include "torch_comm_process_group.h"


namespace c10d {
    TCPStoreOptions from_my_tcp_store_options(const MyTCPStoreOptions &opts) {
        return {
                .port=opts.port,
                .isServer=opts.is_server,
                .numWorkers=opts.num_workers,
                .waitWorkers=opts.wait_workers,
                .timeout=std::chrono::milliseconds(opts.timeout),
                .multiTenant=opts.multi_tenant
        };
    }


    c10::intrusive_ptr<ProcessGroupNCCL::Options> from_nccl_options(
            ProcessGroupNCCLOptions &opts
    ) {
        auto new_opts = ProcessGroupNCCL::Options::create(opts.is_high_priority_stream);
        new_opts->timeout = std::chrono::milliseconds(opts.timeout);
        return new_opts;
    }


    std::unique_ptr<TCPStore> new_tcp_store(
            const char *host,
            const MyTCPStoreOptions &opts) {
        return std::make_unique<TCPStore>(std::string(host), from_my_tcp_store_options(opts));
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

    std::int64_t add(PrefixStore &store, const char *key, std::int64_t value) {
        return store.add(key, value);
    }

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_tcp_store(
            std::unique_ptr<TCPStore> store,
            std::int32_t rank,
            std::int32_t size,
            ProcessGroupNCCLOptions options) {
        return std::make_unique<ProcessGroupNCCL>(
                c10::intrusive_ptr<Store>(std::move(store)),
                rank,
                size,
                from_nccl_options(options)
        );
    }

    std::unique_ptr<ProcessGroupNCCL> new_process_group_nccl_with_prefix_store(
            std::unique_ptr<PrefixStore> store,
            std::int32_t rank,
            std::int32_t size,
            ProcessGroupNCCLOptions options) {
        return std::make_unique<ProcessGroupNCCL>(
                c10::intrusive_ptr<Store>(std::move(store)),
                rank,
                size,
                from_nccl_options(options)
        );
    }
}
