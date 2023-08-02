use crate::cxx_wrapper::torch_comm_store::{PrefixStore, Store, TCPStore};
use cxx::UniquePtr;
use std::time::Duration;

pub use ffi::{ProcessGroupNCCL, ProcessGroupNCCLOptions};

/// 定义支持多进程之间通信的不同通信进程组,可见文档:
/// [pytorch-distributed-process_group](https://pytorch.org/docs/stable/distributed.html#initialization)
#[cxx::bridge]
pub mod ffi {

    /// 表示`c10d::ProcessGroupNCCL::Options`,详见torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp
    #[namespace = "c10d"]
    #[derive(Clone)]
    pub struct ProcessGroupNCCLOptions {
        timeout: i64,
        is_high_priority_stream: bool,
    }

    #[namespace = "c10d"]
    unsafe extern "C++" {
        include!("torch_comm_process_group.h");

        /// 表示`c10d::ProcessGroupNCCL`,详见torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp
        pub type ProcessGroupNCCL;

        type PrefixStore = crate::cxx_wrapper::torch_comm_store::ffi::PrefixStore;
        type TCPStore = crate::cxx_wrapper::torch_comm_store::ffi::TCPStore;

        /// 利用`TCPStore`创建`ProcessGroupNCCL`
        fn new_process_group_nccl_with_tcp_store(
            store: UniquePtr<TCPStore>,
            rank: i32,
            size: i32,
            options: ProcessGroupNCCLOptions,
        ) -> UniquePtr<ProcessGroupNCCL>;

        /// 利用`PrefixStore`创建`ProcessGroupNCCL`
        fn new_process_group_nccl_with_prefix_store(
            store: UniquePtr<PrefixStore>,
            rank: i32,
            size: i32,
            options: ProcessGroupNCCLOptions,
        ) -> UniquePtr<ProcessGroupNCCL>;
    }
}

pub trait FromStore<S: Store + cxx::memory::UniquePtrTarget> {
    fn from_store(
        store: UniquePtr<S>,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<Self>
    where
        Self: Sized + cxx::memory::UniquePtrTarget;
}

impl FromStore<TCPStore> for ProcessGroupNCCL {
    fn from_store(
        store: UniquePtr<TCPStore>,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<Self> {
        ffi::new_process_group_nccl_with_tcp_store(store, rank, size, options)
    }
}

impl FromStore<PrefixStore> for ProcessGroupNCCL {
    fn from_store(
        store: UniquePtr<PrefixStore>,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<Self> {
        ffi::new_process_group_nccl_with_prefix_store(store, rank, size, options)
    }
}

impl ProcessGroupNCCLOptions {
    pub fn new(timeout: Duration, is_high_priority_stream: bool) -> Self {
        Self { timeout: timeout.as_millis() as i64, is_high_priority_stream }
    }
}

impl Default for ProcessGroupNCCLOptions {
    fn default() -> Self {
        Self { timeout: 30 * 60 * 1000, is_high_priority_stream: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cxx_wrapper::torch_comm_store::MyTCPStoreOptions;
    use crate::cxx_wrapper::torch_comm_store::NestPrefixStore;

    #[test]
    fn nccl_process_group() {
        let tcp_opts = MyTCPStoreOptions {
            port: 8081,
            is_server: true,
            num_workers: 1,
            wait_workers: true,
            timeout: 10000,
            multi_tenant: false,
        };
        let tcp_store = TCPStore::new("localhost".into(), &tcp_opts);

        {
            let nccl_opts = ProcessGroupNCCLOptions::new(Duration::from_secs(10), true);
            let _nccl_process_group = ProcessGroupNCCL::from_store(tcp_store, 0, 1, nccl_opts);
        }

        let prefix_store =
            PrefixStore::nest_store("test".into(), TCPStore::new("localhost".into(), &tcp_opts));
        let nccl_opts = ProcessGroupNCCLOptions::new(Duration::from_secs(10), true);
        let _nccl_process_group = ProcessGroupNCCL::from_store(prefix_store, 0, 1, nccl_opts);
    }
}
