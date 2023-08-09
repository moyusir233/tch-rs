pub use crate::wrappers::autocxx_wrappers::torch_distributed::comm_process_group::*;
use crate::wrappers::torch_distributed::comm_store::Store;
use autocxx::prelude::*;
use cxx::UniquePtr;
use std::ops::Deref;

pub trait FromStore<S: Store + cxx::memory::UniquePtrTarget> {
    fn from_store(
        store: &S,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<Self>
    where
        Self: Sized + cxx::memory::UniquePtrTarget;
}

impl FromStore<ArcTCPStore> for ArcProcessGroupNCCL {
    fn from_store(
        store: &ArcTCPStore,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<Self> {
        ArcProcessGroupNCCL::new1(store, rank.into(), size.into(), options).within_unique_ptr()
    }
}

impl FromStore<ArcPrefixStore> for ArcProcessGroupNCCL {
    fn from_store(
        store: &ArcPrefixStore,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<Self> {
        ArcProcessGroupNCCL::new(store, rank.into(), size.into(), options).within_unique_ptr()
    }
}

impl Default for ProcessGroupNCCLOptions {
    fn default() -> Self {
        moveit! {
            let default_opts=Self::new();
        }
        Self { ..*default_opts.deref() }
    }
}

impl Clone for ProcessGroupNCCLOptions {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

unsafe impl Send for ArcWork {}
unsafe impl Send for ArcProcessGroupNCCL {}

#[cfg(test)]
mod nccl_process_group {
    use super::*;
    use crate::wrappers::torch_distributed::comm_store::{MyTCPStoreOptions, NestPrefixStore};

    #[test]
    fn init() {
        let tcp_opts = MyTCPStoreOptions {
            port: 8081,
            isServer: true,
            numWorkers: 1,
            waitWorkers: true,
            timeout: 10000,
            multiTenant: false,
        };
        let tcp_store = ArcTCPStore::new("localhost", &tcp_opts).within_unique_ptr();

        {
            let nccl_opts =
                ProcessGroupNCCLOptions { timeout: 10000, is_high_priority_stream: true };
            let _nccl_process_group =
                ArcProcessGroupNCCL::from_store(tcp_store.as_ref().unwrap(), 0, 1, nccl_opts);
            drop(tcp_store);
        }

        let prefix_store = ArcPrefixStore::nest_store(
            "test".into(),
            ArcTCPStore::new("localhost", &tcp_opts).within_unique_ptr(),
        );

        let nccl_opts = ProcessGroupNCCLOptions { timeout: 10000, is_high_priority_stream: true };
        let mut nccl_process_group =
            ArcProcessGroupNCCL::from_store(prefix_store.as_ref().unwrap(), 0, 1, nccl_opts);

        nccl_process_group.pin_mut().set_sequence_number_for_group();
    }
}
