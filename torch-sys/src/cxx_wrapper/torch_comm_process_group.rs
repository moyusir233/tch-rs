use crate::autocxx_wrapper::torch_comm_process_group::*;
use crate::autocxx_wrapper::torch_comm_store::{ArcPrefixStore, ArcTCPStore};
use crate::cxx_wrapper::torch_comm_store::*;
use autocxx::prelude::*;
use cxx::UniquePtr;
use std::ops::Deref;

pub trait FromStore<S: Store + cxx::memory::UniquePtrTarget> {
    fn from_store(
        store: &S,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<ArcProcessGroupNCCL>
    where
        Self: Sized + cxx::memory::UniquePtrTarget;
}

impl FromStore<ArcTCPStore> for ArcProcessGroupNCCL {
    fn from_store(
        store: &ArcTCPStore,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<ArcProcessGroupNCCL> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autocxx_wrapper::torch_comm_store::MyTCPStoreOptions;

    #[test]
    fn nccl_process_group() {
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
