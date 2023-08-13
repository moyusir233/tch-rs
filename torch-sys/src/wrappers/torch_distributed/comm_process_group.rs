pub use crate::wrappers::autocxx_wrappers::torch_distributed::comm_process_group::*;
use crate::wrappers::{torch_distributed::comm_store::Store, utils::CppArcClone};
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
    #[inline]
    fn from_store(
        store: &ArcTCPStore,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<Self> {
        ArcProcessGroupNCCL::new2(store, rank.into(), size.into(), options).within_unique_ptr()
    }
}

impl FromStore<ArcPrefixStore> for ArcProcessGroupNCCL {
    #[inline]
    fn from_store(
        store: &ArcPrefixStore,
        rank: i32,
        size: i32,
        options: ProcessGroupNCCLOptions,
    ) -> UniquePtr<Self> {
        ArcProcessGroupNCCL::new1(store, rank.into(), size.into(), options).within_unique_ptr()
    }
}

impl Default for ProcessGroupNCCLOptions {
    #[inline]
    fn default() -> Self {
        moveit! {
            let default_opts=Self::new();
        }
        Self { ..*default_opts.deref() }
    }
}

impl Clone for ProcessGroupNCCLOptions {
    #[inline]
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

unsafe impl Send for ArcWork {}
unsafe impl Send for ArcProcessGroupNCCL {}

impl CppArcClone for ArcProcessGroupNCCL {
    #[inline]
    fn arc_clone(&self) -> UniquePtr<Self> {
        self.ArcProcessGroupNCCL_clone_().within_unique_ptr()
    }
}

impl CppArcClone for ArcWork {
    #[inline]
    fn arc_clone(&self) -> UniquePtr<Self> {
        self.ArcWork_clone_().within_unique_ptr()
    }
}

macro_rules! impl_from_slice_to_ptr_array {
    ($ptr_ty:ty,$impl_target_ty:ty) => {
        impl From<&[$ptr_ty]> for $impl_target_ty {
            fn from(value: &[$ptr_ty]) -> Self {
                if value.is_empty() {
                    return Self { ptr: std::ptr::null_mut(), size: 0 };
                }
                Self { ptr: value.as_ptr() as *mut $ptr_ty, size: value.len() as u64 }
            }
        }
    };
}

impl_from_slice_to_ptr_array! {
    i64,I64List
}

impl_from_slice_to_ptr_array! {
    *mut crate::C_tensor,Tensors
}

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
