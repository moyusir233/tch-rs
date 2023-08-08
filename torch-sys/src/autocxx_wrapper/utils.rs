use autocxx::prelude::*;
use cxx::memory::UniquePtrTarget;
use std::pin::Pin;

pub trait CxxClone: UniquePtrTarget + Sized {
    /// 在c++一侧的堆上创建clone的新实例
    fn clone_on_cpp_heap(&self) -> UniquePtr<Self>;
    /// 在rust一侧的堆上创建clone的新实例
    fn clone_on_rust_heap(&self) -> Pin<Box<Self>>;
}

macro_rules! impl_cxx_clone {
    ($t:path) => {
        impl $crate::autocxx_wrapper::utils::CxxClone for $t {
            fn clone_on_cpp_heap(&self) -> UniquePtr<Self> {
                self.clone().within_unique_ptr()
            }

            fn clone_on_rust_heap(&self) -> Pin<Box<Self>> {
                self.clone().within_box()
            }
        }
    };
}
