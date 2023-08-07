use autocxx::prelude::*;
use cxx::memory::UniquePtrTarget;
pub use ffi::*;
use std::pin::Pin;

include_cpp! {
    #include "cxx_wrapper_utils.h"
    safety!(unsafe_ffi)
    generate!("MyIntrusiveTarget")
    generate!("Wrapper")
}

pub trait CxxClone: UniquePtrTarget + Sized {
    /// 在c++一侧的堆上创建clone的新实例
    fn clone_on_cpp_heap(&self) -> UniquePtr<Self>;
    /// 在rust一侧的堆上创建clone的新实例
    fn clone_on_rust_heap(&self) -> Pin<Box<Self>>;
}

impl CxxClone for ffi::Wrapper {
    fn clone_on_cpp_heap(&self) -> UniquePtr<Self> {
        self.clone().within_unique_ptr()
    }

    fn clone_on_rust_heap(&self) -> Pin<Box<Self>> {
        self.clone().within_box()
    }
}

unsafe impl Send for ffi::Wrapper {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::JoinHandle;
    #[test]
    fn auto_cxx_utils() {
        let wrapper = ffi::Wrapper::new(ffi::MyIntrusiveTarget::new().within_unique_ptr())
            .within_unique_ptr();
        let clone_wrapper = wrapper.clone_on_cpp_heap();

        let handlers: Vec<_> = (0..5)
            .map(|_| {
                let clone_wrapper = clone_wrapper.clone_on_cpp_heap();
                std::thread::spawn(move || {
                    let _clone_wrapper = clone_wrapper.clone_on_cpp_heap();
                })
            })
            .collect();
        assert!(handlers.into_iter().map(JoinHandle::join).all(|result| result.is_ok()));
    }
}
