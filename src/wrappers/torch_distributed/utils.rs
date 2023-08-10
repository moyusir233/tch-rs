use cxx::UniquePtr;
use std::future::Future;

/// 类似std::future::IntoFuture,用于封装torch_sys底层的一些类型
pub trait CxxIntoFuture {
    type Output;
    type IntoFuture: Future<Output = Self::Output>;

    fn into_future(unique_ptr: UniquePtr<Self>) -> Self::IntoFuture
    where
        Self: cxx::memory::UniquePtrTarget + Sized;
}
