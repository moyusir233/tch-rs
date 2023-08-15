use cxx::memory::UniquePtrTarget;
pub use torch_sys::wrappers::torch_cuda::cuda_guard::*;

use crate::error::TchResult;

/// 描述被cxx包装的Guard的初始化trait
pub trait InitGuard<'a>: UniquePtrTarget + Sized {
    type GuardItem;
    fn new(guard_item: Self::GuardItem) -> cxx::UniquePtr<Self>;
}

/// 简化RAII Guard使用的trait
pub trait GuardScope<'a>: InitGuard<'a> {
    /// 将传入的闭包在相应Guard包裹的上下文中执行
    fn scope<T>(guard_item: Self::GuardItem, action: T) -> TchResult<()>
    where
        T: FnOnce() -> TchResult<()>,
    {
        let _guard = Self::new(guard_item);
        action()
    }
}

/// 用于实现GuardScope的宏
macro_rules! impl_guard_scope {
    ($guard_type:ty,$guard_item:ty) => {
        impl InitGuard<'_> for $guard_type {
            type GuardItem = $guard_item;
            fn new(guard_item: Self::GuardItem) -> cxx::UniquePtr<Self> {
                Self::new(guard_item)
            }
        }
        impl GuardScope<'_> for $guard_type{}
    };
    // 处理trait associated type GuardItem为引用时,需要显式的声明周期时
    ($guard_type:ty,$guard_item:ty,$lt:lifetime) => {
        impl<$lt> InitGuard<$lt> for $guard_type {
            type GuardItem = &$lt $guard_item;
            fn new(guard_item: Self::GuardItem) -> cxx::UniquePtr<Self> {
                Self::new(guard_item)
            }
        }
        impl GuardScope<'_> for $guard_type{}
    };
    ($($guard_type:ty,$guard_item:ty);+) => {
        $(
            impl_guard_scope!($guard_type,$guard_item);
        )+
    };
    ($($guard_type:ty,$guard_item:ty,$lt:lifetime);+) => {
        $(
            impl_guard_scope!($guard_type,$guard_item,$lt);
        )+
    };
}

impl_guard_scope!(
    CUDAGuard,i8;
    OptionalCUDAGuard,i8
);
impl_guard_scope!(
    CUDAStreamGuard,CUDAStream,'a;
    OptionalCUDAStreamGuard,CUDAStream,'a;
    CUDAMultiStreamGuard,[CUDAStream],'a
);

#[cfg(test)]
mod tests {
    use anyhow::ensure;

    use super::*;

    #[test]
    #[ignore]
    fn cuda_guard() -> anyhow::Result<()> {
        use std::ops::Deref;

        let new_stream = CUDAStream::get_stream_from_pool(true, -1)?;
        let current = CUDAStream::get_current_cuda_stream(-1)?;

        ensure!(current.deref() != new_stream.deref());
        {
            let _guard = CUDAStreamGuard::new(&new_stream);
            ensure!(CUDAStream::get_current_cuda_stream(-1)?.deref() == new_stream.deref());
        }
        ensure!(CUDAStream::get_current_cuda_stream(-1)?.deref() == current.deref());

        Ok(())
    }
}
