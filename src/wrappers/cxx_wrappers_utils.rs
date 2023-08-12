use cxx::UniquePtr;
pub use torch_sys::wrappers::utils::CppArcClone;

/// 负责包装`torch_sys`crate中具有cpp arc语义的cxx wrapper,
/// 方便对cxx wrapper类型进行trait的实现与方法的定义
pub struct CppArc<T: CppArcClone>(pub(crate) UniquePtr<T>);

unsafe impl<T: Send + CppArcClone> Send for CppArc<T> {}
unsafe impl<T: Sync + CppArcClone> Sync for CppArc<T> {}

impl<T: CppArcClone> Clone for CppArc<T> {
    fn clone(&self) -> Self {
        Self(self.0.arc_clone())
    }
}

impl<T: CppArcClone> From<UniquePtr<T>> for CppArc<T> {
    fn from(value: UniquePtr<T>) -> Self {
        Self(value)
    }
}
