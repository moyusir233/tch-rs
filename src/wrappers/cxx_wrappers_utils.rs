use autocxx::cxx::UniquePtr;
pub use torch_sys::wrappers::utils::CppArcClone;

/// 负责包装`torch_sys`crate中具有cpp arc语义的cxx wrapper,
/// 方便对cxx wrapper类型进行trait的实现与方法的定义
pub(crate) struct CppArc<T: CppArcClone>(pub(crate) UniquePtr<T>);

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

impl<T: CppArcClone> std::ops::Deref for CppArc<T> {
    type Target = UniquePtr<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: CppArcClone> std::ops::DerefMut for CppArc<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
