use autocxx::prelude::*;
use cxx::memory::UniquePtrTarget;

/// 表示具有`std::sync::Arc`语义的cxx wrapper类型
pub trait CppArcClone: UniquePtrTarget + Sized + Send {
    /// 类似rust中`Arc::clone`的语义,不过是通过c++一侧相关的智能指针实现的
    fn arc_clone(&self) -> UniquePtr<Self>;
}
