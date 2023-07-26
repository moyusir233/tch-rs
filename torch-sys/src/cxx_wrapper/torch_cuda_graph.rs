use cxx::UniquePtr;
use std::pin::Pin;

pub use ffi::{CUDAGraph, MemPoolId};
unsafe impl Send for ffi::CUDAGraph {}

/// 利用cxx定义的pytorch底层使用的cuda graph
#[cxx::bridge]
pub mod ffi {

    /// 内存池id,作为`CUDAGraph::capture_begin`函数的参数,用于指定cuda graph replay时使用的底层内存池,
    #[derive(Copy, Clone, Hash, Default, Debug)]
    pub struct MemPoolId {
        first: u64,
        second: u64,
    }

    #[namespace = "at::cuda"]
    unsafe extern "C++" {
        include!("torch_cuda_graph.h");

        /// 代表着`at::cuda::CUDAGraph`,详细可见ATen/cuda/CUDAGraph.h
        pub type CUDAGraph;

        /// 创建cuda graph
        ///
        /// # Safety
        ///
        /// cuda graph实例化时会调用`at::cuda::getCurrentCUDAStream()`获得current stream,并将其作为捕获流,
        /// 需要注意**pytorch要求捕获流必须是非默认流**,否则会产生运行时错误
        fn new_cuda_graph() -> UniquePtr<CUDAGraph>;

        /// 获得一个唯一的graph pool id,一般在捕获图时由CUDAGraph类自己生成,而不需要手动调用该api.
        /// 通常是为了让不同的CUDAGraph实例在graph capture时共享内存池时使用该api
        fn create_graph_pool_id() -> Result<MemPoolId>;

        /// 让CUDAGraph实例开始记录之后**current stream**上的所有cuda操作
        fn begin_capture(cuda_graph: Pin<&mut CUDAGraph>, pool: MemPoolId) -> Result<()>;

        /// 通知CUDAGraph实例结束流上操作的记录
        #[rust_name = "end_capture"]
        fn capture_end(self: Pin<&mut CUDAGraph>) -> Result<()>;

        /// 在**current stream**上执行graph记录到的所有操作
        fn replay(self: Pin<&mut CUDAGraph>) -> Result<()>;

        /// 清理graph占用的所有资源,默认CUDAGraph的析构方法会调用reset,不需要手动的调用
        fn reset(self: Pin<&mut CUDAGraph>) -> Result<()>;

        /// 返回graph实例使用的内存池的id
        fn get_pool_id(cuda_graph: Pin<&mut CUDAGraph>) -> Result<MemPoolId>;
    }
}

impl ffi::MemPoolId {
    /// 获得一个唯一的graph pool id,一般在捕获图时由CUDAGraph类自己生成,而不需要手动调用该api.
    /// 通常是为了让不同的CUDAGraph实例在graph capture时共享内存池时使用该api
    pub fn new() -> Result<Self, cxx::Exception> {
        ffi::create_graph_pool_id()
    }
}

impl ffi::CUDAGraph {
    /// 创建cuda graph
    ///
    /// # Safety
    ///
    /// cuda graph实例化时会调用`at::cuda::getCurrentCUDAStream()`获得current stream,并将其作为捕获流,
    /// 需要注意**pytorch要求捕获流必须是非默认流**,否则会产生运行时错误
    pub fn new() -> UniquePtr<Self> {
        ffi::new_cuda_graph()
    }

    /// 返回graph实例使用的内存池的id
    pub fn get_pool_id(self: Pin<&mut Self>) -> Result<ffi::MemPoolId, cxx::Exception> {
        ffi::get_pool_id(self)
    }

    /// 让CUDAGraph实例开始记录之后**current stream**上的所有cuda操作,
    /// 可以传入`MemPoolId`让CUDAGraph实例使用指定的底层内存池,或让CUDAGraph实例创建新的内存池使用
    pub fn begin_capture(
        self: Pin<&mut Self>,
        pool: Option<ffi::MemPoolId>,
    ) -> Result<(), cxx::Exception> {
        ffi::begin_capture(self, pool.unwrap_or(ffi::MemPoolId { first: 0, second: 0 }))
    }
}
#[cfg(test)]
mod cuda_graph_test {
    use super::*;
    #[test]
    fn init_graph() {
        let mut graph = ffi::new_cuda_graph();

        let id_exception = graph.as_mut().unwrap().get_pool_id().unwrap_err();
        // 未进行capture的graph不具有id,会产生exception
        println!("{}", id_exception);
    }
}
