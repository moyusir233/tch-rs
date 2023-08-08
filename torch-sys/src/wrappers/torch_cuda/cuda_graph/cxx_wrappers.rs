pub use ffi::*;

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
        include!("wrappers/torch_cuda/cuda_graph.h");

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
