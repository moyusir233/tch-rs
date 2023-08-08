pub mod cxx_wrappers;

use cxx::UniquePtr;
pub use cxx_wrappers::*;
use std::pin::Pin;

impl MemPoolId {
    /// 获得一个唯一的graph pool id,一般在捕获图时由CUDAGraph类自己生成,而不需要手动调用该api.
    /// 通常是为了让不同的CUDAGraph实例在graph capture时共享内存池时使用该api
    pub fn new() -> Result<Self, cxx::Exception> {
        create_graph_pool_id()
    }
}

impl CUDAGraph {
    /// 创建cuda graph
    ///
    /// # Safety
    ///
    /// cuda graph实例化时会调用`at::cuda::getCurrentCUDAStream()`获得current stream,并将其作为捕获流,
    /// 需要注意**pytorch要求捕获流必须是非默认流**,否则会产生运行时错误
    pub fn new() -> UniquePtr<Self> {
        new_cuda_graph()
    }

    /// 返回graph实例使用的内存池的id
    pub fn get_pool_id(self: Pin<&mut Self>) -> Result<MemPoolId, cxx::Exception> {
        get_pool_id(self)
    }

    /// 让CUDAGraph实例开始记录之后**current stream**上的所有cuda操作,
    /// 可以传入`MemPoolId`让CUDAGraph实例使用指定的底层内存池,或让CUDAGraph实例创建新的内存池使用
    pub fn begin_capture(
        self: Pin<&mut Self>,
        pool: Option<MemPoolId>,
    ) -> Result<(), cxx::Exception> {
        begin_capture(self, pool.unwrap_or(MemPoolId { first: 0, second: 0 }))
    }
}
#[cfg(test)]
mod torch_cuda_graph {
    use super::*;
    #[test]
    fn init_graph() {
        let mut graph = new_cuda_graph();

        let id_exception = graph.as_mut().unwrap().get_pool_id().unwrap_err();
        // 未进行capture的graph不具有id,会产生exception
        println!("{}", id_exception);
    }
}
