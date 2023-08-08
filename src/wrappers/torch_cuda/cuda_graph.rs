use cxx::UniquePtr;
use std::pin::Pin;
pub use torch_sys::wrappers::torch_cuda::cuda_graph::{CUDAGraph, MemPoolId};
use torch_sys::wrappers::torch_cuda::cuda_guard::CUDAStreamGuard;
use torch_sys::wrappers::torch_cuda::cuda_stream::CUDAStream;

use crate::error::TchResult;

pub trait CUDAGraphExt {
    /// 使用可选的捕获流(默认为current stream)与内存池(默认新建内存池)完成图记录操作
    fn record<T>(
        self: Pin<&mut Self>,
        capture_stream: Option<&CUDAStream>,
        pool: Option<MemPoolId>,
        action: T,
    ) -> TchResult<()>
    where
        T: FnOnce() -> TchResult<()>;
}

pub struct RecordGuard<'a> {
    cuda_graph: Pin<&'a mut CUDAGraph>,
    #[allow(dead_code)]
    stream_guard: Option<UniquePtr<CUDAStreamGuard>>,
}
impl<'a> RecordGuard<'a> {
    /// 创建一个用于进行流捕获相关的前置工作与收尾工作的RAII Guard,
    ///
    /// # Warning
    ///
    /// 注意[`RecordGuard`]创建时,如果传入非`None`的`capture_stream`,
    /// 则也会进行cuda stream的切换,相当于还创建了一个[`CUDAStreamGuard`],
    pub fn new(
        mut cuda_graph: Pin<&'a mut CUDAGraph>,
        capture_stream: Option<&CUDAStream>,
        pool: Option<MemPoolId>,
    ) -> Result<Self, crate::TchError> {
        use torch_sys::wrappers::torch_utils::empty_cache;

        // 捕获前的准备工作,参考了torch/cuda/graphs.py
        crate::Cuda::synchronize(-1);
        // 释放内存,确保图捕获时有足够的内存使用
        empty_cache()?;
        // 切换stream的guard
        let stream_guard = capture_stream.map(CUDAStreamGuard::new);
        // 等待捕获流上所在设备的计算完毕,避免捕获到其他无效的操作
        crate::Cuda::synchronize(-1);

        // 开始捕获
        cuda_graph.as_mut().begin_capture(pool)?;
        Ok(Self { cuda_graph, stream_guard })
    }
}

impl Drop for RecordGuard<'_> {
    fn drop(&mut self) {
        // 结束捕获
        self.cuda_graph.as_mut().end_capture().unwrap();
    }
}

impl CUDAGraphExt for CUDAGraph {
    fn record<T>(
        self: Pin<&mut Self>,
        capture_stream: Option<&CUDAStream>,
        pool: Option<MemPoolId>,
        action: T,
    ) -> TchResult<()>
    where
        T: FnOnce() -> TchResult<()>,
    {
        let _guard = RecordGuard::new(self, capture_stream, pool);

        // 执行需要被捕获的cuda操作
        action()?;

        Ok(())
    }
}
impl<G: CUDAGraphExt + cxx::memory::UniquePtrTarget> CUDAGraphExt for UniquePtr<G> {
    fn record<T>(
        self: Pin<&mut Self>,
        capture_stream: Option<&CUDAStream>,
        pool: Option<MemPoolId>,
        action: T,
    ) -> TchResult<()>
    where
        T: FnOnce() -> TchResult<()>,
    {
        let ptr: &mut UniquePtr<G> = self.get_mut();
        ptr.as_mut().unwrap().record(capture_stream, pool, action)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::ensure;

    use super::*;

    #[test]
    fn cuda_graph() -> anyhow::Result<()> {
        let device = crate::Device::Cuda(0);
        let mut pin_tensor = crate::Tensor::zeros([3, 3], (crate::Kind::Int, device));

        let mut graph = CUDAGraph::new();
        use std::ops::Deref;
        graph.pin_mut().record(
            Some(CUDAStream::get_stream_from_pool(true, device.c_int() as i8)?.deref()),
            None,
            || {
                let _ = pin_tensor.f_add_scalar_(1)?;
                Ok(())
            },
        )?;

        let target = 5;
        for _ in 0..target {
            graph.pin_mut().replay()?;
        }
        crate::Cuda::synchronize(device.c_int() as i64);

        let target = pin_tensor.ones_like() * target;
        ensure!(pin_tensor.equal(&target));

        Ok(())
    }
}
