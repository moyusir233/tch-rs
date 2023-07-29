use cxx::UniquePtr;
use std::pin::Pin;

use crate::error::TchResult;

use super::cuda_event::{CUDAEvent, CUDAEventBuilder};
pub use torch_sys::cxx_wrapper::torch_cuda_stream::CUDAStream;

pub trait CUDAStreamExt {
    /// 令cuda stream等待cuda event上的操作执行完毕
    fn wait_event(&self, event: Pin<&mut CUDAEvent>) -> TchResult<()>;

    /// 记录流上的操作
    fn record(&self) -> TchResult<UniquePtr<CUDAEvent>>;

    /// 等待特定的流
    fn wait_stream(&self, stream: &CUDAStream) -> TchResult<()>;
}

impl CUDAStreamExt for CUDAStream {
    fn wait_event(&self, event: Pin<&mut CUDAEvent>) -> TchResult<()> {
        event.block(self)?;
        Ok(())
    }

    fn record(&self) -> TchResult<UniquePtr<CUDAEvent>> {
        let mut event = CUDAEventBuilder::default().build();
        event.as_mut().unwrap().record_stream(self)?;
        Ok(event)
    }

    fn wait_stream(&self, stream: &CUDAStream) -> TchResult<()> {
        self.wait_event(stream.record()?.as_mut().unwrap())
    }
}
