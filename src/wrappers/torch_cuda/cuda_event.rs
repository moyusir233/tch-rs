use std::pin::Pin;

use derive_builder::Builder;
pub use torch_sys::wrappers::torch_cuda::cuda_event::CUDAEvent;
use torch_sys::wrappers::torch_cuda::cuda_stream::CUDAStream;

use crate::error::TchResult;

/// 配置cuda event创建时的标志位,各个标志位的具体功能见:[cuda event docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html),默认创建的是只启动计时功能的CUDAEvent
#[derive(Builder)]
#[builder(name = "CUDAEventBuilder")]
#[builder(build_fn(skip, name = "build"))]
#[allow(dead_code)]
pub struct CUDAEventFlagsConfig {
    disable_timing: bool,
    blocking_sync: bool,
    interprocess: bool,
}

impl CUDAEventBuilder {
    pub fn build(&mut self) -> cxx::UniquePtr<CUDAEvent> {
        let mut flags = 0;
        let blocking = self.blocking_sync.unwrap_or(false);
        let disable_timing = self.disable_timing.unwrap_or(false);
        let interprocess = self.interprocess.unwrap_or(false);

        for i in [blocking, disable_timing, interprocess]
            .into_iter()
            .enumerate()
            .filter_map(|(i, enable_flag)| if enable_flag { Some(1 << i) } else { None })
        {
            flags |= i;
        }
        CUDAEvent::new_with_flags(flags)
    }
}

pub trait CUDAEventExt {
    /// 令current stream等待当前cuda event上记录的事件执行完毕
    fn block_current_stream(self: Pin<&mut Self>) -> TchResult<()>;
}

impl CUDAEventExt for CUDAEvent {
    fn block_current_stream(self: Pin<&mut Self>) -> TchResult<()> {
        let current_stream = CUDAStream::get_current_cuda_stream(-1).unwrap();
        self.block(&current_stream)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// 通过测量GPU上的执行时间与CPU的睡眠时间来比较,判断CUDAEvent的计时是否正确
    #[test]
    #[ignore]
    fn cuda_event_timing() -> anyhow::Result<()> {
        let device = crate::Device::Cuda(0);
        let epoch = 10;
        let time_per_epoch = 100;

        let _device_guard = crate::torch_cuda::cuda_guards::CUDAGuard::new(device.c_int() as i8);

        let mut start_event = CUDAEventBuilder::default().build();
        start_event.pin_mut().record_current_stream()?;

        let mut a = crate::Tensor::arange(9, (crate::Kind::Float, device)).view([3, 3]);
        for _ in 0..epoch {
            a = a.matmul(&a);
        }

        let mut end_event = CUDAEventBuilder::default().build();
        end_event.pin_mut().record_current_stream()?;

        std::thread::sleep(Duration::from_millis(epoch * time_per_epoch));

        // 此时start_event与end_event已经在GPU上异步地完成了计算
        anyhow::ensure!(start_event.query()?);
        anyhow::ensure!(end_event.query()?);

        let spend_time = start_event.elapsed_time(&end_event)?;
        anyhow::ensure!(spend_time > 0.0 && spend_time < (epoch * time_per_epoch) as f32);

        Ok(())
    }
}
