pub mod cxx_wrappers;

use cxx::UniquePtr;

pub use cxx_wrappers::*;

impl CUDAEvent {
    /// 使用默认的配置创建cuda event(默认启用cudaEventDisableTiming flag),
    pub fn new() -> UniquePtr<Self> {
        new_cuda_event()
    }
    /// 来创建启用了指定标志位的cuda event,
    /// 允许的flag可见:[cuda event文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT)
    pub fn new_with_flags(flags: usize) -> UniquePtr<Self> {
        new_cuda_event_with_flags(flags)
    }
}

#[cfg(test)]
mod torch_cuda_event {
    use super::*;

    #[test]
    fn uninit_event() {
        let uninit_cuda_event = new_cuda_event();
        let device_index = uninit_cuda_event.get_device_index();
        let is_created = uninit_cuda_event.is_created();
        assert!(!is_created);
        assert_eq!(device_index, -1);
    }
    #[test]
    fn handle_elapsed_time_error() {
        let uninit_event1 = new_cuda_event();
        let uninit_event2 = new_cuda_event();
        let result = uninit_event1.elapsed_time(uninit_event2.as_ref().unwrap());
        assert!(result.is_err());
    }
}
