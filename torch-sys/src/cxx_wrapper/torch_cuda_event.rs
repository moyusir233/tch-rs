use cxx::UniquePtr;

#[allow(unused)]
use std::pin::Pin;

pub use ffi::CUDAEvent;

/// 利用cxx定义的pytorch底层使用的cuda event
#[cxx::bridge]
pub mod ffi {

    #[namespace = "at::cuda"]
    unsafe extern "C++" {
        include!("torch_cuda_event.h");

        /// 代表着`at::cuda::CUDAEvent`,详细可见ATen/cuda/CUDAEvent.h
        pub type CUDAEvent;
        /// 代表着`c10::cuda::CUDAStream`,详细可见c10/cuda/CUDAStream.h
        #[namespace = "c10::cuda"]
        type CUDAStream = crate::cxx_wrapper::torch_cuda_stream::ffi::CUDAStream;

        /// 使用默认的配置创建cuda event(默认启用cudaEventDisableTiming flag),
        fn new_cuda_event() -> UniquePtr<CUDAEvent>;

        /// 来创建启用了指定标志位的cuda event,
        /// 允许的flag可见:[cuda event文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT)
        fn new_cuda_event_with_flags(flags: usize) -> UniquePtr<CUDAEvent>;

        /// 指定该event是否已经创建(event是懒创建的)
        #[rust_name = "is_created"]
        fn isCreated(self: &CUDAEvent) -> bool;

        /// 返回与该event关联设备的index
        #[rust_name = "get_device_index"]
        fn device_index(self: &CUDAEvent) -> i8;

        /// 查询事件是否执行完毕,底层执行了`cudaEventQuery`,因此会根据创建event时的flag有不同的行为,
        /// 详见文档:[cudaEventQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7)
        fn query(self: &CUDAEvent) -> Result<bool>;

        /// 记录当前流上的cuda操作
        #[rust_name = "record_current_stream"]
        fn record(self: Pin<&mut CUDAEvent>) -> Result<()>;

        /// 记录指定流上的cuda操作,如果event已经记录过相关操作,则不会再记录
        #[rust_name = "record_once"]
        fn recordOnce(self: Pin<&mut CUDAEvent>, stream: &CUDAStream) -> Result<()>;

        /// 记录指定流上的cuda操作
        #[rust_name = "record_stream"]
        fn record(self: Pin<&mut CUDAEvent>, stream: &CUDAStream) -> Result<()>;

        /// 令指定的cuda stream等待当前cuda event上所记录的cuda操作执行完毕
        fn block(self: Pin<&mut CUDAEvent>, stream: &CUDAStream) -> Result<()>;

        /// 返回两个event之间记录的时间差
        fn elapsed_time(self: &CUDAEvent, other: &CUDAEvent) -> Result<f32>;

        /// 等待一个事件执行完毕
        fn synchronize(self: &CUDAEvent) -> Result<()>;
    }
}

impl ffi::CUDAEvent {
    /// 使用默认的配置创建cuda event(默认启用cudaEventDisableTiming flag),
    pub fn new() -> UniquePtr<Self> {
        ffi::new_cuda_event()
    }
    /// 来创建启用了指定标志位的cuda event,
    /// 允许的flag可见:[cuda event文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT)
    pub fn new_with_flags(flags: usize) -> UniquePtr<Self> {
        ffi::new_cuda_event_with_flags(flags)
    }
}
#[cfg(test)]
mod cuda_event_tests {
    use super::*;
    #[test]
    fn uninit_event() {
        let uninit_cuda_event = ffi::new_cuda_event();
        let device_index = uninit_cuda_event.get_device_index();
        let is_created = uninit_cuda_event.is_created();
        assert!(!is_created);
        assert_eq!(device_index, -1);
    }
    #[test]
    fn handle_elapsed_time_error() {
        let uninit_event1 = ffi::new_cuda_event();
        let uninit_event2 = ffi::new_cuda_event();
        let result = uninit_event1.elapsed_time(uninit_event2.as_ref().unwrap());
        assert!(result.is_err());
    }
}
