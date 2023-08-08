pub use ffi::*;

/// 一系列libtorch中常用的与cuda相关的RAII guard类
#[cxx::bridge]
pub mod ffi {

    #[namespace = "c10::cuda"]
    unsafe extern "C++" {
        include!("wrappers/torch_cuda/cuda_guard.h");

        /// 代表着`c10::cuda::CUDAGuard`,详细可见c10/cuda/CUDAGuard.h,
        /// `CUDAGuard`的实例会**确保其被析构时将current device设置为创建实例时的current device**
        pub type CUDAGuard;

        /// 创建`CUDAGuard`并将current device设置为传入的device
        fn new_cuda_guard(device_index: i8) -> UniquePtr<CUDAGuard>;

        /// 设置current device为传入的设备
        #[rust_name = "set_device"]
        fn set_index(self: Pin<&mut CUDAGuard>, device_index: i8) -> Result<()>;

        #[rust_name = "cuda_guard_original_device"]
        fn original_device(guard: &CUDAGuard) -> i8;

        #[rust_name = "cuda_guard_current_device"]
        fn current_device(guard: &CUDAGuard) -> i8;

        /// 代表着`c10::cuda::OptionalCUDAGuard`,详细可见c10/cuda/CUDAGuard.h,
        /// 与`CUDAGuard`不同的是,`OptionalCUDAGuard`允许实例化时不进行设备的设置,
        /// 并且其**实例会在析构时将current device设置为其第一次为实例设置设备时的current device**,
        /// 比如下面的例子中:
        /// ```c++
        /// setDevice(1);
        /// OptionalDeviceGuard g;
        /// setDevice(2);
        /// g.reset_device(Device(DeviceType::CUDA, 3));  // initializes!
        /// ```
        /// g实例析构时会将current device设置为2号设备
        pub type OptionalCUDAGuard;

        fn new_uninit_optional_cuda_guard() -> UniquePtr<OptionalCUDAGuard>;
        fn new_optional_cuda_guard(device_index: i8) -> UniquePtr<OptionalCUDAGuard>;

        #[rust_name = "set_device"]
        fn set_index(self: Pin<&mut OptionalCUDAGuard>, deviec_index: i8) -> Result<()>;

        #[rust_name = "optional_cuda_guard_original_device"]
        fn original_device(guard: &OptionalCUDAGuard) -> i8;

        #[rust_name = "optional_cuda_guard_current_device"]
        fn current_device(guard: &OptionalCUDAGuard) -> i8;

        /// 主动将current device设置为original device,并且将实例恢复为未设置设备的状态
        fn reset(self: Pin<&mut OptionalCUDAGuard>) -> Result<()>;

        /// 代表着`c10::cuda::CUDAStreamGuard`,详细可见c10/cuda/CUDAGuard.h,
        /// `CUDAStreamGuard`的作用与`CUDAGuard`基本类似,
        /// 不过切换current stream时也会将current device设置为传入stream所在的device
        pub type CUDAStreamGuard;
        #[namespace = "c10::cuda"]
        type CUDAStream = crate::wrappers::torch_cuda::cuda_stream::CUDAStream;

        fn new_cuda_stream_guard(stream: &CUDAStream) -> UniquePtr<CUDAStreamGuard>;

        /// 和`CUDAGuard`的`set_device`作用类似,只不过会同时切换current device和current stream
        #[rust_name = "cuda_stream_guard_set_stream"]
        fn reset_stream(guard: Pin<&mut CUDAStreamGuard>, stream: &CUDAStream) -> Result<()>;

        #[rust_name = "cuda_stream_guard_original_stream"]
        fn original_stream(guard: &CUDAStreamGuard) -> UniquePtr<CUDAStream>;

        #[rust_name = "cuda_stream_guard_current_stream"]
        fn current_stream(guard: &CUDAStreamGuard) -> UniquePtr<CUDAStream>;

        #[rust_name = "cuda_stream_guard_original_device"]
        fn original_device(guard: &CUDAStreamGuard) -> i8;

        #[rust_name = "cuda_stream_guard_current_device"]
        fn current_device(guard: &CUDAStreamGuard) -> i8;

        /// 代表着`c10::cuda::OptionalCUDAStreamGuard`,详细可见c10/cuda/CUDAGuard.h
        /// `OptionalCUDAStreamGuard`的作用与`OptionalCUDAGuard`基本类似,
        pub type OptionalCUDAStreamGuard;

        fn new_uninit_optional_cuda_stream_guard() -> UniquePtr<OptionalCUDAStreamGuard>;
        fn new_optional_cuda_stream_guard(
            stream: &CUDAStream,
        ) -> UniquePtr<OptionalCUDAStreamGuard>;

        #[rust_name = "optional_cuda_stream_guard_set_stream"]
        fn reset_stream(
            guard: Pin<&mut OptionalCUDAStreamGuard>,
            stream: &CUDAStream,
        ) -> Result<()>;

        #[rust_name = "optional_cuda_stream_guard_original_stream"]
        fn original_stream(guard: &OptionalCUDAStreamGuard) -> UniquePtr<CUDAStream>;

        #[rust_name = "optional_cuda_stream_guard_current_stream"]
        fn current_stream(guard: &OptionalCUDAStreamGuard) -> UniquePtr<CUDAStream>;

        /// 与`OptionalCUDAGuard::reset(&mut self)`的作用类似
        fn reset(self: Pin<&mut OptionalCUDAStreamGuard>) -> Result<()>;

        /// 代表着`c10::cuda::CUDAMultiStreamGuard`,详细可见c10/cuda/CUDAGuard.h,
        /// 创建实例时其接收一组stream,并分别将每个stream关联的设备上的current stream设置为该stream,
        /// 并在析构时恢复这些设备上原来的current stream
        pub type CUDAMultiStreamGuard;

        /// # Safety
        ///
        /// 为了方便，这里直接传递了streams的数组指针，保证指针有效即可
        unsafe fn new_cuda_multi_stream_guard(
            streams: *const CUDAStream,
            length: usize,
        ) -> UniquePtr<CUDAMultiStreamGuard>;
    }
}
