pub mod cxx_wrappers;
pub use cxx_wrappers::*;

use cxx::UniquePtr;
use std::pin::Pin;

type CxxResult<T> = Result<T, cxx::Exception>;
impl CUDAGuard {
    pub fn new(device_index: i8) -> UniquePtr<Self> {
        new_cuda_guard(device_index)
    }
    pub fn get_original_device(&self) -> i8 {
        cuda_guard_original_device(self)
    }
    pub fn get_current_device(&self) -> i8 {
        cuda_guard_current_device(self)
    }
}

impl OptionalCUDAGuard {
    pub fn new(device_index: i8) -> UniquePtr<Self> {
        new_optional_cuda_guard(device_index)
    }
    pub fn new_without_init() -> UniquePtr<Self> {
        new_uninit_optional_cuda_guard()
    }
    pub fn get_original_device(&self) -> Option<i8> {
        let device_index = optional_cuda_guard_original_device(self);
        if device_index == -1 {
            None
        } else {
            Some(device_index)
        }
    }
    pub fn get_current_device(&self) -> Option<i8> {
        let device_index = optional_cuda_guard_current_device(self);
        if device_index == -1 {
            None
        } else {
            Some(device_index)
        }
    }
}

impl CUDAStreamGuard {
    pub fn new(cuda_stream: &CUDAStream) -> UniquePtr<Self> {
        new_cuda_stream_guard(cuda_stream)
    }
    /// 和`CUDAGuard`的`set_device`作用类似,只不过会同时切换current device和current stream
    pub fn set_stream(self: Pin<&mut Self>, cuda_stream: &CUDAStream) -> CxxResult<()> {
        cuda_stream_guard_set_stream(self, cuda_stream)
    }
    pub fn get_original_stream(&self) -> UniquePtr<CUDAStream> {
        cuda_stream_guard_original_stream(self)
    }
    pub fn get_current_stream(&self) -> UniquePtr<CUDAStream> {
        cuda_stream_guard_current_stream(self)
    }
    pub fn get_original_device(&self) -> i8 {
        cuda_stream_guard_original_device(self)
    }
    pub fn get_current_device(&self) -> i8 {
        cuda_stream_guard_current_device(self)
    }
}

impl OptionalCUDAStreamGuard {
    pub fn new(cuda_stream: &CUDAStream) -> UniquePtr<Self> {
        new_optional_cuda_stream_guard(cuda_stream)
    }
    pub fn new_without_init() -> UniquePtr<Self> {
        new_uninit_optional_cuda_stream_guard()
    }
    pub fn set_stream(self: Pin<&mut Self>, cuda_stream: &CUDAStream) -> CxxResult<()> {
        optional_cuda_stream_guard_set_stream(self, cuda_stream)
    }
    pub fn get_original_stream(&self) -> Option<UniquePtr<CUDAStream>> {
        let ptr = optional_cuda_stream_guard_original_stream(self);
        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }
    pub fn get_current_stream(&self) -> Option<UniquePtr<CUDAStream>> {
        let ptr = optional_cuda_stream_guard_current_stream(self);
        if ptr.is_null() {
            None
        } else {
            Some(ptr)
        }
    }
}

impl CUDAMultiStreamGuard {
    // TODO 改善接口
    pub fn new(streams: &[CUDAStream]) -> UniquePtr<Self> {
        unsafe { new_cuda_multi_stream_guard(streams.as_ptr(), streams.len()) }
    }
}

#[cfg(test)]
mod torch_cuda_guard {
    use super::*;

    #[test]
    fn init() {
        let device = -1;
        let _cuda_guard = CUDAGuard::new(device);
        let _optional_cuda_guard = OptionalCUDAGuard::new(device);
        let cuda_stream =
            CUDAStream::get_default_cuda_stream(device).expect("failed to create cuda stream");
        let _cuda_stream_gurad = CUDAStreamGuard::new(&cuda_stream);
        let _optional_cuda_stream_gurad = OptionalCUDAStreamGuard::new(&cuda_stream);
    }
}
