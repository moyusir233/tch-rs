pub mod cxx_wrappers;

use cxx::UniquePtr;
pub use cxx_wrappers::*;

#[allow(clippy::partialeq_ne_impl)]
impl PartialEq for CUDAStream {
    fn eq(&self, other: &Self) -> bool {
        equal(self, other)
    }

    fn ne(&self, other: &Self) -> bool {
        not_equal(self, other)
    }
}

type CxxResult<T> = Result<T, cxx::Exception>;
impl CUDAStream {
    /// 查询stream的优先级,详见[cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6)
    pub fn get_priority(&self) -> Result<i64, cxx::Exception> {
        priority(self)
    }

    /// 从stream pool中获得stream,关于pytorch底层维护的stream pool,可见c10/torch_cuda/CUDAStream.h
    pub fn get_stream_from_pool(
        is_high_priority: bool,
        device_index: i8,
    ) -> CxxResult<UniquePtr<Self>> {
        get_stream_from_pool(is_high_priority, device_index)
    }

    /// 返回指定设备上的default stream,也是用于大多数计算的stream
    pub fn get_default_cuda_stream(device_index: i8) -> CxxResult<UniquePtr<Self>> {
        get_default_cuda_stream(device_index)
    }

    /// 返回指定设备上的current stream,大多数情况下current stream即设备的default stream
    pub fn get_current_cuda_stream(device_index: i8) -> CxxResult<UniquePtr<Self>> {
        get_current_cuda_stream(device_index)
    }
}

#[cfg(test)]
mod torch_cuda_stream {
    use super::*;
    #[test]
    fn init_stream() {
        // 测试从池中获得stream的相关函数
        let stream1 = ffi::get_default_cuda_stream(0).unwrap();
        let stream2 = ffi::get_default_cuda_stream(0).unwrap();
        // 由于都是default stream,两者相同
        assert!(stream1.as_ref().unwrap() == stream2.as_ref().unwrap());

        let stream3 = ffi::get_current_cuda_stream(0).unwrap();
        // 默认情况下,current stream即default stream
        assert!(stream1.as_ref().unwrap() == stream3.as_ref().unwrap());

        let stream4 = ffi::get_stream_from_pool(true, 0).unwrap();
        // 从high_priority pool中获得的stream与default stream不同
        assert!(stream1.as_ref().unwrap() != stream4.as_ref().unwrap());
    }
}
