#[allow(unused)]
use crate::cxx_wrapper::torch_cuda_event::CUDAEvent;
use cxx::UniquePtr;
pub use ffi::CUDAStream;
/// 利用cxx定义的pytorch底层使用的cuda stream
#[cxx::bridge]
pub mod ffi {

    #[namespace = "c10::cuda"]
    unsafe extern "C++" {
        include!("torch_cuda_stream.h");

        /// 代表着`c10::cuda::CUDAStream`,详细可见c10/cuda/CUDAStream.h
        pub type CUDAStream;

        /// 用于比较stream是否相同的函数,实际上比较的是底层的cuda stream指针
        fn equal(cuda_stream: &CUDAStream, other: &CUDAStream) -> bool;
        fn not_equal(cuda_stream: &CUDAStream, other: &CUDAStream) -> bool;

        /// 获得与该stream关联的设备的下标
        #[rust_name = "get_device_index"]
        fn device_index(self: &CUDAStream) -> i8;

        /// 获得stream的id
        #[rust_name = "get_id"]
        fn id(self: &CUDAStream) -> i64;

        /// 查询stream操作是否执行完毕,详见[cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435)
        fn query(self: &CUDAStream) -> Result<bool>;

        /// 等待stream上的操作执行完毕,详见[cudaStreamSynchronize](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g82b5784f674c17c6df64affe618bf45e)
        fn synchronize(self: &CUDAStream) -> Result<()>;

        /// 查询stream的优先级,详见[cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6)
        #[rust_name = "priority"]
        fn _priority(cuda_stream: &CUDAStream) -> Result<i64>;

        /// 从stream pool中获得stream,关于pytorch底层维护的stream pool,可见c10/cuda/CUDAStream.h
        #[rust_name = "get_stream_from_pool"]
        fn _getStreamFromPool(
            is_high_priority: bool,
            device_index: i8,
        ) -> Result<UniquePtr<CUDAStream>>;

        // /// 从非pytorch维护的stream pool中获得stream
        // #[rust_name = "get_stream_from_external"]
        // fn _getStreamFromExternal(
        //     ext_stream: cudaStream_t,
        //     device_index: i8,
        // ) -> Result<UniquePtr<CUDAStream>>;

        /// 返回指定设备上的default stream,也是用于大多数计算的stream
        #[rust_name = "get_default_cuda_stream"]
        fn _getDefaultCUDAStream(device_index: i8) -> Result<UniquePtr<CUDAStream>>;

        /// 返回指定设备上的current stream,大多数情况下current stream即设备的default stream
        #[rust_name = "get_current_cuda_stream"]
        fn _getCurrentCUDAStream(device_index: i8) -> Result<UniquePtr<CUDAStream>>;
    }
}

#[allow(clippy::partialeq_ne_impl)]
impl PartialEq for ffi::CUDAStream {
    fn eq(&self, other: &Self) -> bool {
        ffi::equal(self, other)
    }

    fn ne(&self, other: &Self) -> bool {
        ffi::not_equal(self, other)
    }
}

type CxxResult<T> = Result<T, cxx::Exception>;
impl ffi::CUDAStream {
    /// 查询stream的优先级,详见[cudaStreamGetPriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g192bb727d15c4407c119747de7d198a6)
    pub fn get_priority(&self) -> Result<i64, cxx::Exception> {
        ffi::priority(self)
    }

    /// 从stream pool中获得stream,关于pytorch底层维护的stream pool,可见c10/cuda/CUDAStream.h
    pub fn get_stream_from_pool(
        is_high_priority: bool,
        device_index: i8,
    ) -> CxxResult<UniquePtr<Self>> {
        ffi::get_stream_from_pool(is_high_priority, device_index)
    }

    /// 返回指定设备上的default stream,也是用于大多数计算的stream
    pub fn get_default_cuda_stream(device_index: i8) -> CxxResult<UniquePtr<Self>> {
        ffi::get_default_cuda_stream(device_index)
    }

    /// 返回指定设备上的current stream,大多数情况下current stream即设备的default stream
    pub fn get_current_cuda_stream(device_index: i8) -> CxxResult<UniquePtr<Self>> {
        ffi::get_current_cuda_stream(device_index)
    }
}

#[cfg(test)]
mod cuda_stream_tests {
    use super::*;
    #[test]
    fn init_stream() {
        // 测试从池中获得stream的相关函数
        let stream1 = ffi::get_default_cuda_stream(1).unwrap();
        let stream2 = ffi::get_default_cuda_stream(1).unwrap();
        // 由于都是default stream,两者相同
        assert!(stream1.as_ref().unwrap() == stream2.as_ref().unwrap());

        let stream3 = ffi::get_current_cuda_stream(1).unwrap();
        // 默认情况下,current stream即default stream
        assert!(stream1.as_ref().unwrap() == stream3.as_ref().unwrap());

        let stream4 = ffi::get_stream_from_pool(true, 1).unwrap();
        // 从high_priority pool中获得的stream与default stream不同
        assert!(stream1.as_ref().unwrap() != stream4.as_ref().unwrap());
    }
}
