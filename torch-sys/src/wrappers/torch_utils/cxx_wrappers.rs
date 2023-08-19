pub use ffi::*;
/// torch相关的utils函数
#[cxx::bridge]
pub mod ffi {

    unsafe extern "C++" {
        include!("wrappers/torch_utils.h");

        /// 进行libtorch所需的一些全局初始化工作,比如初始化日志等,
        /// 参考torch/csrc/Module.cpp
        fn init_torch_module() -> Result<()>;

        /// 释放pytorch底层维护缓存池的缓存
        fn empty_cache() -> Result<()>;

        /// 获得当前使用的device
        fn current_device() -> Result<i8>;
    }
}
