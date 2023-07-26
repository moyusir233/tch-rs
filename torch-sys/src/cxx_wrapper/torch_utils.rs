pub use ffi::*;
/// torch相关的utils函数
#[cxx::bridge]
pub mod ffi {

    unsafe extern "C++" {
        include!("torch_utils.h");

        /// 释放pytorch底层维护缓存池的缓存
        fn empty_cache() -> Result<()>;

        /// 获得当前使用的device
        fn current_device() -> Result<i8>;
    }
}
