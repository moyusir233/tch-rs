use cxx::UniquePtr;
use std::ffi::CString;
use std::pin::Pin;
use std::time::Duration;

pub use ffi::{MyTCPStoreOptions, PrefixStore, TCPStore};

/// 定义实例化进程组时所需要的一系列kv store,可见文档:
/// [pytorch-distributed-kv-store](https://pytorch.org/docs/stable/distributed.html#distributed-key-value-store)
#[cxx::bridge]
pub mod ffi {

    /// 表示可以转换为`c10d::TCPStoreOptions`的shared type,
    /// 详见torch/csrc/distributed/c10d/TCPStore.hpp,
    /// 其中timout单位为ms
    #[namespace = "c10d"]
    #[derive(Clone)]
    pub struct MyTCPStoreOptions {
        port: u16,
        is_server: bool,
        num_workers: usize,
        wait_workers: bool,
        timeout: i64,
        multi_tenant: bool,
    }

    #[namespace = "c10d"]
    unsafe extern "C++" {
        include!("torch_comm_process_group.h");

        /// 表示`c10d::TCPStore`,详见torch/csrc/distributed/c10d/TCPStore.hpp
        pub type TCPStore;

        /// 创建`TCPStore`,其中host是master节点的主机名
        unsafe fn new_tcp_store(
            host: *const c_char,
            opts: &MyTCPStoreOptions,
        ) -> UniquePtr<TCPStore>;

        /// 设置操作的超时时间,继承于`c10d::Store`类的需要实现的方法
        fn set_tcp_store_timeout(tcp_store: Pin<&mut TCPStore>, timeout: i64);

        /// 表示`c10d::PrefixStore`,详见torch/csrc/distributed/c10d/PrefixStore.hpp
        pub type PrefixStore;

        /// 利用`TcpStore`实例来创建拥有指定前缀的`PrefixStore`
        unsafe fn new_prefix_store_with_tcp_store(
            prefix: *const c_char,
            tcp_store: UniquePtr<TCPStore>,
        ) -> UniquePtr<PrefixStore>;

        /// 利用`PrefixStore`实例本身来创建拥有指定前缀的`PrefixStore`
        unsafe fn new_nested_prefix_store(
            prefix: *const c_char,
            prefix_store: UniquePtr<PrefixStore>,
        ) -> UniquePtr<PrefixStore>;

        /// 设置操作的超时时间,继承于`c10d::Store`类的需要实现的方法
        fn set_prefix_store_timeout(prefix_store: Pin<&mut PrefixStore>, timeout: i64);
    }
}

#[inline]
fn cstring_from_string(s: String) -> CString {
    CString::new(s).unwrap()
}

pub trait Store {
    fn set_timeout(self: Pin<&mut Self>, timeout: Duration);
}

pub trait NestPrefixStore<S: Store + cxx::memory::UniquePtrTarget> {
    fn nest_store(prefix: String, store: UniquePtr<S>) -> UniquePtr<PrefixStore>;
}

impl Default for MyTCPStoreOptions {
    fn default() -> Self {
        // 参考torch/csrc/distributed/c10d/TCPStore.hpp所定义的默认值
        Self {
            port: 29500,
            is_server: false,
            num_workers: 0,
            wait_workers: true,
            timeout: 300000,
            multi_tenant: false,
        }
    }
}

impl TCPStore {
    pub fn new(host: String, opts: &MyTCPStoreOptions) -> UniquePtr<Self> {
        unsafe { ffi::new_tcp_store(cstring_from_string(host).as_ptr(), opts) }
    }
}

impl Store for TCPStore {
    fn set_timeout(self: Pin<&mut Self>, timeout: Duration) {
        ffi::set_tcp_store_timeout(self, timeout.as_millis() as i64)
    }
}

impl Store for PrefixStore {
    fn set_timeout(self: Pin<&mut Self>, timeout: Duration) {
        ffi::set_prefix_store_timeout(self, timeout.as_millis() as i64);
    }
}

impl NestPrefixStore<PrefixStore> for PrefixStore {
    fn nest_store(prefix: String, store: UniquePtr<PrefixStore>) -> UniquePtr<PrefixStore> {
        unsafe { ffi::new_nested_prefix_store(cstring_from_string(prefix).as_ptr(), store) }
    }
}

impl NestPrefixStore<TCPStore> for PrefixStore {
    fn nest_store(prefix: String, store: UniquePtr<TCPStore>) -> UniquePtr<PrefixStore> {
        unsafe { ffi::new_prefix_store_with_tcp_store(cstring_from_string(prefix).as_ptr(), store) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn store() {
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let host = String::from("localhost");

        std::thread::spawn({
            let barrier = barrier.clone();
            let host = host.clone();
            move || {
                let tcp_store_opts = MyTCPStoreOptions {
                    port: 8080,
                    is_server: false,
                    num_workers: 2,
                    wait_workers: true,
                    timeout: 10000,
                    multi_tenant: false,
                };
                let _tcp_store = TCPStore::new(host, &tcp_store_opts);
                barrier.wait();
            }
        });

        let tcp_store_opts = MyTCPStoreOptions {
            port: 8080,
            is_server: true,
            num_workers: 2,
            wait_workers: true,
            timeout: 10000,
            multi_tenant: true,
        };
        let mut tcp_store = TCPStore::new(host, &tcp_store_opts);
        barrier.wait();

        tcp_store.pin_mut().set_timeout(Duration::from_secs(100));

        let prefix_store = PrefixStore::nest_store("prefix1".into(), tcp_store);
        let mut prefix_store = PrefixStore::nest_store("prefix2".into(), prefix_store);
        prefix_store.pin_mut().set_timeout(Duration::from_secs(100));
    }
}
