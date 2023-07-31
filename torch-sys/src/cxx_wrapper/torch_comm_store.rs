use cxx::UniquePtr;
use std::pin::Pin;

/// 定义实例化进程组时所需要的一系列kv store,可见文档:
/// [pytorch-distributed-kv-store](https://pytorch.org/docs/stable/distributed.html#distributed-key-value-store)
#[cxx::bridge]
pub mod ffi {

    #[namespace = "c10d"]
    unsafe extern "C++" {
        include!("torch_comm_process_group.h");

        /// 表示`c10d::TCPStore`,详见torch/csrc/distributed/c10d/TCPStore.hpp
        pub type TCPStore;
        /// 表示`c10d::TCPStoreOptions`,详见torch/csrc/distributed/c10d/TCPStore.hpp
        pub type TCPStoreOptions;

        /// 实例化`TCPStoreOptions`,用于后续`TCPStore`的创建
        fn new_tcp_store_options(
            port: u16,
            is_server: bool,
            num_workers: usize,
            wait_workers: bool,
            timeout: i64,
        ) -> UniquePtr<TCPStoreOptions>;

        /// 创建`TCPStore`,其中host是master节点的主机名
        fn new_tcp_store(host: String, opts: &TCPStoreOptions) -> UniquePtr<TCPStore>;

        /// 设置操作的超时时间,继承于`c10d::Store`类的需要实现的方法
        fn set_tcp_store_timeout(tcp_store: Pin<&mut TCPStore>, timeout: i64);

        /// 表示`c10d::PrefixStore`,详见torch/csrc/distributed/c10d/PrefixStore.hpp
        pub type PrefixStore;

        /// 利用`TcpStore`实例来创建拥有指定前缀的`PrefixStore`
        fn new_prefix_store_with_tcp_store(
            prefix: String,
            tcp_store: UniquePtr<TCPStore>,
        ) -> UniquePtr<PrefixStore>;

        /// 利用`PrefixStore`实例本身来创建拥有指定前缀的`PrefixStore`
        fn new_nested_prefix_store(
            prefix: String,
            prefix_store: UniquePtr<PrefixStore>,
        ) -> UniquePtr<PrefixStore>;

        /// 设置操作的超时时间,继承于`c10d::Store`类的需要实现的方法
        fn set_prefix_store_timeout(prefix_store: Pin<&mut PrefixStore>, timeout: i64);
    }
}
