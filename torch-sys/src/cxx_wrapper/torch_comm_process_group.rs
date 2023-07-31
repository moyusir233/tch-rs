use cxx::UniquePtr;
use std::pin::Pin;

/// 定义支持多进程之间通信的不同通信进程组,可见文档:
/// [pytorch-distributed-process_group](https://pytorch.org/docs/stable/distributed.html#initialization)
#[cxx::bridge]
pub mod ffi {

    #[namespace = "c10d"]
    unsafe extern "C++" {
        include!("torch_comm_process_group.h");

        /// 表示`c10d::ProcessGroupNCCL`,详见torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp
        pub type ProcessGroupNCCL;
        /// 表示`c10d::ProcessGroupNCCL::Options`,详见torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp
        pub type ProcessGroupNCCLOptions;

        type PrefixStore = crate::cxx_wrapper::torch_comm_store::ffi::PrefixStore;
        type TCPStore = crate::cxx_wrapper::torch_comm_store::ffi::TCPStore;

        /// 实例化`ProcessGroupNCCLOptions`,用于后续`ProcessGroupNCCL`的创建
        fn new_process_group_nccl_options(
            backend: String,
            timeout: i64,
            is_high_priority_stream: bool,
        ) -> UniquePtr<ProcessGroupNCCLOptions>;

        /// 利用`TCPStore`创建`ProcessGroupNCCL`
        fn new_process_group_nccl_with_tcp_store(
            store: UniquePtr<TCPStore>,
            rank: i32,
            size: i32,
            options: UniquePtr<ProcessGroupNCCLOptions>,
        ) -> UniquePtr<ProcessGroupNCCL>;

        /// 利用`PrefixStore`创建`ProcessGroupNCCL`
        fn new_process_group_nccl_with_prefix_store(
            store: UniquePtr<PrefixStore>,
            rank: i32,
            size: i32,
            options: UniquePtr<ProcessGroupNCCLOptions>,
        ) -> UniquePtr<ProcessGroupNCCL>;
    }
}
