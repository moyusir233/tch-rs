pub mod torch_distributed {

    /// 定义实例化进程组时所需要的一系列kv store,可见文档:
    /// [pytorch-distributed-kv-store](https://pytorch.org/docs/stable/distributed.html#distributed-key-value-store)
    pub mod comm_store {
        use autocxx::prelude::*;

        include_cpp! {
            #include "wrappers/torch_distributed/comm_store.h"
            safety!(unsafe_ffi)
            generate_pod!("c10d::MyTCPStoreOptions")
            generate!("c10d::ArcTCPStore")
            generate!("c10d::ArcPrefixStore")
        }

        pub use ffi::{c10d::*, *};
    }

    /// 定义支持多进程之间通信的不同通信进程组,可见文档:
    /// [pytorch-distributed-process_group](https://pytorch.org/docs/stable/distributed.html#initialization)
    pub mod comm_process_group {
        use autocxx::prelude::*;

        include_cpp! {
            #include "wrappers/torch_distributed/comm_process_group.h"
            name!(ffi2)
            safety!(unsafe_ffi)
            generate_pod!("c10d::ProcessGroupNCCLOptions")
            extern_cpp_opaque_type!("c10d::ArcTCPStore",crate::wrappers::autocxx_wrappers::torch_distributed::comm_store::ArcTCPStore)
            extern_cpp_opaque_type!("c10d::ArcPrefixStore",crate::wrappers::autocxx_wrappers::torch_distributed::comm_store::ArcPrefixStore)
            generate!("c10d::ArcProcessGroupNCCL")
        }

        pub use ffi2::{c10d::*, *};
    }
}
