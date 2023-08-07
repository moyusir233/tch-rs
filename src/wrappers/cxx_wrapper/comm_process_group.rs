use cxx::UniquePtr;
use std::ffi::CString;
use std::pin::Pin;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::time::Duration;
pub use torch_sys::cxx_wrapper::torch_comm_process_group::*;
use torch_sys::cxx_wrapper::torch_comm_store::{
    MyTCPStoreOptions, NestPrefixStore, PrefixStore, Store, TCPStore,
};

pub trait ProcessGroupExt {
    /// 基于tcp store来初始化nccl通信进程组
    async fn new_with_tcp_store(
        group_name: &str,
        address: std::net::SocketAddr,
        timeout: Option<Duration>,
        world_size: usize,
        rank: usize,
        opts: Option<ProcessGroupNCCLOptions>,
    ) -> UniquePtr<ProcessGroupNCCL>;
}

async fn _store_based_barrier(
    store: &mut UniquePtr<PrefixStore>,
    world_size: usize,
    timeout: Duration,
) {
    const STORE_BARRIER_KEY: &str = "store_based_barrier_key";
    let store_key =
        CString::new(format!("{}:{}", STORE_BARRIER_KEY, GROUP_COUNT.load(Ordering::Relaxed)))
            .unwrap();
    store.pin_mut().add(&store_key, 1);

    let start = std::time::Instant::now();
    let check_interval = Duration::from_millis(10);
    let mut interval = tokio::time::interval(check_interval);

    loop {
        interval.tick().await;
        if store.pin_mut().add(&store_key, 0) == world_size as i64 {
            break;
        }
        if start.elapsed() >= timeout {
            panic!("store barrier is timeout!");
        }
    }
}

static GROUP_COUNT: AtomicU16 = AtomicU16::new(0);
impl ProcessGroupExt for ProcessGroupNCCL {
    /// 创建复用的tcp store,然后创建process group
    async fn new_with_tcp_store(
        group_name: &str,
        address: std::net::SocketAddr,
        timeout: Option<Duration>,
        world_size: usize,
        rank: usize,
        opts: Option<ProcessGroupNCCLOptions>,
    ) -> UniquePtr<ProcessGroupNCCL> {
        assert_ne!(world_size, 0, "The world_size of ProcessGroup must greater than zero!");

        let prefix_store = {
            let tcp_opts = {
                let MyTCPStoreOptions {
                    wait_workers: default_wait_workers,
                    timeout: default_timeout,
                    ..
                } = MyTCPStoreOptions::default();
                MyTCPStoreOptions {
                    port: address.port(),
                    is_server: rank == 0,
                    num_workers: world_size,
                    wait_workers: default_wait_workers,
                    timeout: timeout
                        .map(|timeout| timeout.as_millis() as i64)
                        .unwrap_or(default_timeout),
                    multi_tenant: true,
                }
            };
            let tcp_store = TCPStore::new(address.ip().to_string(), &tcp_opts);

            let mut prefix_store = PrefixStore::nest_store(
                format!("{}/{}", GROUP_COUNT.fetch_add(1, Ordering::Relaxed), group_name),
                tcp_store,
            );
            prefix_store.pin_mut().set_timeout(Duration::from_millis(tcp_opts.timeout as u64));
            prefix_store
        };

        let opts = opts.unwrap_or_default();
        let mut process_group =
            ProcessGroupNCCL::from_store(prefix_store, rank as i32, world_size as i32, opts);
        process_group.pin_mut().set_sequence_number_for_group();

        // 利用store进行同步
        // TODO 底层process_group实例实际上是保存了store智能指针的引用,但为了方便rust binding
        //  的编写直接使用了unique_ptr,如何修改binding接口以让它更符合底层c++接口的语义?
        //  关键还是如何处理自定义智能指针的问题
        process_group
    }
}
