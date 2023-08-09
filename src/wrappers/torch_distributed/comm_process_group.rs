use autocxx::WithinUniquePtr;
use cxx::UniquePtr;
use std::ffi::CString;
use std::sync::OnceLock;
use std::time::Duration;
pub use torch_sys::wrappers::torch_distributed::comm_process_group::*;
use torch_sys::wrappers::torch_distributed::comm_store::{
    MyTCPStoreOptions, NestPrefixStore, Store,
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
    ) -> UniquePtr<ArcProcessGroupNCCL>;
}

async fn _store_based_barrier(
    store: &mut UniquePtr<ArcPrefixStore>,
    world_size: usize,
    timeout: Duration,
) {
    static STORE_BARRIER_KEY: OnceLock<CString> = OnceLock::new();
    let store_key =
        STORE_BARRIER_KEY.get_or_init(|| CString::new("store_based_barrier_key").unwrap()).as_ref();

    store.pin_mut().add(store_key, 1);

    let start = std::time::Instant::now();
    let check_interval = Duration::from_millis(10);
    let mut interval = tokio::time::interval(check_interval);

    loop {
        interval.tick().await;
        if store.pin_mut().add(store_key, 0) == world_size as i64 {
            break;
        }
        if start.elapsed() >= timeout {
            panic!("store barrier is timeout!");
        }
    }
}

impl ProcessGroupExt for ArcProcessGroupNCCL {
    /// 创建复用的tcp store,然后创建process group
    async fn new_with_tcp_store(
        group_name: &str,
        address: std::net::SocketAddr,
        timeout: Option<Duration>,
        world_size: usize,
        rank: usize,
        opts: Option<ProcessGroupNCCLOptions>,
    ) -> UniquePtr<ArcProcessGroupNCCL> {
        assert_ne!(world_size, 0, "The world_size of ProcessGroup must greater than zero!");
        assert!(
            rank < world_size,
            "The rank:{} must not greater than the world_size:{}!",
            rank,
            world_size
        );

        let tcp_opts = {
            let MyTCPStoreOptions {
                waitWorkers: default_wait_workers,
                timeout: default_timeout,
                ..
            } = MyTCPStoreOptions::default();
            MyTCPStoreOptions {
                port: address.port(),
                isServer: rank == 0,
                numWorkers: world_size,
                waitWorkers: default_wait_workers,
                timeout: timeout
                    .map(|timeout| timeout.as_millis() as i64)
                    .unwrap_or(default_timeout),
                multiTenant: true,
            }
        };

        let mut prefix_store = {
            // tcp store的创建是阻塞的,因此需要在blocking线程上完成
            let tcp_store = tokio::task::spawn_blocking({
                let tcp_opts = tcp_opts.clone();
                move || ArcTCPStore::new(address.ip().to_string(), &tcp_opts).within_unique_ptr()
            })
            .await
            .unwrap();

            let mut prefix_store = ArcPrefixStore::nest_store(group_name.into(), tcp_store);
            prefix_store.pin_mut().set_timeout(Duration::from_millis(tcp_opts.timeout as u64));
            prefix_store
        };

        let opts = opts.unwrap_or_default();
        let mut process_group = ArcProcessGroupNCCL::from_store(
            prefix_store.as_ref().unwrap(),
            rank as i32,
            world_size as i32,
            opts,
        );
        process_group.pin_mut().set_sequence_number_for_group();

        // 利用store进行同步
        _store_based_barrier(
            &mut prefix_store,
            world_size,
            Duration::from_millis(tcp_opts.timeout as u64),
        )
        .await;

        process_group
    }
}

#[cfg(test)]
mod nccl_process_group {
    use super::*;
    use std::str::FromStr;
    use std::sync::Arc;

    #[tokio::test]
    async fn init() {
        let local_set = tokio::task::LocalSet::new();
        local_set
            .run_until(async {
                let group_name = "nccl_process_group_test";
                let address = std::net::SocketAddr::from_str("127.0.0.1:8080").unwrap();
                let barrier = Arc::new(tokio::sync::Barrier::new(2));

                tokio::task::spawn_local({
                    let barrier = barrier.clone();
                    let address = address.clone();
                    async move {
                        let _client = ArcProcessGroupNCCL::new_with_tcp_store(
                            group_name, address, None, 2, 1, None,
                        )
                        .await;
                        barrier.wait().await;
                    }
                });

                let _master =
                    ArcProcessGroupNCCL::new_with_tcp_store(group_name, address, None, 2, 0, None)
                        .await;
                barrier.wait().await;
            })
            .await;
    }
}
