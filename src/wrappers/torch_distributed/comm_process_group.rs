use autocxx::WithinUniquePtr;
use cxx::UniquePtr;
use std::sync::atomic::{AtomicU16, Ordering};
use std::time::Duration;
pub use torch_sys::wrappers::torch_distributed::comm_process_group::*;
use torch_sys::wrappers::torch_distributed::comm_store::{
    ArcPrefixStore, ArcTCPStore, MyTCPStoreOptions, NestPrefixStore, Store,
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
    const STORE_BARRIER_KEY: &str = "store_based_barrier_key";
    let store_key = format!("{}:{}", STORE_BARRIER_KEY, GROUP_COUNT.load(Ordering::Relaxed));
    cxx::let_cxx_string!(store_key = store_key);

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
            let tcp_store =
                ArcTCPStore::new(address.ip().to_string(), &tcp_opts).within_unique_ptr();

            let mut prefix_store = ArcPrefixStore::nest_store(
                format!("{}/{}", GROUP_COUNT.fetch_add(1, Ordering::Relaxed), group_name),
                tcp_store,
            );
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

    // FIXME 下面的测试似乎会死锁
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn init() {
        let group_name = "nccl_process_group_test";
        let address = std::net::SocketAddr::from_str("127.0.0.1:8080").unwrap();
        let barrier = Arc::new(tokio::sync::Barrier::new(2));

        let local_set = tokio::task::LocalSet::new();
        let _guard = local_set.enter();

        tokio::task::spawn_local({
            let barrier = barrier.clone();
            async move {
                let client = ArcProcessGroupNCCL::new_with_tcp_store(
                    group_name,
                    address.clone(),
                    None,
                    2,
                    1,
                    None,
                )
                .await;
                barrier.wait().await;
            }
        });

        let master =
            ArcProcessGroupNCCL::new_with_tcp_store(group_name, address.clone(), None, 2, 0, None)
                .await;
        barrier.wait().await;
    }
}
