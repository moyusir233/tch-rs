use crate::error::TchResult;
use crate::wrappers::cxx_wrappers_utils::CppArc;
use crate::{TchError, Tensor};
use autocxx::WithinUniquePtr;
use cxx::UniquePtr;
use std::ffi::CString;
use std::pin::Pin;
use std::sync::OnceLock;
use std::time::Duration;
pub use torch_sys::wrappers::torch_distributed::comm_process_group::*;
use torch_sys::wrappers::torch_distributed::comm_store::{
    MyTCPStoreOptions, NestPrefixStore, Store,
};
use torch_sys::C_tensor;

pub type ProcessGroupNCCL = CppArc<ArcProcessGroupNCCL>;

impl ProcessGroupNCCL {
    pub async fn new(
        group_name: &str,
        address: std::net::SocketAddr,
        timeout: Option<Duration>,
        world_size: usize,
        rank: usize,
        opts: Option<ProcessGroupNCCLOptions>,
    ) -> Self {
        todo!()
    }

    pub async fn broadcast(&mut self, tensor: &mut Tensor, src_rank: usize){
        todo!()
    }

}

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

    /// 广播集合通信操作
    async fn broadcast(self: Pin<&mut Self>, tensor: &mut Tensor, src_rank: usize)
        -> TchResult<()>;
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
        use std::cmp::Ordering;
        match store.pin_mut().add(store_key, 0).cmp(&(world_size as i64)) {
            Ordering::Equal => break,
            Ordering::Greater => panic!("Sync count greater than the world_size!"),
            Ordering::Less => (),
        }
        if start.elapsed() >= timeout {
            panic!("store barrier is timeout!");
        }
    }
}

/// 进行进程组的同步,也进行底层nccl comm的创建
async fn sync_process_group(
    store: &mut UniquePtr<ArcPrefixStore>,
    process_group: &mut UniquePtr<ArcProcessGroupNCCL>,
    world_size: usize,
    rank: usize,
    timeout: Duration,
) {
    // 先利用store进行同步
    _store_based_barrier(store, world_size, timeout).await;

    // TODO 利用一次伪的broadcast操作触发底层nccl comm的创建以及其他阻塞的同步操作
    let mut tensor = if rank == 0 {
        crate::Tensor::ones([3, 3], (crate::Kind::Int, crate::Device::Cuda(0))) * world_size
    } else {
        crate::Tensor::zeros([3, 3], (crate::Kind::Int, crate::Device::Cuda(0)))
    };
    // 第一次broadcast时会造成阻塞,需要在其他blocking线程上进行
    ProcessGroupExt::broadcast(process_group.pin_mut(), &mut tensor, 0)
        .await
        .context("failed to sync by the broadcast")
        .unwrap();
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

        let MyTCPStoreOptions {
            waitWorkers: default_wait_workers, timeout: default_timeout, ..
        } = MyTCPStoreOptions::default();
        let timeout = timeout.unwrap_or(Duration::from_millis(default_timeout as u64));
        let tcp_opts = MyTCPStoreOptions {
            port: address.port(),
            isServer: rank == 0,
            numWorkers: world_size,
            waitWorkers: default_wait_workers,
            timeout: timeout.as_millis() as i64,
            multiTenant: true,
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

        // 进行进程组的同步
        sync_process_group(&mut prefix_store, &mut process_group, world_size, rank, timeout).await;

        process_group
    }

    async fn broadcast(
        self: Pin<&mut Self>,
        tensor: &mut Tensor,
        src_rank: usize,
    ) -> TchResult<()> {
        let work =
            unsafe { self.broadcast(tensor.c_tensor as *mut C_tensor, (src_rank as i32).into()) }
                .within_unique_ptr();
        CxxIntoFuture::into_future(work).await
    }
}

#[cfg(test)]
mod nccl_process_group {
    use super::*;
    use anyhow::Context;
    use std::str::FromStr;
    use std::sync::Arc;

    async fn create_process_group<const WORLD_SIZE: usize>(
        group_name: &'static str,
        address: &str,
    ) -> [UniquePtr<ArcProcessGroupNCCL>; WORLD_SIZE] {
        let local_set = tokio::task::LocalSet::new();
        let mut handlers = Vec::with_capacity(WORLD_SIZE);

        let address = std::net::SocketAddr::from_str(address).unwrap();
        let barrier = Arc::new(tokio::sync::Barrier::new(WORLD_SIZE));

        for i in 0..WORLD_SIZE {
            let handler = local_set.spawn_local({
                let barrier = barrier.clone();
                let address = address.clone();
                async move {
                    let process_group = ArcProcessGroupNCCL::new_with_tcp_store(
                        group_name, address, None, WORLD_SIZE, i, None,
                    )
                    .await;
                    barrier.wait().await;
                    process_group
                }
            });
            handlers.push(handler);
        }
        local_set.await;

        let mut process_groups: [_; WORLD_SIZE] = std::array::from_fn(|_| UniquePtr::null());
        for (i, handler) in handlers.into_iter().enumerate() {
            process_groups[i] = handler.await.unwrap();
        }

        process_groups
    }

    #[tokio::test]
    async fn init() {
        create_process_group::<3>("init_test", "127.0.0.1:8081").await;
        create_process_group::<4>("init_test", "127.0.0.1:8081").await;
        create_process_group::<5>("init_test", "127.0.0.1:8081").await;
    }

    // FIXME 修复第一次进行集合通信操作时底层进行的额外同步操作而导致阻塞的问题
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn broadcast() -> anyhow::Result<()> {
        let device = crate::Device::Cuda(0);
        let world_size = 2;

        let mut src_tensor = crate::Tensor::rand([5, 5], (crate::Kind::Float, device));
        let mut dst_tensors: Vec<_> =
            std::iter::repeat_with(|| crate::Tensor::zeros([5, 5], (crate::Kind::Float, device)))
                .into_iter()
                .take(world_size - 1)
                .collect();

        let mut groups = create_process_group::<2>("broadcast_test", "127.0.0.1:8080").await;
        let mut join_set = tokio::task::JoinSet::new();

        for (i, mut group) in groups.into_iter().enumerate() {
            let mut tensor = if i == 0 {
                src_tensor.shallow_clone()
            } else {
                dst_tensors[i - 1].shallow_clone()
            };
            join_set.spawn(async move {
                ProcessGroupExt::broadcast(group.pin_mut(), &mut tensor, 0).await
            });
        }

        while let Some(result) = join_set.join_next().await {
            result.context("failed to join the task")?.context("failed to broadcast")?;
        }

        for dst_tensor in dst_tensors {
            assert!(
                src_tensor.equal(&dst_tensor),
                "dst tensor is different with src tensor!\nsrc:{},dst:{}",
                src_tensor,
                dst_tensor
            );
        }

        Ok(())
    }
}
