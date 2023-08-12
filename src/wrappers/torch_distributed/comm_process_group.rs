use crate::error::TchResult;
use crate::wrappers::cxx_wrappers_utils::CppArc;
use crate::{TchError, Tensor};
use autocxx::WithinUniquePtr;
use cxx::UniquePtr;
use std::ffi::CStr;
use std::future::{Future, IntoFuture};
use std::time::Duration;
pub use torch_sys::wrappers::torch_distributed::comm_process_group::*;
use torch_sys::wrappers::torch_distributed::comm_store::{
    MyTCPStoreOptions, NestPrefixStore, Store,
};

/// 阻塞地等待`Work`完成执行,如果发生了错误,则返回描述错误的字符串
fn wait_work(work: &mut UniquePtr<ArcWork>) -> Result<(), &str> {
    const EMPTY_ERR_MSG: &str = "empty error msg";

    if !work.pin_mut().wait() {
        let err_msg = work.exception();
        if err_msg.is_null() {
            Err(EMPTY_ERR_MSG)
        } else {
            Err(unsafe { std::str::from_utf8_unchecked(CStr::from_ptr(err_msg).to_bytes()) })
        }
    } else {
        Ok(())
    }
}

impl IntoFuture for CppArc<ArcWork> {
    type Output = TchResult<()>;

    type IntoFuture = impl Future<Output = Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        async move {
            let mut work = self.0;
            // 在spawn blocking创建的阻塞线程上阻塞
            tokio::task::spawn_blocking(move || {
                wait_work(&mut work).map_err(|err| {
                    TchError::Communication(format!("Failed communication operation, err: {err}"))
                })
            })
            .await
            .expect("Failed to join blocking thread handler when wait work")
        }
    }
}

pub type ProcessGroupNCCL = CppArc<ArcProcessGroupNCCL>;
// TODO 现已知ncclComms只有当参与集合通信的输入张量的设备不同时才会进行创建,
// 因此可以设置为一个ProcessGroupNCCL仅关联单个设备，并在创建时触发单个设备上的ncclComms初始化即可,
// 这样后续的集合通信操作就不会被阻塞.现在就考虑在rust一侧增加设备字段还是c++一侧,
// 也许在c++一侧增加比较好，可以重用一些已有的字段
impl ProcessGroupNCCL {
    async fn store_based_barrier(
        store: &mut UniquePtr<ArcPrefixStore>,
        world_size: usize,
        timeout: Duration,
    ) {
        const STORE_BARRIER_KEY: &CStr = cstr::cstr!(store_based_barrier_key);

        store.pin_mut().add(STORE_BARRIER_KEY, 1);

        let start = std::time::Instant::now();
        let check_interval = Duration::from_millis(10);
        let mut interval = tokio::time::interval(check_interval);

        loop {
            interval.tick().await;
            use std::cmp::Ordering;
            match store.pin_mut().add(STORE_BARRIER_KEY, 0).cmp(&(world_size as i64)) {
                Ordering::Equal => break,
                Ordering::Greater => panic!("Sync count greater than the world_size!"),
                Ordering::Less => (),
            }
            if start.elapsed() >= timeout {
                panic!("store barrier is timeout!");
            }
        }
    }

    /// 进行进程组的同步,也触发底层ncclComms的创建
    async fn sync_process_group(
        &mut self,
        store: &mut UniquePtr<ArcPrefixStore>,
        world_size: usize,
        rank: usize,
        timeout: Duration,
    ) {
        // 先利用store进行同步
        Self::store_based_barrier(store, world_size, timeout).await;

        // 第一次broadcast时会造成阻塞,需要在其他blocking线程上进行
        tokio::task::spawn_blocking({
            let process_group = self.clone();

            move || {
                let mut process_group = process_group.0;
                let tensor = if rank == 0 {
                    crate::Tensor::ones([3, 3], (crate::Kind::Int, crate::Device::Cuda(0)))
                        * (world_size as i64)
                } else {
                    crate::Tensor::zeros([3, 3], (crate::Kind::Int, crate::Device::Cuda(0)))
                };

                // 利用broadcast进行nccl comm的初始化与同步
                let mut work = unsafe {
                    process_group
                        .pin_mut()
                        .ArcProcessGroupNCCL_broadcast_(tensor.c_tensor, 0.into())
                        .within_unique_ptr()
                };

                wait_work(&mut work).map_err(|err| {
                    TchError::Communication(format!(
                        "Failed to sync process group by nccl Broadcast communication, err: {err}"
                    ))
                })
            }
        })
        .await
        .expect("Failed to join the blocking thread handler when sync process group")
        .unwrap();
    }
}

impl ProcessGroupNCCL {
    /// 基于TCPStore来创建nccl通信进程组，group_name需保证唯一
    pub async fn new(
        group_name: &str,
        address: std::net::SocketAddr,
        timeout: Option<Duration>,
        world_size: usize,
        rank: usize,
        opts: Option<ProcessGroupNCCLOptions>,
    ) -> Self {
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
        let mut process_group = CppArc::from(process_group);
        process_group.sync_process_group(&mut prefix_store, world_size, rank, timeout).await;

        process_group
    }

    /// 使用包装的ArcProcessGroupNCCL进行广播通信
    async fn _broadcast(
        &mut self,
        tensor: *mut torch_sys::C_tensor,
        src_rank: usize,
    ) -> TchResult<()> {
        let work: CppArc<_> = unsafe {
            self.0.pin_mut().ArcProcessGroupNCCL_broadcast_(tensor, (src_rank as i32).into())
        }
        .within_unique_ptr()
        .into();
        work.await
    }

    /// 广播发送操作，将张量发送到进程组中的其他所有rank中
    pub async fn broadcast_send(&mut self, tensor: &Tensor, src_rank: usize) -> TchResult<()> {
        self._broadcast(tensor.c_tensor, src_rank).await
    }

    /// 广播接收操作，从src_rank接收张量，并写入到传入的张量当中
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub async fn broadcast_receive(
        &mut self,
        tensor: &mut Tensor,
        src_rank: usize,
    ) -> TchResult<()> {
        self._broadcast(tensor.c_tensor, src_rank).await
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
    ) -> [ProcessGroupNCCL; WORLD_SIZE] {
        let local_set = tokio::task::LocalSet::new();
        let mut handlers = Vec::with_capacity(WORLD_SIZE);

        let address = std::net::SocketAddr::from_str(address).unwrap();
        let barrier = Arc::new(tokio::sync::Barrier::new(WORLD_SIZE));

        for i in 0..WORLD_SIZE {
            let handler = local_set.spawn_local({
                let barrier = barrier.clone();
                async move {
                    let process_group =
                        ProcessGroupNCCL::new(group_name, address, None, WORLD_SIZE, i, None).await;
                    barrier.wait().await;
                    process_group
                }
            });
            handlers.push(handler);
        }
        local_set.await;

        let mut process_groups: [_; WORLD_SIZE] =
            std::array::from_fn(|_| CppArc(UniquePtr::null()));
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

    #[tokio::test]
    async fn broadcast() -> anyhow::Result<()> {
        let device = crate::Device::Cuda(0);
        const WORLD_SIZE: usize = 2;

        let src_tensor = crate::Tensor::rand([5, 5], (crate::Kind::Float, device));
        let dst_tensors: Vec<_> =
            std::iter::repeat_with(|| crate::Tensor::zeros([5, 5], (crate::Kind::Float, device)))
                .into_iter()
                .take(WORLD_SIZE - 1)
                .collect();

        let groups = create_process_group::<WORLD_SIZE>("broadcast_test", "127.0.0.1:8080").await;
        let local_set = tokio::task::LocalSet::new();
        let mut handlers = Vec::with_capacity(groups.len());

        for (i, mut group) in groups.into_iter().enumerate() {
            let handler = if i == 0 {
                let tensor = src_tensor.shallow_clone();
                local_set.spawn_local(async move { group.broadcast_send(&tensor, 0).await })
            } else {
                let mut tensor = dst_tensors[i - 1].shallow_clone();
                local_set.spawn_local(async move { group.broadcast_receive(&mut tensor, 0).await })
            };
            handlers.push(handler);
        }
        local_set.await;

        for handler in handlers {
            let result = handler.await;
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
