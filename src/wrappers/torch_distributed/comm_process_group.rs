use crate::error::TchResult;
use crate::wrappers::cxx_wrappers_utils::CppArc;
use crate::{TchError, Tensor};
use autocxx::WithinUniquePtr;
use cxx::UniquePtr;
use smallvec::SmallVec;
use std::ffi::CStr;
use std::future::{Future, IntoFuture};
use std::time::Duration;
pub use torch_sys::wrappers::torch_distributed::comm_process_group::*;
use torch_sys::wrappers::torch_distributed::comm_store::{
    MyTCPStoreOptions, NestPrefixStore, Store,
};

pub type ReduceOp = ReduceOp_RedOpType;

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

/// 确保每个进程组名与所使用的设备唯一
fn check_group_name_and_device(group_name: &str, device: crate::Device) {
    use std::collections::HashSet;
    use std::sync::Mutex;

    assert!(device.is_cuda(), "Process group device must be cuda device!");

    let group_key = format!("{}_{}", group_name, device.c_int());
    static PROCESS_GROUP_TABLE: Mutex<Option<HashSet<String>>> = std::sync::Mutex::new(None);
    let mut table = PROCESS_GROUP_TABLE.lock().unwrap();
    if table.is_none() {
        let _ = table.insert(Default::default());
    }

    assert!(!table.as_ref().unwrap().contains(&group_key), "Process group has been registry!");
    table.as_mut().unwrap().insert(group_key);
}

/// 由集合通信操作所需操作的最大张量数量来决定,通常是单机8卡,
/// 一次集合通信操作涉及到最多8个张量
const MAX_SMALL_VEC_SIZE: u8 = 8;
#[inline]
fn to_small_vec(
    tensors: &[Tensor],
) -> SmallVec<[*mut torch_sys::C_tensor; MAX_SMALL_VEC_SIZE as usize]> {
    tensors.iter().map(|tensor| tensor.c_tensor).collect()
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

/// 对[`ArcProcessGroupNCCL`]的进一步封装,
/// 尽管所包装的`inner`是Send的,但因为torch底层的实现每次都需要
/// 加锁来查表以复用ncclComms,并且多线程同时进行cuda相关操作似乎会带来
/// 额外的开销,因此这里利用`PhantomData`来使得`ProcessGroupNCCL`是非Send的,
/// 且保证`ProcessGroupNCCL`独占单个设备
#[allow(dead_code)]
pub struct ProcessGroupNCCL {
    inner: CppArc<ArcProcessGroupNCCL>,
    pub group_name: String,
    pub world_size: usize,
    pub rank: usize,
    pub address: std::net::SocketAddr,
    pub device: crate::Device,
    _unsync_mark: std::marker::PhantomData<*mut ()>,
}

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
        timeout: Duration,
    ) {
        // 先利用store进行同步
        Self::store_based_barrier(store, world_size, timeout).await;

        // 第一次进行nccl集合通信时时会造成阻塞,需要在其他blocking线程上进行
        tokio::task::spawn_blocking({
            let process_group = self.inner.clone();
            let device = self.device;

            move || {
                let mut process_group = process_group.0;

                let mut work = process_group
                    .pin_mut()
                    .ArcProcessGroupNCCL_barrier_(DeviceIDs::from(
                        [device.c_int() as i64].as_slice(),
                    ))
                    .within_unique_ptr();
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

macro_rules! check_tensor_device {
    ($tensor:ident,$target_device:expr) => {
        debug_assert_eq!(
            $tensor.device(),
            $target_device,
            "Input tensor's device is different with PorcessGroup!"
        );
    };
}

macro_rules! batch_check_tensor_device {
    ($tensors:ident,$target_device:expr) => {
        #[cfg(debug_assertions)]
        {
            for tensor in $tensors.iter() {
                debug_assert_eq!(
                    tensor.device(),
                    $target_device,
                    "Input tensor's device is different with PorcessGroup!"
                );
            }
        }
    };
}

macro_rules! generate_and_await_work {
    ($work_generate_exp:expr) => {
        CppArc::from(unsafe { $work_generate_exp.within_unique_ptr() }).await
    };
}

impl ProcessGroupNCCL {
    /// 基于TCPStore来创建独占单个加速器的nccl通信进程组，group_name需保证唯一
    pub async fn new(
        group_name: &str,
        address: std::net::SocketAddr,
        timeout: Option<Duration>,
        world_size: usize,
        rank: usize,
        device: crate::Device,
        opts: Option<ProcessGroupNCCLOptions>,
    ) -> Self {
        assert_ne!(world_size, 0, "The world_size of ProcessGroup must greater than zero!");
        assert!(
            rank < world_size,
            "The rank:{} must not greater than the world_size:{}!",
            rank,
            world_size
        );
        check_group_name_and_device(group_name, device);

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

        let mut process_group = Self {
            inner: process_group.into(),
            group_name: group_name.into(),
            world_size,
            rank,
            address,
            device,
            _unsync_mark: std::marker::PhantomData,
        };
        // 进行进程组的同步
        process_group.sync_process_group(&mut prefix_store, world_size, timeout).await;

        process_group
    }

    /// 使用包装的ArcProcessGroupNCCL进行广播通信
    async fn _broadcast(
        &mut self,
        tensor: *mut torch_sys::C_tensor,
        src_rank: usize,
    ) -> TchResult<()> {
        generate_and_await_work!(self
            .inner
            .pin_mut()
            .ArcProcessGroupNCCL_broadcast_(tensor, (src_rank as i32).into()))
    }

    /// 广播发送操作，将张量发送到进程组中的其他所有rank中
    pub async fn broadcast_send(&mut self, tensor: &Tensor) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        self._broadcast(tensor.c_tensor, self.rank).await
    }

    /// 广播接收操作，从src_rank接收张量，并写入到传入的张量当中
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub async fn broadcast_receive(
        &mut self,
        tensor: &mut Tensor,
        src_rank: usize,
    ) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        self._broadcast(tensor.c_tensor, src_rank).await
    }

    /// allreduce操作
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub async fn all_reduce(&mut self, tensor: &mut Tensor, reduce_op: ReduceOp) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        generate_and_await_work!(self
            .inner
            .pin_mut()
            .ArcProcessGroupNCCL_all_reduce_(tensor.c_tensor, reduce_op))
    }

    async fn _reduce(
        &mut self,
        tensor: *mut torch_sys::C_tensor,
        dst_rank: usize,
        reduce_op: ReduceOp,
    ) -> TchResult<()> {
        generate_and_await_work!(self.inner.pin_mut().ArcProcessGroupNCCL_reduce_(
            tensor,
            (dst_rank as i32).into(),
            reduce_op
        ))
    }

    /// reduce的发送操作,发送的张量将最终归约到`dst_rank`
    pub async fn reduce_send(
        &mut self,
        tensor: &Tensor,
        dst_rank: usize,
        reduce_op: ReduceOp,
    ) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        self._reduce(tensor.c_tensor, dst_rank, reduce_op).await
    }

    /// reduce的接收操作,从其他rank处接收与规约张量
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub async fn reduce_receive(
        &mut self,
        tensor: &mut Tensor,
        reduce_op: ReduceOp,
    ) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        self._reduce(tensor.c_tensor, self.rank, reduce_op).await
    }

    pub async fn all_gather(
        &mut self,
        input_tensor: &Tensor,
        output_tensors: impl AsRef<[Tensor]>,
    ) -> TchResult<()> {
        check_tensor_device!(input_tensor, self.device);

        let output_tensors = output_tensors.as_ref();
        batch_check_tensor_device!(output_tensors, self.device);

        generate_and_await_work!(self.inner.pin_mut().ArcProcessGroupNCCL_all_gather_(
            input_tensor.c_tensor,
            to_small_vec(output_tensors).as_slice().into(),
        ))
    }

    pub async fn all_gather_into_tensor(
        &mut self,
        input_tensor: &Tensor,
        output_tensor: &mut Tensor,
    ) -> TchResult<()> {
        check_tensor_device!(input_tensor, self.device);
        check_tensor_device!(output_tensor, self.device);

        generate_and_await_work!(self.inner.pin_mut().ArcProcessGroupNCCL_all_gather_into_tensor_(
            input_tensor.c_tensor,
            output_tensor.c_tensor,
        ))
    }

    async fn _gather(
        &mut self,
        input_tensor: *mut torch_sys::C_tensor,
        output_tensors: Tensors,
        dst_rank: usize,
    ) -> TchResult<()> {
        generate_and_await_work!(self.inner.pin_mut().ArcProcessGroupNCCL_gather_(
            input_tensor,
            output_tensors,
            (dst_rank as i32).into()
        ))
    }

    pub async fn gather_send(&mut self, input_tensor: &Tensor, dst_rank: usize) -> TchResult<()> {
        self._gather(input_tensor.c_tensor, [].as_slice().into(), dst_rank).await
    }

    pub async fn gather_receive(
        &mut self,
        input_tensor: &Tensor,
        mut output_tensors: impl AsMut<[Tensor]>,
    ) -> TchResult<()> {
        self._gather(
            input_tensor.c_tensor,
            to_small_vec(output_tensors.as_mut()).as_slice().into(),
            self.rank,
        )
        .await
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
        devices: [crate::Device; WORLD_SIZE],
    ) -> [ProcessGroupNCCL; WORLD_SIZE] {
        let local_set = tokio::task::LocalSet::new();
        let address = std::net::SocketAddr::from_str(address).unwrap();
        let barrier = Arc::new(tokio::sync::Barrier::new(WORLD_SIZE));

        let handlers: [_; WORLD_SIZE] = std::array::from_fn(|i| {
            local_set.spawn_local({
                let barrier = barrier.clone();
                async move {
                    let process_group = ProcessGroupNCCL::new(
                        group_name, address, None, WORLD_SIZE, i, devices[i], None,
                    )
                    .await;
                    barrier.wait().await;
                    process_group
                }
            })
        });
        local_set.await;

        let mut process_groups: [_; WORLD_SIZE] = std::array::from_fn(|_| ProcessGroupNCCL {
            inner: CppArc(UniquePtr::null()),
            group_name: Default::default(),
            world_size: 0,
            rank: 0,
            address: "127.0.0.1:8080".parse().unwrap(),
            device: crate::Device::Cpu,
            _unsync_mark: Default::default(),
        });
        for (i, handler) in handlers.into_iter().enumerate() {
            process_groups[i] = handler.await.unwrap();
        }

        process_groups
    }

    #[tokio::test]
    async fn init() {
        crate::Cuda::device_count();
        create_process_group::<2>(
            "init_test",
            "127.0.0.1:8081",
            std::array::from_fn(crate::Device::Cuda),
        )
        .await;
    }

    #[should_panic]
    #[tokio::test]
    async fn failed_init() {
        crate::Cuda::device_count();
        create_process_group::<1>(
            "init_test",
            "127.0.0.1:8081",
            std::array::from_fn(crate::Device::Cuda),
        )
        .await;
        create_process_group::<1>(
            "init_test",
            "127.0.0.1:8081",
            std::array::from_fn(crate::Device::Cuda),
        )
        .await;
    }

    #[tokio::test]
    async fn broadcast() -> anyhow::Result<()> {
        const WORLD_SIZE: usize = 2;
        let devices = std::array::from_fn(crate::Device::Cuda);

        let src_tensor = crate::Tensor::rand([5, 5], (crate::Kind::Float, devices[0]));
        let dst_tensors: Vec<_> = devices
            .iter()
            .skip(1)
            .map(|device| crate::Tensor::zeros([5, 5], (crate::Kind::Float, *device)))
            .collect();

        let groups =
            create_process_group::<WORLD_SIZE>("broadcast_test", "127.0.0.1:8080", devices).await;

        let local_set = tokio::task::LocalSet::new();
        let mut handlers = Vec::with_capacity(groups.len());

        for (i, mut group) in groups.into_iter().enumerate() {
            let handler = if i == 0 {
                let tensor = src_tensor.shallow_clone();
                local_set.spawn_local(async move { group.broadcast_send(&tensor).await })
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
