use crate::error::TchResult;
use crate::wrappers::cxx_wrappers_utils::CppArc;
use crate::{TchError, Tensor};
use autocxx::WithinUniquePtr;
use cxx::UniquePtr;
use smallvec::SmallVec;
use std::ffi::CStr;
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

    assert!(
        !table.as_ref().unwrap().contains(&group_key),
        "Process group {group_key} has been registry!"
    );
    table.as_mut().unwrap().insert(group_key);
}

/// 由集合通信操作所需操作的最大张量数量来决定,通常是单机8卡,
/// 一次集合通信操作涉及到最多8个张量
const MAX_SMALL_VEC_SIZE: u8 = 8;
#[inline]
fn to_tensor_ptr_small_vec(
    tensors: &[Tensor],
) -> SmallVec<[*mut torch_sys::C_tensor; MAX_SMALL_VEC_SIZE as usize]> {
    tensors.iter().map(|tensor| tensor.c_tensor).collect()
}

/// 对[`ArcProcessGroupNCCL`]的进一步封装,
#[allow(dead_code)]
pub struct ProcessGroupNCCL {
    inner: CppArc<ArcProcessGroupNCCL>,
    pub group_name: String,
    pub world_size: usize,
    pub rank: usize,
    pub address: std::net::SocketAddr,
    pub device: crate::Device,
}

unsafe impl Send for ProcessGroupNCCL {}

impl ProcessGroupNCCL {
    fn store_based_barrier(
        store: &mut UniquePtr<ArcPrefixStore>,
        world_size: usize,
        timeout: Duration,
    ) {
        const STORE_BARRIER_KEY: &CStr = cstr::cstr!(store_based_barrier_key);

        store.pin_mut().add(STORE_BARRIER_KEY, 1);

        let start = std::time::Instant::now();
        let check_interval = Duration::from_millis(10);

        loop {
            std::thread::sleep(check_interval);
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
    fn sync_process_group(
        &mut self,
        store: &mut UniquePtr<ArcPrefixStore>,
        world_size: usize,
        timeout: Duration,
    ) {
        // 先利用store进行同步
        Self::store_based_barrier(store, world_size, timeout);

        let mut work = self
            .inner
            .pin_mut()
            .ArcProcessGroupNCCL_barrier_([self.device.c_int() as i64].as_slice().into())
            .within_unique_ptr();
        wait_work(&mut work)
            .map_err(|err| {
                TchError::Communication(format!(
                    "Failed to sync process group by nccl Broadcast communication, err: {err}"
                ))
            })
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

macro_rules! wait_work {
    ($generate_work_expr:expr) => {{
        let mut work = $generate_work_expr.within_unique_ptr();
        wait_work(&mut work).map_err(|err| {
            TchError::Communication(format!("Failed to do communication operation, err: {err}"))
        })
    }};
}

impl ProcessGroupNCCL {
    /// 基于TCPStore来创建独占单个加速器的nccl通信进程组，group_name需保证唯一
    pub fn new(
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
            let tcp_store =
                ArcTCPStore::new(address.ip().to_string(), &tcp_opts).within_unique_ptr();

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
        };
        // 进行进程组的同步
        process_group.sync_process_group(&mut prefix_store, world_size, timeout);

        process_group
    }

    /// 使用包装的ArcProcessGroupNCCL进行广播通信
    fn _broadcast(&mut self, tensor: *mut torch_sys::C_tensor, src_rank: usize) -> TchResult<()> {
        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_broadcast_(tensor, (src_rank as i32).into())
        })
    }

    /// 广播发送操作，将张量发送到进程组中的其他所有rank中
    pub fn broadcast_send(&mut self, tensor: &Tensor) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        self._broadcast(tensor.c_tensor, self.rank)
    }

    /// 广播接收操作，从src_rank接收张量，并写入到传入的张量当中
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn broadcast_receive(&mut self, tensor: &mut Tensor, src_rank: usize) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        self._broadcast(tensor.c_tensor, src_rank)
    }

    /// allreduce操作
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn all_reduce(&mut self, tensor: &mut Tensor, reduce_op: ReduceOp) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_all_reduce_(tensor.c_tensor, reduce_op)
        })
    }

    fn _reduce(
        &mut self,
        tensor: *mut torch_sys::C_tensor,
        dst_rank: usize,
        reduce_op: ReduceOp,
    ) -> TchResult<()> {
        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_reduce_(
                tensor,
                (dst_rank as i32).into(),
                reduce_op,
            )
        })
    }

    /// reduce的发送操作,发送的张量将最终归约到`dst_rank`
    pub fn reduce_send(
        &mut self,
        tensor: &Tensor,
        dst_rank: usize,
        reduce_op: ReduceOp,
    ) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        self._reduce(tensor.c_tensor, dst_rank, reduce_op)
    }

    /// reduce的接收操作,从其他rank处接收与规约张量
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn reduce_receive(&mut self, tensor: &mut Tensor, reduce_op: ReduceOp) -> TchResult<()> {
        check_tensor_device!(tensor, self.device);

        self._reduce(tensor.c_tensor, self.rank, reduce_op)
    }

    pub fn all_gather(
        &mut self,
        input_tensor: &Tensor,
        mut output_tensors: impl AsMut<[Tensor]>,
    ) -> TchResult<()> {
        check_tensor_device!(input_tensor, self.device);

        let output_tensors = output_tensors.as_mut();
        batch_check_tensor_device!(output_tensors, self.device);

        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_all_gather_(
                input_tensor.c_tensor,
                to_tensor_ptr_small_vec(output_tensors).as_slice().into(),
            )
        })
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn all_gather_into_tensor(
        &mut self,
        input_tensor: &Tensor,
        output_tensor: &mut Tensor,
    ) -> TchResult<()> {
        check_tensor_device!(input_tensor, self.device);
        check_tensor_device!(output_tensor, self.device);

        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_all_gather_into_tensor_(
                input_tensor.c_tensor,
                output_tensor.c_tensor,
            )
        })
    }

    fn _gather(
        &mut self,
        input_tensor: *mut torch_sys::C_tensor,
        output_tensors: Tensors,
        dst_rank: usize,
    ) -> TchResult<()> {
        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_gather_(
                input_tensor,
                output_tensors,
                (dst_rank as i32).into(),
            )
        })
    }

    pub fn gather_send(&mut self, input_tensor: &Tensor, dst_rank: usize) -> TchResult<()> {
        self._gather(input_tensor.c_tensor, [].as_slice().into(), dst_rank)
    }

    pub fn gather_receive(
        &mut self,
        input_tensor: &Tensor,
        mut output_tensors: impl AsMut<[Tensor]>,
    ) -> TchResult<()> {
        self._gather(
            input_tensor.c_tensor,
            to_tensor_ptr_small_vec(output_tensors.as_mut()).as_slice().into(),
            self.rank,
        )
    }

    fn _scatter(
        &mut self,
        input_tensors: Tensors,
        output_tensor: *mut torch_sys::C_tensor,
        src_rank: usize,
    ) -> TchResult<()> {
        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_scatter_(
                input_tensors,
                output_tensor,
                (src_rank as i32).into(),
            )
        })
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn scatter_send(
        &mut self,
        input_tensors: impl AsRef<[Tensor]>,
        output_tensor: &mut Tensor,
    ) -> TchResult<()> {
        let input_tensors = input_tensors.as_ref();
        batch_check_tensor_device!(input_tensors, self.device);

        check_tensor_device!(output_tensor, self.device);

        self._scatter(
            to_tensor_ptr_small_vec(input_tensors).as_slice().into(),
            output_tensor.c_tensor,
            self.rank,
        )
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn scatter_receive(
        &mut self,
        output_tensor: &mut Tensor,
        src_rank: usize,
    ) -> TchResult<()> {
        check_tensor_device!(output_tensor, self.device);

        self._scatter([].as_slice().into(), output_tensor.c_tensor, src_rank)
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn reduce_scatter(
        &mut self,
        input_tensors: impl AsRef<[Tensor]>,
        output_tensor: &mut Tensor,
        reduce_op: ReduceOp,
    ) -> TchResult<()> {
        let input_tensors = input_tensors.as_ref();
        batch_check_tensor_device!(input_tensors, self.device);

        check_tensor_device!(output_tensor, self.device);

        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_reduce_scatter_(
                to_tensor_ptr_small_vec(input_tensors).as_slice().into(),
                output_tensor.c_tensor,
                reduce_op,
            )
        })
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn reduce_scatter_into_tensor(
        &mut self,
        input_tensor: &Tensor,
        output_tensor: &mut Tensor,
        reduce_op: ReduceOp,
    ) -> TchResult<()> {
        check_tensor_device!(input_tensor, self.device);

        check_tensor_device!(output_tensor, self.device);

        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_reduce_scatter_tensor_(
                input_tensor.c_tensor,
                output_tensor.c_tensor,
                reduce_op,
            )
        })
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn all_to_all_single(
        &mut self,
        input_tensor: &Tensor,
        output_tensor: &mut Tensor,
        input_split_sizes: impl AsRef<[i64]>,
        output_split_sizes: impl AsRef<[i64]>,
    ) -> TchResult<()> {
        check_tensor_device!(input_tensor, self.device);

        check_tensor_device!(output_tensor, self.device);

        wait_work!(unsafe {
            self.inner.pin_mut().ArcProcessGroupNCCL_all_to_all_single_(
                input_tensor.c_tensor,
                output_tensor.c_tensor,
                input_split_sizes.as_ref().into(),
                output_split_sizes.as_ref().into(),
            )
        })
    }

    pub fn all_to_all(
        &mut self,
        input_tensors: impl AsRef<[Tensor]>,
        mut output_tensors: impl AsMut<[Tensor]>,
    ) -> TchResult<()> {
        let input_tensors = input_tensors.as_ref();
        batch_check_tensor_device!(input_tensors, self.device);

        let output_tensors = output_tensors.as_mut();
        batch_check_tensor_device!(output_tensors, self.device);

        wait_work!(self.inner.pin_mut().ArcProcessGroupNCCL_alltoall_(
            to_tensor_ptr_small_vec(input_tensors).as_slice().into(),
            to_tensor_ptr_small_vec(output_tensors).as_slice().into(),
        ))
    }

    pub fn barrier(&mut self) -> TchResult<()> {
        wait_work!(self
            .inner
            .pin_mut()
            .ArcProcessGroupNCCL_barrier_([self.device.c_int().into()].as_slice().into()))
    }
}

#[cfg(test)]
mod nccl_process_group {
    use crate::{Device, Kind};

    use super::*;
    use anyhow::Context;
    use std::str::FromStr;
    use std::sync::atomic::{AtomicU64, Ordering};

    const CUDA_DEVICE_COUNT_STR: &str =
        konst::option::unwrap_or!(option_env!("CARGO_TEST_CUDA_DEVICE_COUNT"), "1");

    const CUDA_DEVICE_COUNT: usize =
        konst::unwrap_ctx!(konst::primitive::parse_usize(CUDA_DEVICE_COUNT_STR));

    fn create_process_group<const WORLD_SIZE: usize>(
        group_name: &str,
        address: &str,
        devices: [crate::Device; WORLD_SIZE],
    ) -> [ProcessGroupNCCL; WORLD_SIZE] {
        let address = std::net::SocketAddr::from_str(address).unwrap();

        let mut handlers: [_; WORLD_SIZE] = std::array::from_fn(|i| {
            let group_name = group_name.to_owned();
            Some(std::thread::spawn(move || {
                ProcessGroupNCCL::new(&group_name, address, None, WORLD_SIZE, i, devices[i], None)
            }))
        });

        std::array::from_fn(|i| {
            let handler = std::mem::take(&mut handlers[i]).unwrap();
            handler.join().unwrap()
        })
    }

    #[derive(Debug, Default)]
    struct RankState {
        input_tensors: Vec<Tensor>,
        output_tensors: Vec<Tensor>,
    }
    impl RankState {
        fn add_input(mut self, tensor: Tensor) -> Self {
            self.input_tensors.push(tensor);
            self
        }

        fn add_inputs<const N: usize>(mut self, create_tensor: impl Fn() -> Tensor) -> Self {
            self.input_tensors.extend(std::array::from_fn::<Tensor, N, _>(|_| create_tensor()));
            self
        }

        fn add_output(mut self, tensor: Tensor) -> Self {
            self.output_tensors.push(tensor);
            self
        }

        fn add_outputs<const N: usize>(mut self, create_tensor: impl Fn() -> Tensor) -> Self {
            self.output_tensors.extend(std::array::from_fn::<Tensor, N, _>(|_| create_tensor()));
            self
        }
    }

    trait GroupOperator: Send {
        fn handle(
            &mut self,
            process_group: ProcessGroupNCCL,
            rank_state: RankState,
        ) -> TchResult<RankState>;
    }

    macro_rules! create_box_group_operator {
        ($create_expr:expr) => {{
            // 类型推断有问题,因此需要先绑定到一个临时变量上来让编译器进行类型推导
            let operator: Box<dyn GroupOperator> = Box::new($create_expr);
            operator
        }};
    }

    #[allow(invalid_value)]
    fn collective_comm_test<F1, F2, const WORLD_SIZE: usize>(
        create_rank_state: F1,
        create_group_oper: F2,
    ) -> [RankState; WORLD_SIZE]
    where
        F1: Fn(usize) -> RankState,
        F2: Fn(usize) -> Box<dyn GroupOperator>,
    {
        static TEST_COUNT: AtomicU64 = AtomicU64::new(0);

        if TEST_COUNT.load(Ordering::Relaxed) == 0 {
            // 进行torch底层的全局初始化,包括日志等
            crate::wrappers::utils::init_torch_module()
                .expect("Failed to init torch module globally!");
        }

        let group_name =
            format!("collective_comm_test_{}", TEST_COUNT.fetch_add(1, Ordering::Relaxed));
        let address = format!("127.0.0.1:{}", 8080 + TEST_COUNT.load(Ordering::Relaxed));

        let mut groups = create_process_group::<WORLD_SIZE>(
            &group_name,
            &address,
            std::array::from_fn(crate::Device::Cuda),
        );

        let handlers: [_; WORLD_SIZE] = std::array::from_fn(|rank| {
            let rank_state = create_rank_state(rank);
            let mut group_oper = create_group_oper(rank);
            let group = std::mem::replace(&mut groups[rank], unsafe {
                // Safety: 后续利用了std::mem::forget避免了在未初始化的值上调用drop
                std::mem::MaybeUninit::uninit().assume_init()
            });
            std::thread::spawn(move || group_oper.handle(group, rank_state))
        });
        // groups中的元素都被未初始化的值所替代了,因此不再需要drop
        std::mem::forget(groups);

        let mut rank_states = std::array::from_fn(|_| Default::default());
        for (i, handler) in handlers.into_iter().enumerate() {
            let result = handler.join();
            rank_states[i] = result.unwrap().context("failed to do communication").unwrap();
        }

        rank_states
    }

    #[test]
    fn init() {
        create_process_group::<CUDA_DEVICE_COUNT>(
            "init_test",
            "127.0.0.1:7079",
            std::array::from_fn(crate::Device::Cuda),
        );
    }

    #[ignore] // 避免持有注册锁时panic,进而造成其他测试线程panic
    #[should_panic]
    #[test]
    fn failed_init() {
        create_process_group::<1>(
            "init_test1",
            "127.0.0.1:7078",
            std::array::from_fn(crate::Device::Cuda),
        );
        create_process_group::<1>(
            "init_test1",
            "127.0.0.1:7078",
            std::array::from_fn(crate::Device::Cuda),
        );
    }

    #[test]
    fn broadcast() {
        struct BroadcastSender;
        struct BroadcastReceiver {
            src_rank: usize,
        }

        impl GroupOperator for BroadcastSender {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.broadcast_send(&rank_state.input_tensors[0])?;
                Ok(rank_state)
            }
        }
        impl GroupOperator for BroadcastReceiver {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                let src_rank = self.src_rank;

                process_group.broadcast_receive(&mut rank_state.output_tensors[0], src_rank)?;
                Ok(rank_state)
            }
        }

        let broadcast_rank = 0;
        let shape = [5, 5];
        let kind = Kind::Float;

        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                if rank == broadcast_rank {
                    RankState::default().add_input(Tensor::rand(shape, (kind, Device::Cuda(rank))))
                } else {
                    RankState::default()
                        .add_output(Tensor::zeros(shape, (kind, Device::Cuda(rank))))
                }
            },
            |rank| {
                if rank == broadcast_rank {
                    create_box_group_operator!(BroadcastSender)
                } else {
                    create_box_group_operator!(BroadcastReceiver { src_rank: broadcast_rank })
                }
            },
        );
        let expected_tensor = rank_states[broadcast_rank].input_tensors[0].shallow_clone();

        for (rank, output) in rank_states
            .into_iter()
            .enumerate()
            .filter(|(rank, _state)| *rank != broadcast_rank)
            .map(|(rank, mut state)| (rank, std::mem::take(&mut state.output_tensors[0])))
        {
            if rank == broadcast_rank {
                continue;
            }
            assert!(
                output.equal(&expected_tensor.to_device(Device::Cuda(rank))),
                "The rank: {rank}'s result: {output} is different with expected:{expected_tensor}!",
            )
        }
    }

    #[test]
    fn all_reduce() {
        struct AllReduceRank;
        impl GroupOperator for AllReduceRank {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.all_reduce(&mut rank_state.input_tensors[0], ReduceOp::SUM)?;
                Ok(rank_state)
            }
        }

        let shape = [5, 5];
        let kind = Kind::Float;
        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| RankState::default().add_input(Tensor::ones(shape, (kind, Device::Cuda(rank)))),
            |_rank| create_box_group_operator!(AllReduceRank),
        );

        let expected_tensor =
            Tensor::ones(shape, (kind, Device::Cuda(0))) * CUDA_DEVICE_COUNT as i64;
        for (rank, output) in rank_states
            .into_iter()
            .map(|mut state| std::mem::take(&mut state.input_tensors[0]))
            .enumerate()
        {
            assert!(output.equal(&expected_tensor.to_device(Device::Cuda(rank))));
        }
    }

    #[test]
    fn reduce() {
        struct ReduceSender {
            dst_rank: usize,
        }
        struct ReduceReceiver;

        impl GroupOperator for ReduceSender {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                rank_state: RankState,
            ) -> TchResult<RankState> {
                let dst_rank = self.dst_rank;

                process_group.reduce_send(&rank_state.input_tensors[0], dst_rank, ReduceOp::SUM)?;
                Ok(rank_state)
            }
        }
        impl GroupOperator for ReduceReceiver {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.reduce_receive(&mut rank_state.output_tensors[0], ReduceOp::SUM)?;
                Ok(rank_state)
            }
        }

        let dst_rank = 0;
        let shape = [5, 5];
        let kind = Kind::Float;
        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                if rank == dst_rank {
                    RankState::default()
                        .add_output(Tensor::zeros(shape, (kind, Device::Cuda(rank))))
                } else {
                    RankState::default().add_input(Tensor::ones(shape, (kind, Device::Cuda(rank))))
                }
            },
            |rank| {
                if rank == dst_rank {
                    create_box_group_operator!(ReduceReceiver)
                } else {
                    create_box_group_operator!(ReduceSender { dst_rank })
                }
            },
        );

        let expected_tensor =
            Tensor::ones(shape, (kind, Device::Cuda(0))) * (CUDA_DEVICE_COUNT - 1) as i64;
        let reduced_tensor = &rank_states[dst_rank].output_tensors[0];
        assert!(
            rank_states[dst_rank].output_tensors[0]
                .equal(&expected_tensor.to_device(Device::Cuda(dst_rank))),
            "Reduced tensor: {reduced_tensor} is different with expected: {expected_tensor}!"
        );
    }

    #[test]
    fn all_gather() {
        struct AllGatherRank;
        impl GroupOperator for AllGatherRank {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                {
                    process_group.all_gather(
                        &rank_state.input_tensors[0],
                        rank_state.output_tensors.as_mut_slice(),
                    )?;
                    Ok(rank_state)
                }
            }
        }

        let shape = [5, 5];
        let kind = Kind::Float;

        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                RankState::default()
                    .add_input(Tensor::ones(shape, (kind, Device::Cuda(rank))))
                    .add_outputs::<CUDA_DEVICE_COUNT>(|| {
                    Tensor::zeros(shape, (kind, Device::Cuda(rank)))
                })
            },
            |_rank| create_box_group_operator!(AllGatherRank),
        );

        let expected_tensors = Vec::from_iter(
            std::iter::repeat_with(|| Tensor::ones(shape, (kind, Device::Cuda(0))))
                .take(CUDA_DEVICE_COUNT),
        );

        for (rank, output_tensors) in
            rank_states.into_iter().map(|state| state.output_tensors).enumerate()
        {
            assert!(output_tensors.iter().zip(expected_tensors.iter()).all(
                |(output_tensor, expected_tensor)| {
                    output_tensor.equal(&expected_tensor.to_device(Device::Cuda(rank)))
                },
            ),
            "Output tensors: {output_tensors:?} is different with expected tensors: {expected_tensors:?}"
        )
        }
    }

    #[test]
    fn all_gather_into_tensor() {
        struct AllGatherRank;
        impl GroupOperator for AllGatherRank {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                {
                    process_group.all_gather_into_tensor(
                        &rank_state.input_tensors[0],
                        &mut rank_state.output_tensors[0],
                    )?;
                    Ok(rank_state)
                }
            }
        }

        let shape = [5, 5];
        let kind = Kind::Float;

        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                RankState::default()
                    .add_input(Tensor::ones(shape, (kind, Device::Cuda(rank))))
                    .add_output(Tensor::zeros(
                        [shape[0] * CUDA_DEVICE_COUNT as i64, shape[1]],
                        (kind, Device::Cuda(rank)),
                    ))
            },
            |_rank| create_box_group_operator!(AllGatherRank),
        );

        let expected_tensor =
            Tensor::ones([shape[0] * CUDA_DEVICE_COUNT as i64, shape[1]], (kind, Device::Cuda(0)));
        for (rank, output_tensor) in rank_states
            .into_iter()
            .map(|mut state| std::mem::take(&mut state.output_tensors[0]))
            .enumerate()
        {
            assert!(output_tensor.equal(&expected_tensor.to_device(Device::Cuda(rank))),         "Output tensor: {output_tensor:?} is different with expected tensors: {expected_tensor:?}"
        )
        }
    }

    #[test]
    fn gather() {
        struct GatherSender {
            dst_rank: usize,
        }
        struct GatherReceiver;

        impl GroupOperator for GatherSender {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.gather_send(&rank_state.input_tensors[0], self.dst_rank)?;
                Ok(rank_state)
            }
        }
        impl GroupOperator for GatherReceiver {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.gather_receive(
                    &rank_state.input_tensors[0],
                    rank_state.output_tensors.as_mut_slice(),
                )?;
                Ok(rank_state)
            }
        }

        let dst_rank = 0;
        let shape = [5, 5];
        let kind = Kind::Float;
        let mut rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                if rank == dst_rank {
                    RankState::default()
                        .add_input(Tensor::ones(shape, (kind, Device::Cuda(rank))))
                        .add_outputs::<CUDA_DEVICE_COUNT>(|| {
                        Tensor::zeros(shape, (kind, Device::Cuda(rank)))
                    })
                } else {
                    RankState::default().add_input(Tensor::ones(shape, (kind, Device::Cuda(rank))))
                }
            },
            |rank| {
                if rank == dst_rank {
                    create_box_group_operator!(GatherReceiver)
                } else {
                    create_box_group_operator!(GatherSender { dst_rank })
                }
            },
        );

        let expected_tensors = Vec::from_iter(
            std::iter::repeat_with(|| Tensor::ones(shape, (kind, Device::Cuda(0))))
                .take(CUDA_DEVICE_COUNT),
        );

        let output_tensors = std::mem::take(&mut rank_states[dst_rank].output_tensors);
        assert!(output_tensors.iter().zip(expected_tensors.iter()).all(
                |(output_tensor, expected_tensor)| {
                    output_tensor.equal(&expected_tensor.to_device(Device::Cuda(dst_rank)))
                },
            ),
            "Output tensors: {output_tensors:?} is different with expected tensors: {expected_tensors:?}"
        )
    }

    #[test]
    fn scatter() {
        struct ScatterSender;
        struct ScatterReceiver {
            src_rank: usize,
        }

        impl GroupOperator for ScatterSender {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.scatter_send(
                    rank_state.input_tensors.as_slice(),
                    &mut rank_state.output_tensors[0],
                )?;
                Ok(rank_state)
            }
        }
        impl GroupOperator for ScatterReceiver {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.scatter_receive(&mut rank_state.output_tensors[0], self.src_rank)?;
                Ok(rank_state)
            }
        }

        let src_rank = 0;
        let shape = [5, 5];
        let kind = Kind::Float;
        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                if rank == src_rank {
                    RankState::default()
                        .add_output(Tensor::zeros(shape, (kind, Device::Cuda(rank))))
                        .add_inputs::<CUDA_DEVICE_COUNT>(|| {
                            Tensor::ones(shape, (kind, Device::Cuda(rank)))
                        })
                } else {
                    RankState::default()
                        .add_output(Tensor::zeros(shape, (kind, Device::Cuda(rank))))
                }
            },
            |rank| {
                if rank == src_rank {
                    create_box_group_operator!(ScatterSender)
                } else {
                    create_box_group_operator!(ScatterReceiver { src_rank })
                }
            },
        );

        let expected_tensor = Tensor::ones(shape, (kind, Device::Cuda(0)));
        for (rank, output_tensor) in rank_states
            .into_iter()
            .map(|mut state| std::mem::take(&mut state.output_tensors[0]))
            .enumerate()
        {
            assert!(output_tensor.equal(&expected_tensor.to_device(Device::Cuda(rank))),
                "Output tensors: {output_tensor:?} is different with expected tensors: {expected_tensor:?}"
            )
        }
    }

    #[test]
    fn reduce_scatter() {
        struct ReduceScatterRank;
        impl GroupOperator for ReduceScatterRank {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.reduce_scatter(
                    rank_state.input_tensors.as_slice(),
                    &mut rank_state.output_tensors[0],
                    ReduceOp::SUM,
                )?;
                Ok(rank_state)
            }
        }

        let shape = [5, 5];
        let kind = Kind::Float;
        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                RankState::default()
                    .add_inputs::<CUDA_DEVICE_COUNT>(|| {
                        Tensor::ones(shape, (kind, Device::Cuda(rank)))
                    })
                    .add_output(Tensor::zeros(shape, (kind, Device::Cuda(rank))))
            },
            |_rank| create_box_group_operator!(ReduceScatterRank),
        );

        let expected_tensor = Tensor::ones(shape, (kind, Device::Cuda(0)));
        for (rank, output) in rank_states
            .into_iter()
            .map(|mut state| std::mem::take(&mut state.input_tensors[0]))
            .enumerate()
        {
            assert!(output.equal(&expected_tensor.to_device(Device::Cuda(rank))));
        }
    }

    #[test]
    fn reduce_scatter_into_tensor() {
        struct ReduceScatterRank;
        impl GroupOperator for ReduceScatterRank {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.reduce_scatter_into_tensor(
                    &rank_state.input_tensors[0],
                    &mut rank_state.output_tensors[0],
                    ReduceOp::SUM,
                )?;
                Ok(rank_state)
            }
        }

        let shape = [5, 5];
        let kind = Kind::Float;
        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                RankState::default()
                    .add_input(Tensor::ones(
                        [shape[0] * CUDA_DEVICE_COUNT as i64, shape[1]],
                        (kind, Device::Cuda(rank)),
                    ))
                    .add_output(Tensor::zeros(shape, (kind, Device::Cuda(rank))))
            },
            |_rank| create_box_group_operator!(ReduceScatterRank),
        );

        let expected_tensor = Tensor::ones(shape, (kind, Device::Cuda(0)));
        for (rank, output) in rank_states
            .into_iter()
            .map(|mut state| std::mem::take(&mut state.input_tensors[0]))
            .enumerate()
        {
            assert!(output.equal(&expected_tensor.to_device(Device::Cuda(rank))));
        }
    }

    #[test]
    fn all_to_all_single() {
        struct AllToAllRank {
            input_split_sizes: [i64; CUDA_DEVICE_COUNT],
            output_split_sizes: [i64; CUDA_DEVICE_COUNT],
        }
        impl GroupOperator for AllToAllRank {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.all_to_all_single(
                    &rank_state.input_tensors[0],
                    &mut rank_state.output_tensors[0],
                    self.input_split_sizes.as_slice(),
                    self.output_split_sizes.as_slice(),
                )?;
                Ok(rank_state)
            }
        }

        let shape = [5, 5];
        let kind = Kind::Float;
        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                RankState::default()
                    .add_input(Tensor::ones(
                        [shape[0] * CUDA_DEVICE_COUNT as i64, shape[1]],
                        (kind, Device::Cuda(rank)),
                    ))
                    .add_output(Tensor::zeros(
                        [shape[0] * CUDA_DEVICE_COUNT as i64, shape[1]],
                        (kind, Device::Cuda(rank)),
                    ))
            },
            |_rank| {
                create_box_group_operator!(AllToAllRank {
                    input_split_sizes: std::array::from_fn(|_| shape[0]),
                    output_split_sizes: std::array::from_fn(|_| shape[0]),
                })
            },
        );

        let expected_tensor =
            Tensor::ones([shape[0] * CUDA_DEVICE_COUNT as i64, shape[1]], (kind, Device::Cuda(0)));
        for (rank, output) in rank_states
            .into_iter()
            .map(|mut state| std::mem::take(&mut state.output_tensors[0]))
            .enumerate()
        {
            assert!(output.equal(&expected_tensor.to_device(Device::Cuda(rank))));
        }
    }

    #[test]
    fn all_to_all() {
        struct ReduceScatterRank;
        impl GroupOperator for ReduceScatterRank {
            fn handle(
                &mut self,
                mut process_group: ProcessGroupNCCL,
                mut rank_state: RankState,
            ) -> TchResult<RankState> {
                process_group.all_to_all(
                    rank_state.input_tensors.as_slice(),
                    rank_state.output_tensors.as_mut_slice(),
                )?;
                Ok(rank_state)
            }
        }

        let shape = [5, 5];
        let kind = Kind::Float;
        let rank_states = collective_comm_test::<_, _, CUDA_DEVICE_COUNT>(
            |rank| {
                RankState::default()
                    .add_inputs::<CUDA_DEVICE_COUNT>(|| {
                        Tensor::ones(shape, (kind, Device::Cuda(rank)))
                    })
                    .add_outputs::<CUDA_DEVICE_COUNT>(|| {
                        Tensor::zeros(shape, (kind, Device::Cuda(rank)))
                    })
            },
            |_rank| create_box_group_operator!(ReduceScatterRank),
        );

        let expected_tensors = Vec::from_iter(
            std::iter::repeat_with(|| Tensor::ones(shape, (kind, Device::Cuda(0))))
                .take(CUDA_DEVICE_COUNT),
        );
        for (rank, output_tensors) in
            rank_states.into_iter().map(|state| state.output_tensors).enumerate()
        {
            assert!(output_tensors.iter().zip(expected_tensors.iter()).all(
                |(output_tensor, expected_tensor)| {
                    output_tensor.equal(&expected_tensor.to_device(Device::Cuda(rank)))
                },
            ),
            "Output tensors: {output_tensors:?} is different with expected tensors: {expected_tensors:?}"
            )
        }
    }
}
