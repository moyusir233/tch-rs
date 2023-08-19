#ifndef TCH_TORCH_PROCESS_GROUP_NCCL_H
#define TCH_TORCH_PROCESS_GROUP_NCCL_H

#include <ATen/core/TensorBody.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>

#include <string>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <utility>
#include <vector>

#include "Work.hpp"
#include "torch_api.h"
#include "wrappers/torch_distributed/comm_store.h"
#include "wrappers/utils.h"

namespace c10d {

CREATE_CONTAINER_CLASS(_ArcWork, Work)

class ArcWork : public _ArcWork {
private:
  explicit ArcWork(const c10::intrusive_ptr<Work> &work)
      : _ArcWork(c10::intrusive_ptr<Work>(work)) {}

public:
  explicit ArcWork(c10::intrusive_ptr<Work> &&work)
      : _ArcWork(std::forward<c10::intrusive_ptr<Work>>(work)) {}

  [[nodiscard]] ArcWork clone_() const { return ArcWork(inner); }

  // Checks if request has completed. Non-blocking operation.
  bool is_completed() { return inner->isCompleted(); }

  // Returns if the work completed successfully.
  // If false, the exception function can be called to get details.
  [[nodiscard]] bool is_success() const { return inner->isSuccess(); }

  // Returns exception if isSuccess() returned false.
  [[nodiscard]] const char *exception() const {
    auto exception_ptr = inner->exception();
    const char *err_msg = nullptr;

    try {
      if (exception_ptr) {
        std::rethrow_exception(exception_ptr);
      }
    } catch (const std::exception &e) {
      err_msg = e.what();
    }

    return err_msg;
  }

  // Returns source rank if this objects represents a recv-from-any.
  [[nodiscard]] int source_rank() const { return inner->sourceRank(); }

  // Returns result tensors, if applicable.
  // If work is not supposed to have result, we return empty list.
  // # 注意返回的裸指针要进行回收
  std::vector<tensor> result() {
    auto tensors = inner->result();
    std::vector<tensor> result(tensors.size(), nullptr);

    for (int i = 0; i < tensors.size(); ++i) {
      result[i] = new at::Tensor(tensors[i]);
    }

    return result;
  }

  // Ensures that operations on the output tensors that are invoked
  // after this function returns are correctly sequenced after the
  // asynchronous completion of this work.
  //
  // For CUDA tensors, it inserts stream synchronization such that
  // the streams of the caller wait for completion of the
  // asynchronous operations on the destination tensors.
  //
  // For CPU tensors, it is currently a nop.
  //
  // This function should only be used if the caller polls for
  // completion through the `isCompleted` function, it has returned
  // true, and the `isSuccess` function also has returned true.
  //
  void synchronize() { inner->synchronize(); }

  // Waits until request completes. Blocking operation.
  // Throws if the work completed with an exception.
  // Returns false if the work is aborted.
  // Otherwise, it always returns true, indicating the work is completed.
  //
  // Functionally equivalent to:
  //
  //   while (!isCompleted()) { /* nop */ }
  //   auto success = isSuccess();
  //   if (!success) { std::rethrow_exception(exception()); }
  //   return success;
  //
  bool wait() {
    try {
      inner->wait();
    } catch (const std::exception &e) {
      return false;
    }

    return true;
  }

  void abort() { inner->abort(); }

  OpType retrieveOpType() { return inner->retrieveOpType(); }
};

struct ProcessGroupNCCLOptions {
  std::int64_t timeout = 300000;
  bool is_high_priority_stream = false;

  explicit operator c10::intrusive_ptr<ProcessGroupNCCL::Options>() const {
    auto opt = ProcessGroupNCCL::Options::create(is_high_priority_stream);
    opt->timeout = std::chrono::milliseconds(timeout);
    return opt;
  }
};

CREATE_CONTAINER_CLASS(_ArcProcessGroupNCCL, ProcessGroupNCCL)

CREATE_PTR_ARRAY(Tensors, tensor)
CREATE_PTR_ARRAY(I64List, std::int64_t)

class ArcProcessGroupNCCL : public _ArcProcessGroupNCCL {
private:
  // 用于实现clone的构造函数
  explicit ArcProcessGroupNCCL(
      const c10::intrusive_ptr<ProcessGroupNCCL> &process_group)
      : _ArcProcessGroupNCCL(
            c10::intrusive_ptr<ProcessGroupNCCL>(process_group)) {}

  // 用于重用vector
  template <class T> class ObjectPool {
  private:
    std::vector<T> pool;
    std::function<T()> create_object;
    std::uint64_t cursor = 0;

    T &get_object() {
      // 扩容
      if (cursor == pool.size()) {
        pool.emplace_back(create_object());
      }

      auto obj = pool.begin() + cursor;
      cursor += 1;
      return *obj;
    }

    void return_object() { cursor -= 1; }

  public:
    template <class T2 = T> class ObjectWrapper {
    private:
      ObjectPool<T2> &parent;

    public:
      T2 &inner;

      explicit ObjectWrapper(ObjectPool<T2> &parent_pool)
          : parent(parent_pool), inner(parent_pool.get_object()) {}

      operator T2 &() const { return inner; }

      ~ObjectWrapper() { parent.return_object(); }
    };

    // 模板特化,处理vec元素的清理
    template <> class ObjectWrapper<vector<at::Tensor>> {
    private:
      ObjectPool<vector<at::Tensor>> &parent;

    public:
      vector<at::Tensor> &inner;

      explicit ObjectWrapper(ObjectPool<vector<at::Tensor>> &parent_pool)
          : parent(parent_pool), inner(parent_pool.get_object()) {}

      operator vector<at::Tensor> &() const { return inner; }

      ~ObjectWrapper() {
        inner.clear();
        parent.return_object();
      }
    };

    template <> class ObjectWrapper<vector<vector<at::Tensor>>> {
    private:
      ObjectPool<vector<vector<at::Tensor>>> &parent;

    public:
      vector<vector<at::Tensor>> &inner;

      explicit ObjectWrapper(
          ObjectPool<vector<vector<at::Tensor>>> &parent_pool)
          : parent(parent_pool), inner(parent_pool.get_object()) {}

      operator vector<vector<at::Tensor>> &() const { return inner; }

      ~ObjectWrapper() {
        inner[0].clear();
        parent.return_object();
      }
    };

    explicit ObjectPool(std::function<T()> &&create_object_fn)
        : create_object(create_object_fn) {}

    ObjectWrapper<T> get_object_wrapper() { return ObjectWrapper<T>(*this); }
  };

  // IMPROVE:
  // 每次调用集合通信相关的api就需要创建新的vector,造成频繁的堆内存申请,如何改善?
  // 用于clone Tensor指针
  static ObjectPool<std::vector<at::Tensor>>::ObjectWrapper<
      std::vector<at::Tensor>>
  clone_tensor_prt(const tensor &tensor) {
    thread_local ObjectPool<std::vector<at::Tensor>> tensor_vec_pool(
        std::function<std::vector<at::Tensor>()>(
            []() { return std::vector<at::Tensor>(); }));

    auto obj = tensor_vec_pool.get_object_wrapper();

    if (tensor == nullptr) {
      return obj;
    }

    obj.inner.emplace_back(*tensor);

    return obj;
  }

  static ObjectPool<std::vector<std::vector<at::Tensor>>>::ObjectWrapper<
      std::vector<std::vector<at::Tensor>>>
  clone_tensor_prt(const Tensors &tensors) {
    thread_local ObjectPool<std::vector<std::vector<at::Tensor>>>
        tensor_vec_pool(std::function<std::vector<std::vector<at::Tensor>>()>(
            []() { return std::vector<std::vector<at::Tensor>>(); }));

    auto obj = tensor_vec_pool.get_object_wrapper();
    if (tensors.size == 0) {
      throw "TODO: 此时应该返回空的vector,而不能继续复用包含单个元素的vecotr";
    }

    if (obj.inner.empty()) {
      obj.inner.emplace_back();
    }

    obj.inner[0].reserve(tensors.size);
    for (const auto &i : c10::irange(tensors.size)) {
      obj.inner[0].emplace_back(at::Tensor(*(tensors.ptr[i])));
    }

    return obj;
  }

public:
  explicit ArcProcessGroupNCCL(const ArcPrefixStore &store, std::int32_t rank,
                               std::int32_t size,
                               ProcessGroupNCCLOptions options)
      : _ArcProcessGroupNCCL(c10::make_intrusive<ProcessGroupNCCL>(
            store.inner, rank, size,
            (c10::intrusive_ptr<ProcessGroupNCCL::Options>)options)) {}

  explicit ArcProcessGroupNCCL(const ArcTCPStore &store, std::int32_t rank,
                               std::int32_t size,
                               ProcessGroupNCCLOptions options)
      : _ArcProcessGroupNCCL(c10::make_intrusive<ProcessGroupNCCL>(
            store.inner, rank, size,
            (c10::intrusive_ptr<ProcessGroupNCCL::Options>)options)) {}

  [[nodiscard]] ArcProcessGroupNCCL clone_() const {
    return ArcProcessGroupNCCL(inner);
  }

  void set_sequence_number_for_group() { inner->setSequenceNumberForGroup(); }

  ArcWork broadcast_(tensor tensor, std::int64_t src_rank) {
    BroadcastOptions opts = {
        .rootRank = src_rank,
        .rootTensor = 0,
    };
    auto tensors = clone_tensor_prt(tensor);

    return ArcWork(inner->broadcast(tensors, opts));
  }

  ArcWork all_reduce_(tensor tensor, ReduceOp::RedOpType reduce_op) {
    auto all_reduce_opts = AllreduceOptions{
        .reduceOp = reduce_op,
    };
    auto tensors = clone_tensor_prt(tensor);

    return ArcWork(inner->allreduce(tensors, all_reduce_opts));
  }

  ArcWork reduce_(tensor tensor, std::int64_t dst_rank,
                  ReduceOp::RedOpType reduce_op) {
    auto reduce_opts = ReduceOptions{
        .reduceOp = reduce_op,
        .rootRank = dst_rank,
    };

    auto tensors = clone_tensor_prt(tensor);

    return ArcWork(inner->reduce(tensors, reduce_opts));
  }

  ArcWork all_gather_(tensor input_tensor, Tensors output_tensors) {
    auto input_tensors = clone_tensor_prt(input_tensor);

    auto output_tensors_vec = clone_tensor_prt(output_tensors);

    return ArcWork(inner->allgather(output_tensors_vec, input_tensors));
    ;
  }

  ArcWork all_gather_into_tensor_(tensor input_tensor, tensor output_tensor) {
    return ArcWork(inner->_allgather_base(*output_tensor, *input_tensor));
  }

  ArcWork gather_(tensor input_tensor, Tensors output_tensors,
                  std::int64_t dst_rank) {
    auto input_tensors = clone_tensor_prt(input_tensor);

    auto output_tensors_vec = clone_tensor_prt(output_tensors);

    GatherOptions opts = {.rootRank = dst_rank};

    return ArcWork(inner->gather(output_tensors_vec, input_tensors, opts));
  }

  ArcWork scatter_(Tensors input_tensors, tensor output_tensor,
                   std::int64_t src_rank) {
    auto input_tensors_vec = clone_tensor_prt(input_tensors);

    auto output_tensors = clone_tensor_prt(output_tensor);

    ScatterOptions opts = {.rootRank = src_rank};

    return ArcWork(inner->scatter(output_tensors, input_tensors_vec, opts));
  }

  ArcWork reduce_scatter_(Tensors input_tensors, tensor output_tensor,
                          ReduceOp::RedOpType reduce_op) {
    auto input_tensors_vec = clone_tensor_prt(input_tensors);

    auto output_tensors = clone_tensor_prt(output_tensor);

    ReduceScatterOptions opts = {.reduceOp = reduce_op};

    return ArcWork(
        inner->reduce_scatter(output_tensors, input_tensors_vec, opts));
  }

  ArcWork reduce_scatter_tensor_(tensor input_tensor, tensor output_tensor,
                                 ReduceOp::RedOpType reduce_op) {
    ReduceScatterOptions opts = {.reduceOp = reduce_op};

    return ArcWork(
        inner->_reduce_scatter_base(*output_tensor, *input_tensor, opts));
  }

  ArcWork all_to_all_single_(tensor input_tensor, tensor output_tensor,
                             I64List output_split_sizes,
                             I64List input_split_sizes) {
    auto input_split_vec = std::vector<std::int64_t>(
        input_split_sizes.ptr, input_split_sizes.ptr + input_split_sizes.size);

    auto output_split_vec = std::vector<std::int64_t>(
        output_split_sizes.ptr,
        output_split_sizes.ptr + output_split_sizes.size);

    return ArcWork(inner->alltoall_base(*output_tensor, *input_tensor,
                                        output_split_vec, input_split_vec));
  }

  ArcWork alltoall_(Tensors input_tensors, Tensors output_tensors) {
    auto input_tensors_vec = clone_tensor_prt(input_tensors).inner;

    auto output_tensors_vec = clone_tensor_prt(output_tensors).inner;

    return ArcWork(
        inner->alltoall(output_tensors_vec.front(), input_tensors_vec.front()));
  }

  ArcWork barrier_(const I64List device_ids) {
    std::vector<std::int64_t> vec(device_ids.ptr,
                                  device_ids.ptr + device_ids.size);

    BarrierOptions opts = {.device_ids = vec};
    return ArcWork(inner->barrier(opts));
  }

  ArcWork send_(tensor tensor, std::int64_t dst_rank) {
    auto input_tensors = clone_tensor_prt(tensor);
    return ArcWork(inner->send(input_tensors, dst_rank, 0));
  }

  ArcWork receive_(tensor tensor, std::int64_t src_rank) {
    auto output_tensors = clone_tensor_prt(tensor);
    return ArcWork(inner->recv(output_tensors, src_rank, 0));
  }
};
} // namespace c10d

#endif // TCH_TORCH_PROCESS_GROUP_NCCL_H
