#ifndef TCH_TORCH_PROCESS_GROUP_NCCL_H
#define TCH_TORCH_PROCESS_GROUP_NCCL_H

#include <ATen/core/TensorBody.h>
#include <c10/util/intrusive_ptr.h>
#include <memory>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <utility>

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

  ArcWork clone_() const { return ArcWork(inner); }

  // Checks if request has completed. Non-blocking operation.
  bool is_completed() { return inner->isCompleted(); }

  // Returns if the work completed successfully.
  // If false, the exception function can be called to get details.
  bool is_success() const { return inner->isSuccess(); }

  // Returns exception if isSuccess() returned false.
  std::string exception() const {
    auto exception_ptr = inner->exception();
    std::string err_msg;

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
  int source_rank() const { return inner->sourceRank(); }

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
  bool wait() { return inner->wait(); }

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

class ArcProcessGroupNCCL : public _ArcProcessGroupNCCL {
private:
  // 用于实现clone的构造函数
  explicit ArcProcessGroupNCCL(
      const c10::intrusive_ptr<ProcessGroupNCCL> &process_group)
      : _ArcProcessGroupNCCL(
            c10::intrusive_ptr<ProcessGroupNCCL>(process_group)) {}

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

  ArcProcessGroupNCCL clone_() const { return ArcProcessGroupNCCL(inner); }

  void set_sequence_number_for_group() { inner->setSequenceNumberForGroup(); }

  ArcWork broadcast_(tensor tensor, std::int32_t src) {
    BroadcastOptions opts = {.rootRank = src, .rootTensor = 0};
    // 因为rust侧的broadcast是通过传入张量引用,进行就地操作的语义,因此这里不能移动,
    // 而是显式地创建拷贝
    vector<at::Tensor> tensors = {at::Tensor(*tensor)};

    return ArcWork(inner->broadcast(tensors, opts));
  }
};

} // namespace c10d

#endif // TCH_TORCH_PROCESS_GROUP_NCCL_H
