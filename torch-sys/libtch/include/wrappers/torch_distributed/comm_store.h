#ifndef TCH_TORCH_COMM_STORE_H
#define TCH_TORCH_COMM_STORE_H

#include "wrappers/utils.h"
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

namespace c10d {
struct MyTCPStoreOptions {
  std::uint16_t port = TCPStoreOptions::kDefaultPort;
  bool isServer = false;
  std::size_t numWorkers = 1;
  bool waitWorkers = true;
  std::int64_t timeout = 300000;

  // A boolean value indicating whether multiple store instances can be
  // initialized with the same host:port pair.
  bool multiTenant = false;

  explicit operator TCPStoreOptions() const {
    return {.port = port,
            .isServer = isServer,
            .numWorkers = numWorkers,
            .waitWorkers = waitWorkers,
            .timeout = std::chrono::milliseconds(timeout),
            .multiTenant = multiTenant};
  }
};

CREATE_CONTAINER_CLASS(_ArcTCPStore, TCPStore)

class ArcTCPStore : public _ArcTCPStore {
public:
  explicit ArcTCPStore(std::string host, const MyTCPStoreOptions &opts)
      : _ArcTCPStore(c10::make_intrusive<TCPStore>(std::move(host),
                                                   (TCPStoreOptions)opts)) {}

  void set_timeout_(std::int64_t milliseconds) {
    inner->setTimeout(std::chrono::milliseconds(milliseconds));
  }
};

CREATE_CONTAINER_CLASS(_ArcPrefixStore, PrefixStore)

class ArcPrefixStore : public _ArcPrefixStore {
private:
  // 用于实现clone的构造函数,利用了智能指针引用的拷贝构造函数
  explicit ArcPrefixStore(const c10::intrusive_ptr<PrefixStore> &store)
      : _ArcPrefixStore(c10::intrusive_ptr<PrefixStore>(store)) {}

public:
  explicit ArcPrefixStore(std::string prefix, ArcTCPStore store)
      : _ArcPrefixStore(c10::make_intrusive<PrefixStore>(
            std::move(prefix), std::move(store.inner))) {}

  explicit ArcPrefixStore(std::string prefix, ArcPrefixStore store)
      : _ArcPrefixStore(c10::make_intrusive<PrefixStore>(
            std::move(prefix), std::move(store.inner))) {}

  ArcPrefixStore clone_() const { return ArcPrefixStore(inner); }

  void set_timeout_(std::int64_t milliseconds) {
    inner->setTimeout(std::chrono::milliseconds(milliseconds));
  }

  // 注意add操作内部利用了线程锁mutex
  int64_t add_(const char *key, int64_t value) {
    return inner->add(std::string(key), value);
  }
};

} // namespace c10d

#endif // TCH_TORCH_COMM_STORE_H
