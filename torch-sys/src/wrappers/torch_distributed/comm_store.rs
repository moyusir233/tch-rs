pub use crate::wrappers::autocxx_wrappers::torch_distributed::comm_store::*;
use autocxx::prelude::*;
use cxx::UniquePtr;
use std::ops::Deref;
use std::pin::Pin;
use std::time::Duration;

pub trait Store {
    fn set_timeout(self: Pin<&mut Self>, timeout: Duration);
}

pub trait NestPrefixStore<S: Store + cxx::memory::UniquePtrTarget> {
    fn nest_store(prefix: String, store: UniquePtr<S>) -> UniquePtr<ArcPrefixStore>;
}

impl Default for MyTCPStoreOptions {
    fn default() -> Self {
        moveit! {
            let default_opts=Self::new();
        }
        Self { ..*default_opts.deref() }
    }
}

impl Clone for MyTCPStoreOptions {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl Store for ArcTCPStore {
    fn set_timeout(self: Pin<&mut Self>, timeout: Duration) {
        self.setTimeout((timeout.as_millis() as i64).into());
    }
}

impl Store for ArcPrefixStore {
    fn set_timeout(self: Pin<&mut Self>, timeout: Duration) {
        self.setTimeout((timeout.as_millis() as i64).into());
    }
}

impl NestPrefixStore<ArcPrefixStore> for ArcPrefixStore {
    fn nest_store(prefix: String, store: UniquePtr<ArcPrefixStore>) -> UniquePtr<ArcPrefixStore> {
        ArcPrefixStore::new2(prefix, store).within_unique_ptr()
    }
}

impl NestPrefixStore<ArcTCPStore> for ArcPrefixStore {
    fn nest_store(prefix: String, store: UniquePtr<ArcTCPStore>) -> UniquePtr<ArcPrefixStore> {
        ArcPrefixStore::new1(prefix, store).within_unique_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wrappers::torch_distributed::comm_store::ToCppString;
    use std::sync::Arc;

    #[test]
    fn store() {
        fn create_prefix_store(host: &str, opts: &MyTCPStoreOptions) -> UniquePtr<ArcPrefixStore> {
            let mut tcp_store = ArcTCPStore::new(host, opts).within_unique_ptr();
            tcp_store.pin_mut().set_timeout(Duration::from_secs(100));

            let prefix_store = ArcPrefixStore::nest_store("prefix1".into(), tcp_store);
            let mut prefix_store = ArcPrefixStore::nest_store("prefix2".into(), prefix_store);
            prefix_store.pin_mut().set_timeout(Duration::from_secs(100));

            prefix_store
        }

        let barrier = Arc::new(std::sync::Barrier::new(2));
        let host = "localhost";
        let key = "store_key";

        std::thread::spawn({
            let barrier = barrier.clone();
            move || {
                let tcp_store_opts = MyTCPStoreOptions {
                    port: 8080,
                    isServer: false,
                    numWorkers: 2,
                    timeout: 10000,
                    waitWorkers: true,
                    multiTenant: false,
                };
                let mut prefix_store = create_prefix_store(host, &tcp_store_opts);
                prefix_store.pin_mut().add(&key.into_cpp(), 1);
                barrier.wait();
                assert_eq!(prefix_store.pin_mut().add(&key.into_cpp(), 0), 2);
            }
        });

        let tcp_store_opts = MyTCPStoreOptions {
            port: 8080,
            isServer: true,
            numWorkers: 2,
            waitWorkers: true,
            timeout: 10000,
            multiTenant: true,
        };
        let mut prefix_store = create_prefix_store(&host, &tcp_store_opts);
        prefix_store.pin_mut().add(&key.into_cpp(), 1);
        barrier.wait();
        assert_eq!(prefix_store.pin_mut().add(&key.into_cpp(), 0), 2);
    }
}
