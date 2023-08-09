pub use crate::wrappers::autocxx_wrappers::torch_distributed::comm_store::*;
use autocxx::prelude::*;
use cxx::UniquePtr;
use std::ffi::CStr;
use std::fmt::{Debug, Formatter};
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

impl Debug for ArcTCPStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArcTCPStore").finish()
    }
}

/// 其底层使用了线程锁来保证操作正确,因此可以Send
unsafe impl Send for ArcTCPStore {}
/// 其底层使用了线程锁来保证操作正确,因此可以Send
unsafe impl Send for ArcPrefixStore {}

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

impl ArcPrefixStore {
    #[inline]
    pub fn add(self: Pin<&mut Self>, key: &CStr, value: i64) -> i64 {
        unsafe { self.ArcPrefixStore_add_(key.as_ptr(), value) }
    }
}

#[cfg(test)]
mod store {
    use super::*;
    use std::ffi::CString;
    use std::sync::Arc;

    #[test]
    fn init() {
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
        let key = CString::new("store_key").unwrap();
        let key = key.as_ref() as *const CStr;
        let key: &'static CStr = unsafe { &*key };

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
                prefix_store.pin_mut().add(key, 1);
                barrier.wait();
                assert_eq!(prefix_store.pin_mut().add(key, 0), 2);
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
        prefix_store.pin_mut().add(key, 1);
        barrier.wait();
        assert_eq!(prefix_store.pin_mut().add(key, 0), 2);
    }

    #[test]
    fn send_store() {
        let tcp_store_opts = MyTCPStoreOptions {
            port: 8080,
            isServer: true,
            numWorkers: 1,
            timeout: 10000,
            waitWorkers: true,
            multiTenant: true,
        };
        let tcp_store = ArcTCPStore::new("localhost", &tcp_store_opts).within_unique_ptr();
        std::thread::spawn(move || {
            println!("{:?}", tcp_store);
        })
        .join()
        .unwrap();

        let tcp_store = std::thread::spawn(move || {
            ArcTCPStore::new("localhost", &tcp_store_opts).within_unique_ptr()
        })
        .join()
        .unwrap();
        println!("{:?}", tcp_store);

        let mut prefix_store = ArcPrefixStore::nest_store("prefix1".into(), tcp_store);
        let key = CString::new("tmp").unwrap();
        for i in 0..9 {
            let value = prefix_store.pin_mut().add(&key, i % 2);
            println!("{i}:{value}")
        }
    }
}
