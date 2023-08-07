use autocxx::prelude::*;
pub use ffi::*;

include_cpp! {
    #include "cxx_wrapper_utils.h"
    safety!(unsafe_ffi)
    generate!("new_target")
    generate!("clone_target")
}

unsafe impl Send for IntrusivePtrContainer {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::JoinHandle;
    #[test]
    fn auto_cxx_utils() {
        let mut target = ffi::new_target().within_unique_ptr();
        let cloned_target = ffi::clone_target(target.pin_mut()).within_unique_ptr();

        let handlers: Vec<_> = (0..5)
            .map(|_| {
                let mut cloned_target = ffi::clone_target(target.pin_mut()).within_unique_ptr();
                std::thread::spawn(move || {
                    clone_target(cloned_target.pin_mut());
                })
            })
            .collect();
        assert!(handlers.into_iter().map(JoinHandle::join).all(|result| result.is_ok()))
    }
}
