use std::{
    ffi::c_void,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[derive(Default)]
pub struct QueueLock {
    held: AtomicBool,
    pub calls: AtomicUsize,
}
pub struct Guard<'a>(&'a QueueLock);
impl QueueLock {
    pub fn lock(&self) -> Result<Guard<'_>, &'static str> {
        unsafe {
            lock((self as *const Self).cast_mut().cast());
        }
        Ok(Guard(self))
    }
}
impl Drop for Guard<'_> {
    fn drop(&mut self) {
        unsafe {
            unlock((self.0 as *const QueueLock).cast_mut().cast());
        }
    }
}
pub unsafe extern "C" fn lock(user: *mut c_void) {
    let lock = unsafe { &*user.cast::<QueueLock>() };
    while lock.held.swap(true, Ordering::Acquire) {
        std::thread::yield_now();
    }
    lock.calls.fetch_add(1, Ordering::Relaxed);
}
pub unsafe extern "C" fn unlock(user: *mut c_void) {
    unsafe { &*user.cast::<QueueLock>() }
        .held
        .store(false, Ordering::Release);
}
