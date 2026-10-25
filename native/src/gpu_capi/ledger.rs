use std::collections::VecDeque;

// CPU submission numbering and retirement, deliberately independent of a GPU.
// External completion may lag the internal fence; retire only a contiguous
// prefix for which both completion conditions hold.
pub(super) struct Ledger<T> {
    pub issued: u64,
    pub completed: u64,
    pub pending: VecDeque<(u64, T)>,
    pub capacity: usize,
}
impl<T> Ledger<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            issued: 0,
            completed: 0,
            pending: VecDeque::new(),
            capacity,
        }
    }
    pub fn next(&self) -> Option<u64> {
        if self.pending.len() == self.capacity {
            None
        } else {
            self.issued.checked_add(1)
        }
    }
    pub fn accept(&mut self, value: T) -> Option<u64> {
        let sequence = self.next()?;
        self.issued = sequence;
        self.pending.push_back((sequence, value));
        Some(sequence)
    }
    pub fn retire(&mut self, internal: u64, ready: impl Fn(&T) -> bool) {
        while self
            .pending
            .front()
            .is_some_and(|(seq, value)| *seq <= internal && ready(value))
        {
            self.completed = self.pending.pop_front().unwrap().0;
        }
    }
    // Native timeline queries can fail. Inspect the eligible prefix before
    // releasing any payload, so a later query error retains the entire batch.
    pub fn try_retire<E>(
        &mut self,
        internal: u64,
        ready: impl Fn(&T) -> Result<bool, E>,
    ) -> Result<(), E> {
        let mut through = 0;
        for (sequence, value) in &self.pending {
            if *sequence > internal || !ready(value)? {
                break;
            }
            through = *sequence;
        }
        self.retire(through, |_| true);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn external_signal_is_required_even_after_internal_completion() {
        let mut ledger = Ledger::new(2);
        assert_eq!(ledger.accept(10), Some(1));
        assert_eq!(ledger.accept(20), Some(2));
        ledger.retire(2, |value| *value <= 9);
        assert_eq!(ledger.completed, 0);
        assert_eq!(ledger.next(), None);
        ledger.retire(2, |value| *value <= 10);
        assert_eq!(ledger.completed, 1);
        assert_eq!(ledger.accept(30), Some(3));
        ledger.retire(2, |_| true);
        assert_eq!(ledger.completed, 2);
        assert_eq!(ledger.pending.len(), 1);
    }
    #[test]
    fn retirement_releases_payloads_and_numbers_never_wrap() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        struct Payload(Arc<AtomicUsize>);
        impl Drop for Payload {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let mut ledger = Ledger::new(1);
        ledger.accept(Payload(drops.clone())).unwrap();
        ledger.retire(0, |_| true);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        ledger.retire(1, |_| true);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        ledger.issued = u64::MAX;
        assert_eq!(ledger.next(), None);
    }
    #[test]
    fn native_query_error_keeps_previously_ready_resources() {
        let mut ledger = Ledger::new(3);
        ledger.accept(1);
        ledger.accept(2);
        assert_eq!(
            ledger.try_retire(2, |value| if *value == 2 {
                Err("device removed")
            } else {
                Ok(true)
            }),
            Err("device removed")
        );
        assert_eq!(ledger.completed, 0);
        assert_eq!(ledger.pending.len(), 2);
    }
}
