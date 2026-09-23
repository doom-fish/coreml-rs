use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static TIMEOUT_NANOS: AtomicU64 = AtomicU64::new(0);

pub fn set_blocking_timeout(timeout: Option<Duration>) {
    let nanos = timeout.map_or(0, |timeout| {
        u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX).max(1)
    });
    TIMEOUT_NANOS.store(nanos, Ordering::Relaxed);
}

#[must_use]
pub fn blocking_timeout() -> Option<Duration> {
    match TIMEOUT_NANOS.load(Ordering::Relaxed) {
        0 => None,
        nanos => Some(Duration::from_nanos(nanos)),
    }
}

pub(crate) fn timeout_seconds() -> f64 {
    blocking_timeout().map_or(0.0, |timeout| timeout.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{blocking_timeout, set_blocking_timeout, timeout_seconds};

    #[test]
    fn blocking_timeout_round_trips_and_defaults_to_waiting() {
        assert_eq!(blocking_timeout(), None);
        assert!(timeout_seconds() <= 0.0);
        set_blocking_timeout(Some(Duration::from_millis(1500)));
        assert_eq!(blocking_timeout(), Some(Duration::from_millis(1500)));
        assert!((timeout_seconds() - 1.5).abs() < 1e-9);
        set_blocking_timeout(Some(Duration::ZERO));
        assert_eq!(blocking_timeout(), Some(Duration::from_nanos(1)));
        assert!(timeout_seconds() > 0.0);
        set_blocking_timeout(Some(Duration::MAX));
        assert_eq!(blocking_timeout(), Some(Duration::from_nanos(u64::MAX)));
        set_blocking_timeout(None);
        assert_eq!(blocking_timeout(), None);
    }
}
