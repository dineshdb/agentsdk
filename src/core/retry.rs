use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryAction {
    /// Retry the operation after the given delay.
    Retry(Duration),
    /// Do not retry, bubble up the error.
    DoNotRetry,
}
