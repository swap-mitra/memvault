//! The working set: session-scoped, ephemeral state (product doc §5,
//! §6.7). `DashMap`-backed and sharded end to end -- getting a session's
//! handle only ever touches that session's own shard of the outer map,
//! and every operation after that is on the session's own independent
//! inner map, so distinct sessions never contend with each other.
//!
//! Holds one thing so far: per-fact last-access timestamps within a
//! session, for `decay::apply_decay`'s "age from last access, retrieval
//! reinforces" requirement (product doc §6.4) -- not yet wired into the
//! read path (explain.rs still stands in with the fact's own valid_from,
//! per its own ponytail note), just built as its own correct, tested
//! unit here.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub Uuid);

pub struct SessionState {
    last_accessed: DashMap<Uuid, DateTime<Utc>>,
}

impl SessionState {
    fn new() -> Self {
        SessionState { last_accessed: DashMap::new() }
    }

    pub fn record_access(&self, fact_id: Uuid, at: DateTime<Utc>) {
        self.last_accessed.insert(fact_id, at);
    }

    pub fn last_accessed(&self, fact_id: Uuid) -> Option<DateTime<Utc>> {
        self.last_accessed.get(&fact_id).map(|e| *e)
    }
}

/// Cheaply cloned out of the `WorkingSet` and then operated on
/// independently -- callers don't hold any lock on the outer map while
/// using it.
pub type SessionHandle = Arc<SessionState>;

pub struct WorkingSet {
    sessions: DashMap<SessionId, SessionHandle>,
}

impl Default for WorkingSet {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkingSet {
    pub fn new() -> Self {
        WorkingSet { sessions: DashMap::new() }
    }

    pub fn get_or_create(&self, session: SessionId) -> SessionHandle {
        Arc::clone(&self.sessions.entry(session).or_insert_with(|| Arc::new(SessionState::new())))
    }

    pub fn drop_session(&self, session: SessionId) {
        self.sessions.remove(&session);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Instant;

    fn hammer(ws: &WorkingSet, session: SessionId, n: usize) {
        let handle = ws.get_or_create(session);
        let now = Utc::now();
        for i in 0..n {
            let fact_id = Uuid::from_u128(i as u128);
            handle.record_access(fact_id, now);
            let _ = handle.last_accessed(fact_id);
        }
    }

    #[test]
    fn get_or_create_returns_the_same_session_state() {
        let ws = WorkingSet::new();
        let session = SessionId(Uuid::from_u128(1));
        let fact_id = Uuid::from_u128(42);
        let now = Utc::now();

        ws.get_or_create(session).record_access(fact_id, now);

        assert_eq!(ws.get_or_create(session).last_accessed(fact_id), Some(now));
    }

    #[test]
    fn distinct_sessions_have_independent_state() {
        let ws = WorkingSet::new();
        let a = SessionId(Uuid::from_u128(1));
        let b = SessionId(Uuid::from_u128(2));
        let fact_id = Uuid::from_u128(42);

        ws.get_or_create(a).record_access(fact_id, Utc::now());

        assert!(ws.get_or_create(b).last_accessed(fact_id).is_none());
    }

    #[test]
    fn drop_session_discards_its_state() {
        let ws = WorkingSet::new();
        let session = SessionId(Uuid::from_u128(1));
        let fact_id = Uuid::from_u128(42);
        ws.get_or_create(session).record_access(fact_id, Utc::now());

        ws.drop_session(session);

        // A fresh handle for the same id starts empty again.
        assert!(ws.get_or_create(session).last_accessed(fact_id).is_none());
    }

    /// Acceptance test: two threads hammering distinct sessions
    /// concurrently finish in roughly the time one thread alone takes for
    /// the same amount of work -- not double that, which is what a shared
    /// exclusive lock across sessions would cost.
    #[test]
    fn test_concurrent_sessions_dont_block() {
        const N: usize = 20_000;
        let ws = Arc::new(WorkingSet::new());

        let baseline_start = Instant::now();
        hammer(&ws, SessionId(Uuid::from_u128(1000)), N);
        let baseline = baseline_start.elapsed();

        let concurrent_start = Instant::now();
        let t1 = {
            let ws = Arc::clone(&ws);
            thread::spawn(move || hammer(&ws, SessionId(Uuid::from_u128(1)), N))
        };
        let t2 = {
            let ws = Arc::clone(&ws);
            thread::spawn(move || hammer(&ws, SessionId(Uuid::from_u128(2)), N))
        };
        t1.join().unwrap();
        t2.join().unwrap();
        let concurrent = concurrent_start.elapsed();

        // Generous bound to avoid flaking on a busy/throttled runner while
        // still catching real cross-session contention (which would push
        // this toward ~2x baseline).
        assert!(
            concurrent < baseline * 17 / 10,
            "concurrent (2 threads, distinct sessions) took {concurrent:?}, \
             baseline (1 thread, same per-thread workload) took {baseline:?} \
             -- looks like sessions are blocking each other"
        );
    }
}
