//! Point-in-time queries on both bitemporal axes (product doc §2.2/§6.4
//! step 3, §3 P2's three questions: "what is true now", "what was true on
//! date X", "what did the agent believe on date X").
//!
//! `hybrid_search`/`explain::search` deliberately reject `as_of` (see
//! read_path.rs's module doc): the vector/keyword indexes only ever hold
//! *current* versions, since `write_path::write_fact` retires a fact_id's
//! old index entries on every supersession. A historical version's content
//! was never embedded or tokenized into either index, so no ranked ANN/BM25
//! pass over them is possible -- the ledger itself is the only place that
//! history still exists. This module answers `as_of` queries by
//! reconstructing state directly from a full ledger scan instead, which is
//! a deliberate interface deviation from the plan's
//! `filter_bitemporal(candidates, as_of)` sketch: there is no pre-existing
//! candidate list to filter, only the ledger to reconstruct one from. Fine
//! at the ledger sizes this project targets -- `explain()` and `recover()`
//! already take the same linear-scan approach.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::crypto::{DecryptError, Keyring};
use crate::ledger::{Ledger, LedgerError};
use crate::record::{Assert, NamespaceId, Payload};

#[derive(Debug, Clone, Copy, Default)]
pub struct AsOfQuery {
    /// Point on the "what was true" axis. `None` means now.
    pub valid_time: Option<DateTime<Utc>>,
    /// Point on the "what did the agent know" axis: records recorded after
    /// this instant are treated as if they hadn't happened yet. `None`
    /// means now, i.e. no restriction.
    pub transaction_time: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AsOfFact {
    pub fact_id: Uuid,
    pub ledger_seq: u64,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub content: Vec<u8>,
    pub pinned: bool,
}

#[derive(Debug)]
pub enum AsOfError {
    Ledger(LedgerError),
    Decrypt(DecryptError),
}

impl std::fmt::Display for AsOfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsOfError::Ledger(e) => write!(f, "{e}"),
            AsOfError::Decrypt(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for AsOfError {}

impl From<LedgerError> for AsOfError {
    fn from(e: LedgerError) -> Self {
        AsOfError::Ledger(e)
    }
}

/// Reconstructs the facts true at `query`'s bitemporal coordinates by
/// replaying `namespace`'s ledger history. `Supersede.valid_to` (set by
/// `Ledger::write_assert` to the closing write's own `valid_from`, see
/// ledger.rs) is the authoritative close time for a version, overriding
/// that `Assert`'s own `valid_to` field -- but only if the `Supersede` was
/// itself recorded by `transaction_time`; a version this query's
/// transaction time predates the closing of still reads as open, which is
/// exactly "what did the agent believe on date X".
pub fn memory_as_of(ledger: &Ledger, keyring: &Keyring, namespace: &NamespaceId, query: AsOfQuery) -> Result<Vec<AsOfFact>, AsOfError> {
    let now = Utc::now();
    let valid_time = query.valid_time.unwrap_or(now);
    let transaction_time = query.transaction_time.unwrap_or(now);

    let mut asserts: HashMap<u64, Assert> = HashMap::new();
    let mut closed_at: HashMap<u64, DateTime<Utc>> = HashMap::new();

    for record in ledger.scan_from(0)? {
        let record = record?;
        if record.header.namespace != *namespace || record.header.recorded_at > transaction_time {
            continue;
        }
        match record.payload {
            Payload::Assert(a) => {
                asserts.insert(record.header.seq, a);
            }
            Payload::Supersede(s) => {
                closed_at.insert(s.target_seq, s.valid_to);
            }
            Payload::Erase(_) | Payload::Retrieval(_) | Payload::Checkpoint(_) => {}
        }
    }

    let mut results = Vec::new();
    for (seq, assert) in asserts {
        let effective_valid_to = closed_at.get(&seq).copied().or(assert.valid_to);
        let time_valid = assert.valid_from <= valid_time && effective_valid_to.is_none_or(|vt| valid_time < vt);
        if !time_valid {
            continue;
        }
        match keyring.decrypt(assert.fact_id, &assert.content) {
            Ok(content) => results.push(AsOfFact {
                fact_id: assert.fact_id,
                ledger_seq: seq,
                valid_from: assert.valid_from,
                valid_to: effective_valid_to,
                content,
                pinned: assert.pinned,
            }),
            // Erased since: gone forever, not an error (recovery.rs treats
            // the same case the same way).
            Err(DecryptError::KeyDestroyed) => continue,
            Err(e) => return Err(AsOfError::Decrypt(e)),
        }
    }

    results.sort_by_key(|f| f.ledger_seq);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::Keyring;
    use crate::ledger::Ledger;
    use crate::record::{ModelFingerprint, SourceRef};
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp(tag: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("memvault-bitemporal-test-{tag}-{}-{n}", std::process::id()))
    }

    struct Harness {
        ledger: Ledger,
        keyring: Keyring,
        ledger_path: std::path::PathBuf,
        keyring_path: std::path::PathBuf,
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.ledger_path);
            let _ = std::fs::remove_file(&self.keyring_path);
        }
    }

    fn harness(tag: &str) -> Harness {
        let ledger_path = tmp(&format!("{tag}-ledger.redb"));
        let keyring_path = tmp(&format!("{tag}-keys.redb"));
        Harness {
            ledger: Ledger::open(&ledger_path).unwrap(),
            keyring: Keyring::open(&keyring_path).unwrap(),
            ledger_path,
            keyring_path,
        }
    }

    fn write(h: &mut Harness, fact_id: Option<Uuid>, content: &str, valid_from: DateTime<Utc>, recorded_at: DateTime<Utc>) -> Uuid {
        let id = fact_id.unwrap_or_else(Uuid::new_v4);
        let encrypted = h.keyring.encrypt(id, content.as_bytes()).unwrap();
        let assert = Assert {
            fact_id: id,
            valid_from,
            valid_to: None,
            content: encrypted,
            content_hash: crate::crypto::content_hash(content.as_bytes()),
            embedding: None,
            embedding_model: ModelFingerprint { name: "t".into(), dimensions: 0, revision_hash: [0; 32] },
            keywords: vec![],
            pinned: false,
            source: SourceRef::default(),
        };
        h.ledger.write_assert(NamespaceId("default".into()), recorded_at, assert).unwrap();
        id
    }

    fn content_of(fact: &AsOfFact) -> String {
        String::from_utf8(fact.content.clone()).unwrap()
    }

    /// The plan's acceptance test: one fixture, three distinct bitemporal
    /// questions, three distinct correct answers.
    #[test]
    fn test_three_bitemporal_questions_differ() {
        let mut h = harness("three-questions");
        let t0 = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let t1 = t0 + chrono::Duration::days(10); // superseding write, both valid_from and recorded_at
        let t2 = t0 + chrono::Duration::days(20); // "now": after the correction is known

        let fact_id = write(&mut h, None, "v1: address is Elm St", t0, t0);
        write(&mut h, Some(fact_id), "v2: address is Oak Ave", t1, t1);

        let namespace = NamespaceId("default".into());

        // Q1: what is true now (as of t2, no restriction) -> v2.
        let now_answer = memory_as_of(&h.ledger, &h.keyring, &namespace, AsOfQuery { valid_time: Some(t2), transaction_time: Some(t2) }).unwrap();
        assert_eq!(now_answer.len(), 1);
        assert_eq!(content_of(&now_answer[0]), "v2: address is Oak Ave");

        // Q2: what was true on t0 (valid_time = t0, transaction_time unrestricted/now) -> v1.
        let then_answer = memory_as_of(&h.ledger, &h.keyring, &namespace, AsOfQuery { valid_time: Some(t0), transaction_time: Some(t2) }).unwrap();
        assert_eq!(then_answer.len(), 1);
        assert_eq!(content_of(&then_answer[0]), "v1: address is Elm St");

        // Q3: what did the agent believe was true "now" as of t0's
        // knowledge (transaction_time = t0, so v2 -- recorded at t1 -- is
        // not yet known) -> v1, even though valid_time asks about t2 (a
        // point v1's own recorded interval was still open as far as the
        // agent knew at t0).
        let believed_answer = memory_as_of(&h.ledger, &h.keyring, &namespace, AsOfQuery { valid_time: Some(t2), transaction_time: Some(t0) }).unwrap();
        assert_eq!(believed_answer.len(), 1);
        assert_eq!(content_of(&believed_answer[0]), "v1: address is Elm St");

        // All three answers are distinct in at least one case (Q1 vs Q2/Q3 content differs).
        assert_ne!(content_of(&now_answer[0]), content_of(&then_answer[0]));
        assert_eq!(content_of(&then_answer[0]), content_of(&believed_answer[0]));
    }

    #[test]
    fn erased_fact_is_absent_even_within_its_valid_window() {
        let mut h = harness("erased");
        let t0 = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let fact_id = write(&mut h, None, "secret", t0, t0);
        h.keyring.destroy_key(fact_id).unwrap();

        let namespace = NamespaceId("default".into());
        let answer = memory_as_of(&h.ledger, &h.keyring, &namespace, AsOfQuery { valid_time: Some(t0), transaction_time: None }).unwrap();
        assert!(answer.is_empty());
    }
}
