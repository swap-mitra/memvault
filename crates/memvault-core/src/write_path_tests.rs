use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use uuid::Uuid;

use crate::crypto::Keyring;
use crate::index::{Indexes, KeywordIndex, VectorIndex};
use crate::ledger::{Ledger, LedgerError};
use crate::record::{ModelFingerprint, NamespaceId, Payload, SourceRef};
use crate::write_path::{supersede_fact, write_fact, SupersedeError, WriteError, WriteInput};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn fingerprint() -> ModelFingerprint {
    ModelFingerprint {
        name: "test-model".into(),
        dimensions: 4,
        revision_hash: [1u8; 32],
    }
}

struct Harness {
    dir: PathBuf,
    ledger: Ledger,
    keyring: Keyring,
    indexes: Indexes,
}

impl Drop for Harness {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn harness(tag: &str) -> Harness {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("memvault-write-path-test-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let ledger = Ledger::open(&dir.join("ledger.redb")).unwrap();
    let keyring = Keyring::open(&dir.join("keys.redb")).unwrap();
    let vector = VectorIndex::open_or_create(&dir.join("vectors.usearch"), &fingerprint()).unwrap();
    let keyword = KeywordIndex::open_or_create(&dir.join("keyword")).unwrap();

    Harness { dir, ledger, keyring, indexes: Indexes { vector, keyword } }
}

fn input(namespace: &str, content: &str, fact_id: Option<Uuid>) -> WriteInput {
    WriteInput {
        namespace: NamespaceId(namespace.into()),
        content: content.as_bytes().to_vec(),
        embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
        embedding_model: fingerprint(),
        valid_from: chrono::Utc::now(),
        valid_to: None,
        fact_id,
        keywords: vec![],
        pinned: false,
        source: SourceRef::default(),
    }
}

#[test]
fn write_fact_round_trips_through_ledger() {
    let mut h = harness("roundtrip");
    let fact_id = write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, input("default", "hello", None)).unwrap();

    let record = h.ledger.read(0).unwrap().unwrap();
    match record.payload {
        Payload::Assert(a) => {
            assert_eq!(a.fact_id, fact_id);
            let plaintext = h.keyring.decrypt(fact_id, &a.content).unwrap();
            assert_eq!(plaintext, b"hello");
        }
        other => panic!("expected an Assert, got {other:?}"),
    }
}

#[test]
fn rejects_oversized_content() {
    let mut h = harness("oversized");
    let mut too_big = input("default", "x", None);
    too_big.content = vec![0u8; crate::write_path::DEFAULT_MAX_CONTENT_BYTES + 1];

    let result = write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, too_big);
    assert!(matches!(result, Err(WriteError::ContentTooLarge { .. })));
    assert_eq!(h.ledger.head().unwrap(), 0, "rejected write must never reach the ledger");
}

#[test]
fn rejects_embedding_dimension_mismatch() {
    let mut h = harness("dim-mismatch");
    let mut bad = input("default", "x", None);
    bad.embedding = Some(vec![0.1, 0.2]); // fingerprint expects 4 dims

    let result = write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, bad);
    assert!(matches!(result, Err(WriteError::EmbeddingDimensionMismatch { expected: 4, actual: 2 })));
    assert_eq!(h.ledger.head().unwrap(), 0);
}

#[test]
fn rejects_incoherent_interval() {
    let mut h = harness("bad-interval");
    let mut bad = input("default", "x", None);
    bad.valid_to = Some(bad.valid_from - chrono::Duration::seconds(1));

    let result = write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, bad);
    assert!(matches!(result, Err(WriteError::InvalidInterval)));
    assert_eq!(h.ledger.head().unwrap(), 0);
}

/// The plan's acceptance test: write a fact, write again with the same
/// fact_id, assert a Supersede record closes the first interval and the
/// index reflects only the latest content.
#[test]
fn test_write_then_write_with_same_fact_id_supersedes() {
    let mut h = harness("supersede");
    let fact_id = write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, input("default", "versionone", None)).unwrap();

    let fact_id_again =
        write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, input("default", "revisiontwo", Some(fact_id))).unwrap();
    assert_eq!(fact_id, fact_id_again);

    assert_eq!(h.ledger.head().unwrap(), 3); // Assert, Supersede, Assert
    match h.ledger.read(1).unwrap().unwrap().payload {
        Payload::Supersede(s) => {
            assert_eq!(s.fact_id, fact_id);
            assert_eq!(s.target_seq, 0);
        }
        other => panic!("expected a Supersede record at seq 1, got {other:?}"),
    }
    assert_eq!(h.ledger.open_assert_seq(fact_id), Some(2));

    // The keyword index holds only the latest content: searching for the
    // superseded version's distinctive term finds nothing.
    let stale = h.indexes.keyword.search("versionone", 5).unwrap();
    assert!(stale.is_empty(), "stale version should not be searchable after supersession");
    let current = h.indexes.keyword.search("revisiontwo", 5).unwrap();
    assert_eq!(current[0].0, fact_id);

    h.ledger.verify().unwrap();
}

/// The plan's acceptance test: memory_supersede closes an open interval
/// without a new memory_write.
#[test]
fn test_supersede_without_rewrite_closes_interval() {
    let mut h = harness("supersede-no-rewrite");
    let fact_id = write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, input("default", "still true for now", None)).unwrap();

    let valid_to = chrono::Utc::now();
    supersede_fact(&h.ledger, &mut h.indexes, fact_id, valid_to, Some("no longer true".into())).unwrap();

    assert_eq!(h.ledger.open_assert_seq(fact_id), None);
    match h.ledger.read(1).unwrap().unwrap().payload {
        Payload::Supersede(s) => {
            assert_eq!(s.fact_id, fact_id);
            assert_eq!(s.target_seq, 0);
            assert_eq!(s.valid_to, valid_to);
            assert_eq!(s.reason.as_deref(), Some("no longer true"));
        }
        other => panic!("expected a Supersede record at seq 1, got {other:?}"),
    }

    // No new Assert was written -- just the original plus the Supersede.
    assert_eq!(h.ledger.head().unwrap(), 2);

    // Closed, so no longer searchable.
    let hits = h.indexes.keyword.search("still true for now", 5).unwrap();
    assert!(hits.is_empty(), "superseded fact should not be searchable");

    h.ledger.verify().unwrap();
}

#[test]
fn supersede_unknown_fact_id_is_not_found() {
    let mut h = harness("supersede-unknown");
    let result = supersede_fact(&h.ledger, &mut h.indexes, Uuid::from_u128(999), chrono::Utc::now(), None);
    assert!(matches!(result, Err(SupersedeError::NotFound)));
}

#[test]
fn different_fact_ids_do_not_supersede_each_other() {
    let mut h = harness("no-cross-supersede");
    write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, input("default", "a", None)).unwrap();
    write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, input("default", "b", None)).unwrap();

    assert_eq!(h.ledger.head().unwrap(), 2); // two Asserts, no Supersede
    for seq in 0..2 {
        match h.ledger.read(seq).unwrap().unwrap().payload {
            Payload::Assert(_) => {}
            other => panic!("expected an Assert at seq {seq}, got {other:?}"),
        }
    }
}

/// A `fact_id` is a global handle: `supersede` and `erase` both take one
/// with no namespace and close the fact wherever it lives. So a write that
/// supplies another namespace's `fact_id` is a caller mistake, not a
/// request to move the fact between namespaces, and it has to be refused
/// rather than honoured.
///
/// Honouring it used to file the closing `Supersede` under the *writer's*
/// namespace, where the original owner's `memory_as_of` -- which replays
/// only its own namespace's records -- could never see it. The fact ended
/// up closed as far as the fact view was concerned and still open as far
/// as its owner could tell: two answers to "is this fact live", neither
/// recoverable from the other.
#[test]
fn test_write_rejects_superseding_a_fact_from_another_namespace() {
    let mut h = harness("cross-namespace-supersede");

    let fact_id = write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, input("tenant-a", "alpha", None)).unwrap();
    let head_before = h.ledger.head().unwrap();

    let err = write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, input("tenant-b", "beta", Some(fact_id)))
        .expect_err("a write in tenant-b must not be allowed to supersede tenant-a's fact");
    assert!(
        matches!(err, WriteError::Ledger(LedgerError::CrossNamespaceSupersede { .. })),
        "expected a cross-namespace supersede rejection, got: {err}"
    );

    // Rejected before anything was appended, so tenant-a's fact is still
    // open, still at its original seq, and still owned by tenant-a.
    assert_eq!(h.ledger.head().unwrap(), head_before, "a rejected write must leave no record behind");

    let open_seq = h.ledger.open_assert_seq(fact_id).expect("tenant-a's fact must still be open");
    let record = h.ledger.read(open_seq).unwrap().unwrap();
    assert_eq!(record.header.namespace, NamespaceId("tenant-a".into()));
    match record.payload {
        Payload::Assert(a) => assert_eq!(a.fact_id, fact_id),
        other => panic!("expected an Assert, got {other:?}"),
    }

    h.ledger.verify().unwrap();
}
