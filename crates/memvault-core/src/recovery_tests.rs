use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use uuid::Uuid;

use crate::crypto::Keyring;
use crate::index::{Indexes, KeywordIndex, VectorIndex};
use crate::ledger::Ledger;
use crate::record::{ModelFingerprint, NamespaceId, SourceRef};
use crate::recovery::{recover, IndexKind, RecoveryConfig};
use crate::write_path::{write_fact, WriteInput};

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
    let dir = std::env::temp_dir().join(format!("memvault-recovery-test-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let ledger = Ledger::open(&dir.join("ledger.redb")).unwrap();
    let keyring = Keyring::open(&dir.join("keys.redb")).unwrap();
    let vector = VectorIndex::open_or_create(&dir.join("vectors.usearch"), &fingerprint()).unwrap();
    let keyword = KeywordIndex::open_or_create(&dir.join("keyword")).unwrap();
    Harness { dir, ledger, keyring, indexes: Indexes { vector, keyword } }
}

/// Distinct, reproducible embeddings per fact so search results are
/// comparable between two independently-built index sets.
fn vector_for(seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..4)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn write_n_facts(h: &mut Harness, n: u64) {
    for i in 0..n {
        write_fact(
            &h.ledger,
            &mut h.indexes,
            &mut h.keyring,
            WriteInput {
                namespace: NamespaceId("default".into()),
                content: format!("fact number {i}").into_bytes(),
                embedding: Some(vector_for(i)),
                embedding_model: fingerprint(),
                valid_from: chrono::Utc::now(),
                valid_to: None,
                fact_id: None,
                keywords: vec![],
                pinned: false,
                source: SourceRef::default(),
            },
        )
        .unwrap();
    }
}

#[test]
fn recover_is_a_no_op_when_indexes_are_already_current() {
    let mut h = harness("no-op");
    write_n_facts(&mut h, 5);

    let report = recover(&h.ledger, &mut h.indexes, &h.keyring, &fingerprint(), RecoveryConfig::default()).unwrap();

    assert_eq!(report.replayed_from, None);
    assert!(report.rebuilt.is_empty());
}

#[test]
fn recover_rebuilds_on_fingerprint_mismatch() {
    let mut h = harness("fingerprint-mismatch");
    write_n_facts(&mut h, 3);

    let different = ModelFingerprint { name: "other-model".into(), dimensions: 4, revision_hash: [9u8; 32] };
    let report = recover(&h.ledger, &mut h.indexes, &h.keyring, &different, RecoveryConfig::default()).unwrap();

    assert_eq!(report.rebuilt, vec![IndexKind::Vector]);
    assert_eq!(h.indexes.vector.fingerprint(), &different);
    assert_eq!(h.indexes.vector.watermark(), h.ledger.head().unwrap());
}

/// The plan's acceptance test: simulate a crash between commit and
/// post-commit index insert by rolling the index watermarks back to an
/// earlier point than the ledger head (the state a real crash there would
/// leave -- write_fact's remove-then-insert makes replay idempotent
/// regardless of whether the target already happened to have the data).
/// recover() must leave both indexes at exactly ledger.head(), and a
/// subsequent search must match a from-scratch rebuild exactly.
#[test]
fn test_recovery_after_simulated_crash() {
    let mut h = harness("simulated-crash");
    write_n_facts(&mut h, 10);
    let head = h.ledger.head().unwrap();

    let crash_point = head - 3;
    h.indexes.vector.set_watermark(crash_point).unwrap();
    h.indexes.keyword.set_watermark(crash_point).unwrap();

    let report = recover(&h.ledger, &mut h.indexes, &h.keyring, &fingerprint(), RecoveryConfig { verify_chain: true }).unwrap();

    assert_eq!(report.replayed_from, Some(crash_point));
    assert!(report.rebuilt.is_empty());
    assert!(report.verified);
    assert_eq!(h.indexes.vector.watermark(), head);
    assert_eq!(h.indexes.keyword.watermark(), head);

    // Compare against a fresh, from-scratch rebuild (a brand new index
    // pair, watermark 0, replayed via the exact same recover() call).
    let dir2 = std::env::temp_dir().join(format!(
        "memvault-recovery-test-simulated-crash-fresh-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir2).unwrap();
    let mut fresh_indexes = Indexes {
        vector: VectorIndex::open_or_create(&dir2.join("vectors.usearch"), &fingerprint()).unwrap(),
        keyword: KeywordIndex::open_or_create(&dir2.join("keyword")).unwrap(),
    };
    recover(&h.ledger, &mut fresh_indexes, &h.keyring, &fingerprint(), RecoveryConfig::default()).unwrap();

    for i in 0..10u64 {
        let query_vec = vector_for(i);
        let recovered = h.indexes.vector.search(&query_vec, 10).unwrap();
        let rebuilt = fresh_indexes.vector.search(&query_vec, 10).unwrap();
        let recovered_ids: Vec<Uuid> = recovered.iter().map(|(id, _, _)| *id).collect();
        let rebuilt_ids: Vec<Uuid> = rebuilt.iter().map(|(id, _, _)| *id).collect();
        assert_eq!(recovered_ids, rebuilt_ids, "search results diverge from a from-scratch rebuild for seed {i}");
    }

    let _ = std::fs::remove_dir_all(&dir2);
}
