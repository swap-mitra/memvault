use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{Duration, Utc};
use uuid::Uuid;

use crate::crypto::Keyring;
use crate::explain::search;
use crate::index::{Indexes, KeywordIndex, VectorIndex};
use crate::ledger::Ledger;
use crate::read_path::Query;
use crate::record::{ModelFingerprint, NamespaceId, Outcome, SourceRef};
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
    let dir = std::env::temp_dir().join(format!("memvault-explain-test-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let ledger = Ledger::open(&dir.join("ledger.redb")).unwrap();
    let keyring = Keyring::open(&dir.join("keys.redb")).unwrap();
    let vector = VectorIndex::open_or_create(&dir.join("vectors.usearch"), &fingerprint()).unwrap();
    let keyword = KeywordIndex::open_or_create(&dir.join("keyword")).unwrap();
    Harness { dir, ledger, keyring, indexes: Indexes { vector, keyword } }
}

fn write(h: &mut Harness, content: &str, embedding: Vec<f32>, valid_from: chrono::DateTime<Utc>, valid_to: Option<chrono::DateTime<Utc>>) -> Uuid {
    write_fact(
        &h.ledger,
        &mut h.indexes,
        &mut h.keyring,
        WriteInput {
            namespace: NamespaceId("default".into()),
            content: content.as_bytes().to_vec(),
            embedding: Some(embedding),
            embedding_model: fingerprint(),
            valid_from,
            valid_to,
            fact_id: None,
            keywords: vec![],
            pinned: false,
            source: SourceRef::default(),
        },
    )
    .unwrap()
}

/// The plan's acceptance test: construct a query where at least one
/// candidate is cut by budget, one by k, one by the time filter; assert
/// all three appear in the Explanation list with correct outcome values,
/// not just the injected one.
#[test]
fn test_explanation_includes_all_outcomes() {
    let mut h = harness("all-outcomes");
    let now = Utc::now();

    // Small content -> fits the tight token budget below.
    let injected = write(&mut h, "a", vec![1.0, 0.0, 0.0, 0.0], now - Duration::days(1), None);
    // Large content -> ranks 2nd (within k=2) but doesn't fit the budget.
    let big_content: String = "x".repeat(200);
    let cut_by_budget = write(&mut h, &big_content, vec![0.9, 0.436, 0.0, 0.0], now - Duration::days(1), None);
    // Ranks 3rd -- beyond k=2, so cut before budget packing is even reached.
    let cut_by_k = write(&mut h, "c", vec![0.5, 0.866, 0.0, 0.0], now - Duration::days(1), None);
    // Was valid, but its interval already closed before "now".
    let filtered_by_time = write(
        &mut h,
        "d",
        vec![0.0, 1.0, 0.0, 0.0],
        now - Duration::days(10),
        Some(now - Duration::days(1)),
    );

    h.indexes.keyword.commit().unwrap();

    let query = Query {
        text: None,
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        embedding_model: None,
        namespace: NamespaceId("default".into()),
        as_of: None,
        k: 2,
        max_tokens: 10, // enough for "injected"'s small ciphertext, not for the 200-byte one
    };

    let (explanations, retrieval_id) = search(&h.ledger, &h.indexes, query).unwrap();

    let outcome_of = |fact_id: Uuid| {
        explanations
            .iter()
            .find(|e| e.fact_id == fact_id)
            .unwrap_or_else(|| panic!("fact_id {fact_id} missing from explanations -- lossy output"))
            .outcome
    };

    assert_eq!(outcome_of(injected), Outcome::Injected);
    assert_eq!(outcome_of(cut_by_budget), Outcome::CutByBudget);
    assert_eq!(outcome_of(cut_by_k), Outcome::CutByK);
    assert_eq!(outcome_of(filtered_by_time), Outcome::FilteredByTime);
    assert_eq!(explanations.len(), 4, "every considered candidate must appear, not just the winners");

    // The full trail was also durably written as a Retrieval record.
    let retrieval_record = h
        .ledger
        .scan_from(0)
        .unwrap()
        .map(|r| r.unwrap())
        .find(|r| matches!(&r.payload, crate::record::Payload::Retrieval(ret) if ret.retrieval_id == retrieval_id))
        .expect("Retrieval record not found in ledger");
    match retrieval_record.payload {
        crate::record::Payload::Retrieval(ret) => assert_eq!(ret.candidates.len(), 4),
        _ => unreachable!(),
    }
}
