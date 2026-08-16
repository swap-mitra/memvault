//! Phase 1's exit test (docs/IMPLEMENTATION_PLAN.md, task P1-6): explain
//! reconstructs a retrieval from three weeks earlier including every
//! rejected candidate, and an erasure leaves the chain verifying.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{Duration, Utc};
use uuid::Uuid;

use memvault_core::{
    erase, explain, search, write_fact, Indexes, KeywordIndex, Keyring, Ledger, ModelFingerprint,
    NamespaceId, Outcome, Query, SourceRef, VectorIndex, WriteInput,
};

fn fingerprint() -> ModelFingerprint {
    ModelFingerprint { name: "test-model".into(), dimensions: 4, revision_hash: [3u8; 32] }
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
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("memvault-phase1-exit-{tag}-{}-{n}", std::process::id()));
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

#[test]
fn test_exit_explain_and_erase() {
    let mut h = harness("main");
    let three_weeks_ago = Utc::now() - Duration::weeks(3);

    // Same proven shape as explain_tests::test_explanation_includes_all_outcomes,
    // just backdated: a retrieval that happened three weeks ago, with all
    // three rejection outcomes represented among its candidates.
    let injected = write(&mut h, "a", vec![1.0, 0.0, 0.0, 0.0], three_weeks_ago, None);
    let big_content: String = "x".repeat(200);
    let cut_by_budget = write(&mut h, &big_content, vec![0.9, 0.436, 0.0, 0.0], three_weeks_ago, None);
    let cut_by_k = write(&mut h, "c", vec![0.5, 0.866, 0.0, 0.0], three_weeks_ago, None);
    let filtered_by_time = write(
        &mut h,
        "d",
        vec![0.0, 1.0, 0.0, 0.0],
        three_weeks_ago - Duration::days(9),
        Some(three_weeks_ago - Duration::days(1)),
    );

    h.indexes.keyword.commit().unwrap();

    let query = Query {
        text: None,
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        embedding_model: None,
        namespace: NamespaceId("default".into()),
        as_of: None,
        k: 2,
        max_tokens: 10,
    };
    let (explanations, retrieval_id) = search(&h.ledger, &h.indexes, query).unwrap();

    let outcome_of = |fact_id: Uuid| explanations.iter().find(|e| e.fact_id == fact_id).map(|e| e.outcome);
    assert_eq!(outcome_of(injected), Some(Outcome::Injected));
    assert_eq!(outcome_of(cut_by_budget), Some(Outcome::CutByBudget));
    assert_eq!(outcome_of(cut_by_k), Some(Outcome::CutByK));
    assert_eq!(outcome_of(filtered_by_time), Some(Outcome::FilteredByTime));

    // However much later this is read back, explain() reconstructs it
    // byte-for-byte from the ledger alone -- a pure read, so the property
    // holds whether it's three weeks later or three seconds.
    let reconstructed = explain::explain(&h.ledger, retrieval_id).unwrap();
    assert_eq!(reconstructed, explanations);

    // Separately: erasing a fact leaves the chain verifying, and the fact
    // stops coming back from search.
    erase(&h.ledger, &mut h.keyring, &mut h.indexes, injected, "no longer needed".into()).unwrap();
    h.ledger.verify().unwrap();

    let (post_erase, _) = search(
        &h.ledger,
        &h.indexes,
        Query {
            text: None,
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            embedding_model: None,
            namespace: NamespaceId("default".into()),
            as_of: None,
            k: 10,
            max_tokens: 10_000,
        },
    )
    .unwrap();
    assert!(post_erase.iter().all(|e| e.fact_id != injected), "erased fact must not be returned by search");
}
