use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use uuid::Uuid;

use crate::index::{Indexes, KeywordIndex, VectorIndex};
use crate::read_path::{hybrid_search, Query, SearchError};
use crate::record::{ModelFingerprint, NamespaceId};

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
    indexes: Indexes,
}

impl Drop for Harness {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn harness(tag: &str) -> Harness {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("memvault-read-path-test-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let vector = VectorIndex::open_or_create(&dir.join("vectors.usearch"), &fingerprint()).unwrap();
    let keyword = KeywordIndex::open_or_create(&dir.join("keyword")).unwrap();
    Harness { dir, indexes: Indexes { vector, keyword } }
}

fn base_query() -> Query {
    Query {
        text: None,
        embedding: None,
        embedding_model: None,
        namespace: NamespaceId("default".into()),
        as_of: None,
        k: 10,
        max_tokens: 4096,
    }
}

/// The plan's acceptance test: construct a fake ANN list and BM25 list
/// where one doc is rank 1 in both and another is rank 1 in only one;
/// assert the doubly-ranked doc wins fusion.
#[test]
fn test_rrf_favors_candidate_ranked_high_in_both_lists() {
    let mut h = harness("rrf");

    let doubly_ranked = Uuid::from_u128(1);
    let ann_only_winner = Uuid::from_u128(2);
    let filler = Uuid::from_u128(3);

    // doubly_ranked: close to the query vector AND its content matches the
    // query text. ann_only_winner: closest vector, but irrelevant text.
    h.indexes.vector.insert(doubly_ranked, &[1.0, 0.0, 0.0, 0.0]).unwrap();
    h.indexes.vector.insert(ann_only_winner, &[0.99, 0.0, 0.0, 0.0]).unwrap();
    h.indexes.vector.insert(filler, &[-1.0, 0.0, 0.0, 0.0]).unwrap();

    h.indexes.keyword.insert(doubly_ranked, "the hash-chained ledger", &[]).unwrap();
    h.indexes.keyword.insert(ann_only_winner, "an unrelated paragraph about weather", &[]).unwrap();
    h.indexes.keyword.insert(filler, "another unrelated paragraph about food", &[]).unwrap();
    h.indexes.keyword.commit().unwrap();

    let mut query = base_query();
    query.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
    query.text = Some("hash-chained ledger".into());

    let results = hybrid_search(&h.indexes, &query).unwrap();
    assert_eq!(results[0].fact_id, doubly_ranked, "candidate ranked #1 in both lists should win RRF fusion");
    assert!(results[0].ann_rank.is_some());
    assert!(results[0].bm25_rank.is_some());
}

#[test]
fn as_of_query_is_rejected_as_not_yet_supported() {
    let h = harness("as-of");
    let mut query = base_query();
    query.as_of = Some(chrono::Utc::now());

    let result = hybrid_search(&h.indexes, &query);
    assert!(matches!(result, Err(SearchError::AsOfNotYetSupported)));
}

#[test]
fn mismatched_embedding_model_is_rejected() {
    let h = harness("fingerprint-mismatch");
    let mut query = base_query();
    query.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
    query.embedding_model = Some(ModelFingerprint {
        name: "different-model".into(),
        dimensions: 4,
        revision_hash: [9u8; 32],
    });

    let result = hybrid_search(&h.indexes, &query);
    assert!(matches!(result, Err(SearchError::EmbeddingModelMismatch)));
}

#[test]
fn empty_query_returns_no_candidates() {
    let h = harness("empty");
    let results = hybrid_search(&h.indexes, &base_query()).unwrap();
    assert!(results.is_empty());
}
