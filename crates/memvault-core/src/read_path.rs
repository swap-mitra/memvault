//! Read path, steps 1-4 of product doc §6.4: ANN search, BM25 search, a
//! bitemporal check, RRF fuse. Decay (step 5), budget packing (step 6),
//! and provenance emission (step 7) are later tasks; this module's output
//! (`FusedCandidate`) is what they'll consume.
//!
//! Bitemporal filtering here is intentionally thin: `write_fact` retires a
//! fact_id's old index entries before inserting its new ones (see
//! write_path.rs), so the vector/keyword indexes only ever hold *current*
//! versions -- which makes "what is true now" (`as_of: None`) correct with
//! no extra filtering. Point-in-time queries (`as_of: Some(_)`) need
//! historical versions the indexes don't retain, and that's real
//! bitemporal-query work the plan scopes to a later task; asking for one
//! here is a typed error, not a silent wrong answer.

use uuid::Uuid;

use crate::index::{IndexError, Indexes};
use crate::record::{ModelFingerprint, NamespaceId};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Constant from Cormack et al., used by product doc §6.4's RRF step.
const RRF_K: f32 = 60.0;

pub struct Query {
    pub text: Option<String>,
    pub embedding: Option<Vec<f32>>,
    /// Checked against the vector index's stored fingerprint when both are
    /// present (product doc: "A query whose fingerprint does not match the
    /// index is rejected rather than silently returning nonsense"). `None`
    /// skips the check. Not in the plan's bare sketch -- there was no other
    /// way to satisfy that requirement without it.
    pub embedding_model: Option<ModelFingerprint>,
    pub namespace: NamespaceId,
    pub as_of: Option<DateTime<Utc>>,
    pub k: usize,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusedCandidate {
    pub fact_id: Uuid,
    pub ann_rank: Option<u32>,
    pub ann_distance: Option<f32>,
    pub bm25_rank: Option<u32>,
    pub bm25_score: Option<f32>,
    pub rrf_score: f32,
}

#[derive(Debug)]
pub enum SearchError {
    Index(IndexError),
    /// See the module doc comment: real point-in-time queries need
    /// historical index state this read path doesn't retain.
    AsOfNotYetSupported,
    EmbeddingModelMismatch,
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::Index(e) => write!(f, "{e}"),
            SearchError::AsOfNotYetSupported => {
                write!(f, "point-in-time (as_of) queries are not supported yet")
            }
            SearchError::EmbeddingModelMismatch => {
                write!(f, "query embedding's model fingerprint does not match the index")
            }
        }
    }
}

impl std::error::Error for SearchError {}

impl From<IndexError> for SearchError {
    fn from(e: IndexError) -> Self {
        SearchError::Index(e)
    }
}

pub fn hybrid_search(indexes: &Indexes, query: &Query) -> Result<Vec<FusedCandidate>, SearchError> {
    if query.as_of.is_some() {
        return Err(SearchError::AsOfNotYetSupported);
    }

    if let (Some(_), Some(expected)) = (&query.embedding, &query.embedding_model) {
        if indexes.vector.fingerprint() != expected {
            return Err(SearchError::EmbeddingModelMismatch);
        }
    }

    let ann_results = match &query.embedding {
        Some(embedding) => indexes.vector.search(embedding, query.k)?,
        None => Vec::new(),
    };
    let bm25_results = match &query.text {
        Some(text) => indexes.keyword.search(text, query.k)?,
        None => Vec::new(),
    };

    let mut fused: HashMap<Uuid, FusedCandidate> = HashMap::new();
    let empty = |fact_id: Uuid| FusedCandidate {
        fact_id,
        ann_rank: None,
        ann_distance: None,
        bm25_rank: None,
        bm25_score: None,
        rrf_score: 0.0,
    };

    // RRF: score(d) = sum over lists L of 1 / (k + rank_L(d)), rank 1-indexed.
    for (fact_id, distance, rank) in ann_results {
        let entry = fused.entry(fact_id).or_insert_with(|| empty(fact_id));
        entry.ann_rank = Some(rank);
        entry.ann_distance = Some(distance);
        entry.rrf_score += 1.0 / (RRF_K + rank as f32 + 1.0);
    }
    for (fact_id, score, rank) in bm25_results {
        let entry = fused.entry(fact_id).or_insert_with(|| empty(fact_id));
        entry.bm25_rank = Some(rank);
        entry.bm25_score = Some(score);
        entry.rrf_score += 1.0 / (RRF_K + rank as f32 + 1.0);
    }

    let mut results: Vec<FusedCandidate> = fused.into_values().collect();
    results.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).expect("rrf_score is never NaN"));
    Ok(results)
}
