//! Provenance emission (product doc §6.4 step 7): orchestrates hybrid
//! search -> bitemporal check -> decay -> top-k cut -> budget packing into
//! the full `Explanation` list, lossy for nothing -- every candidate
//! `hybrid_search` returned appears here with a populated `outcome`, and
//! the whole list is written into a `Retrieval` ledger record so
//! `memvault explain <retrieval_id>` (a later task) can reconstruct it
//! exactly from the ledger alone.

use std::collections::HashMap;

use chrono::Utc;
use uuid::Uuid;

use crate::budget::pack_to_budget;
use crate::decay::{apply_decay, DecayConfig};
use crate::index::{IndexError, Indexes};
use crate::ledger::{Ledger, LedgerError};
use crate::read_path::{hybrid_search, FusedCandidate, Query, SearchError as HybridSearchError};
use crate::record::{Explanation, NamespaceId, Outcome, Payload, Retrieval};

// ponytail: doc §6.9's namespace-config defaults, hardcoded until a real
// per-namespace config loader exists. Upgrade path: read these from
// namespace config once that's built.
const DEFAULT_HALF_LIFE_DAYS: f64 = 30.0;
const DEFAULT_DECAY_FLOOR: f64 = 0.15;

#[derive(Debug)]
pub enum SearchError {
    Hybrid(HybridSearchError),
    Ledger(LedgerError),
    Index(IndexError),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::Hybrid(e) => write!(f, "{e}"),
            SearchError::Ledger(e) => write!(f, "{e}"),
            SearchError::Index(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SearchError {}

impl From<HybridSearchError> for SearchError {
    fn from(e: HybridSearchError) -> Self {
        SearchError::Hybrid(e)
    }
}
impl From<LedgerError> for SearchError {
    fn from(e: LedgerError) -> Self {
        SearchError::Ledger(e)
    }
}
impl From<IndexError> for SearchError {
    fn from(e: IndexError) -> Self {
        SearchError::Index(e)
    }
}

struct Resolved {
    fused: FusedCandidate,
    ledger_seq: u64,
    pinned: bool,
    /// ponytail: stand-in for real last-access tracking, which no store
    /// exists for yet -- uses the fact's own valid_from, so decay behaves
    /// sanely but does not yet "reinforce" on retrieval as product doc
    /// §6.4 specifies. Upgrade path: a persisted fact_id -> last-accessed-
    /// at store, updated by this function once one exists, threaded in
    /// here instead of derived from the Assert record.
    last_accessed_stand_in: chrono::DateTime<Utc>,
    /// ponytail: token_cost ~= ciphertext byte length / 4, a rough
    /// English-text approximation with no tokenizer dependency. Upgrade
    /// path: a real tokenizer, when one is available without violating
    /// P5 (no model calls on the hot path).
    token_cost: u32,
}

fn explanation(fused: &FusedCandidate, ledger_seq: u64, decay_weight: f32, final_score: f32, outcome: Outcome, token_cost: u32) -> Explanation {
    Explanation {
        fact_id: fused.fact_id,
        ledger_seq,
        ann_rank: fused.ann_rank,
        ann_distance: fused.ann_distance,
        bm25_rank: fused.bm25_rank,
        bm25_score: fused.bm25_score,
        rrf_score: fused.rrf_score,
        decay_weight,
        final_score,
        outcome,
        token_cost,
    }
}

pub fn search(ledger: &Ledger, indexes: &Indexes, query: Query) -> Result<(Vec<Explanation>, Uuid), SearchError> {
    let namespace = NamespaceId(query.namespace.0.clone());
    let k = query.k;
    let max_tokens = query.max_tokens;
    let query_text = query.text.clone();
    let query_embedding_model = query.embedding_model.clone();
    let as_of = query.as_of;

    let fused = hybrid_search(indexes, &query)?;
    let now = Utc::now();

    let mut resolved = Vec::with_capacity(fused.len());
    let mut explanations = Vec::with_capacity(fused.len());

    for candidate in fused {
        let Some(ledger_seq) = ledger.open_assert_seq(candidate.fact_id) else {
            // Stale index entry: the fact was superseded/erased since it
            // was indexed. No longer valid at any as_of, by definition.
            explanations.push(explanation(&candidate, 0, 0.0, 0.0, Outcome::FilteredByTime, 0));
            continue;
        };
        let record = ledger.read(ledger_seq)?.ok_or(LedgerError::Decode(crate::record::DecodeError::TrailingBytes))?;
        let Payload::Assert(assert) = record.payload else {
            unreachable!("open_facts only ever points at Assert records");
        };

        let effective_as_of = as_of.unwrap_or(now);
        let time_valid = assert.valid_from <= effective_as_of && assert.valid_to.is_none_or(|vt| effective_as_of < vt);
        if !time_valid {
            explanations.push(explanation(&candidate, ledger_seq, 0.0, 0.0, Outcome::FilteredByTime, 0));
            continue;
        }

        resolved.push(Resolved {
            fused: candidate,
            ledger_seq,
            pinned: assert.pinned,
            last_accessed_stand_in: assert.valid_from,
            token_cost: (assert.content.ciphertext.len() as u32) / 4 + 1,
        });
    }

    let pinned_map: HashMap<Uuid, bool> = resolved.iter().map(|r| (r.fused.fact_id, r.pinned)).collect();
    let access_map: HashMap<Uuid, chrono::DateTime<Utc>> = resolved.iter().map(|r| (r.fused.fact_id, r.last_accessed_stand_in)).collect();
    let seq_map: HashMap<Uuid, u64> = resolved.iter().map(|r| (r.fused.fact_id, r.ledger_seq)).collect();
    let cost_map: HashMap<Uuid, u32> = resolved.iter().map(|r| (r.fused.fact_id, r.token_cost)).collect();

    let candidates: Vec<FusedCandidate> = resolved.into_iter().map(|r| r.fused).collect();
    let cfg = DecayConfig { half_life_days: DEFAULT_HALF_LIFE_DAYS, floor: DEFAULT_DECAY_FLOOR };
    let mut scored = apply_decay(candidates, |id| pinned_map[&id], |id| access_map[&id], now, &cfg);
    scored.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).expect("final_score is never NaN"));

    let cut_by_k = if scored.len() > k { scored.split_off(k) } else { Vec::new() };
    for c in &cut_by_k {
        explanations.push(Explanation {
            fact_id: c.fact_id,
            ledger_seq: seq_map[&c.fact_id],
            ann_rank: c.ann_rank,
            ann_distance: c.ann_distance,
            bm25_rank: c.bm25_rank,
            bm25_score: c.bm25_score,
            rrf_score: c.rrf_score,
            decay_weight: c.decay_weight,
            final_score: c.final_score,
            outcome: Outcome::CutByK,
            token_cost: cost_map[&c.fact_id],
        });
    }

    let (packed, skipped) = pack_to_budget(scored, max_tokens, |id| cost_map[&id]);
    for p in &packed {
        explanations.push(Explanation {
            fact_id: p.candidate.fact_id,
            ledger_seq: seq_map[&p.candidate.fact_id],
            ann_rank: p.candidate.ann_rank,
            ann_distance: p.candidate.ann_distance,
            bm25_rank: p.candidate.bm25_rank,
            bm25_score: p.candidate.bm25_score,
            rrf_score: p.candidate.rrf_score,
            decay_weight: p.candidate.decay_weight,
            final_score: p.candidate.final_score,
            outcome: Outcome::Injected,
            token_cost: p.token_cost,
        });
    }
    for s in &skipped {
        explanations.push(Explanation {
            fact_id: s.candidate.fact_id,
            ledger_seq: seq_map[&s.candidate.fact_id],
            ann_rank: s.candidate.ann_rank,
            ann_distance: s.candidate.ann_distance,
            bm25_rank: s.candidate.bm25_rank,
            bm25_score: s.candidate.bm25_score,
            rrf_score: s.candidate.rrf_score,
            decay_weight: s.candidate.decay_weight,
            final_score: s.candidate.final_score,
            outcome: Outcome::CutByBudget,
            token_cost: s.token_cost,
        });
    }

    let retrieval_id = Uuid::new_v4();
    let retrieval = Retrieval {
        retrieval_id,
        query_text,
        query_embedding_model,
        as_of,
        max_tokens,
        k: k as u32,
        candidates: explanations.clone(),
    };
    ledger.append(namespace, now, Payload::Retrieval(retrieval))?;

    Ok((explanations, retrieval_id))
}

#[derive(Debug)]
pub enum ExplainError {
    Ledger(LedgerError),
    NotFound,
}

impl std::fmt::Display for ExplainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExplainError::Ledger(e) => write!(f, "{e}"),
            ExplainError::NotFound => write!(f, "no retrieval with that id in the ledger"),
        }
    }
}

impl std::error::Error for ExplainError {}

/// Reconstructs a past retrieval exactly from its `Retrieval` ledger
/// record. A linear scan: fine at the ledger sizes this project targets,
/// and there's no retrieval_id index yet to do better with.
pub fn explain(ledger: &Ledger, retrieval_id: Uuid) -> Result<Vec<Explanation>, ExplainError> {
    for record in ledger.scan_from(0).map_err(ExplainError::Ledger)? {
        let record = record.map_err(ExplainError::Ledger)?;
        if let Payload::Retrieval(r) = record.payload {
            if r.retrieval_id == retrieval_id {
                return Ok(r.candidates);
            }
        }
    }
    Err(ExplainError::NotFound)
}
