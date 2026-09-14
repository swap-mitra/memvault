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
use crate::crypto::{DecryptError, Keyring};
use crate::decay::{apply_decay, DecayConfig, ScoredCandidate};
use crate::index::{IndexError, Indexes};
use crate::ledger::{Ledger, LedgerError};
use crate::read_path::{hybrid_search, FusedCandidate, Query, SearchError as HybridSearchError};
use crate::record::{Encrypted, Explanation, NamespaceId, Outcome, Payload, Retrieval};

// ponytail: doc §6.9's namespace-config defaults, hardcoded until a real
// per-namespace config loader exists. Upgrade path: read these from
// namespace config once that's built.
const DEFAULT_HALF_LIFE_DAYS: f64 = 30.0;
const DEFAULT_DECAY_FLOOR: f64 = 0.15;

#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error(transparent)]
    Hybrid(#[from] HybridSearchError),
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    #[error(transparent)]
    Index(#[from] IndexError),
    /// An `Injected` candidate's content would not decrypt: erased between
    /// the search and the read, or a keyring that doesn't match the ledger.
    #[error("cannot read content of injected fact {fact_id}: {source}")]
    Decrypt { fact_id: Uuid, source: DecryptError },
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
    /// See [`token_cost`].
    token_cost: u32,
}

/// What a fact costs in context, for budget packing and the `tokens`
/// column.
///
/// ponytail: without the `tokenizer` feature this is ciphertext bytes / 4,
/// a rough English-prose approximation that drifts on code, and it needs
/// no plaintext. The stream cipher keeps the ciphertext the plaintext's
/// length plus a 16-byte tag, so it is a byte count in disguise.
#[cfg(not(feature = "tokenizer"))]
fn token_cost(_keyring: &Keyring, _fact_id: Uuid, content: &Encrypted) -> Result<u32, SearchError> {
    Ok((content.ciphertext.len() as u32) / 4 + 1)
}

/// With the `tokenizer` feature: the cl100k_base BPE count of the decrypted
/// content. Not any one vendor's exact tokenizer, but within a few percent
/// of all of them, which is what a budget needs. The vocabulary is compiled
/// in; nothing is downloaded and no model runs.
#[cfg(feature = "tokenizer")]
fn token_cost(keyring: &Keyring, fact_id: Uuid, content: &Encrypted) -> Result<u32, SearchError> {
    static BPE: std::sync::OnceLock<tiktoken_rs::CoreBPE> = std::sync::OnceLock::new();
    let bpe = BPE.get_or_init(|| tiktoken_rs::cl100k_base().expect("cl100k_base is compiled into tiktoken-rs"));
    let plaintext = keyring.decrypt(fact_id, content).map_err(|source| SearchError::Decrypt { fact_id, source })?;
    Ok(bpe.encode_ordinary(&String::from_utf8_lossy(&plaintext)).len() as u32)
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

/// Same, for a candidate that has been through decay: it carries its own
/// weight and final score, and `resolved` holds what the ledger said.
fn scored_explanation(c: &ScoredCandidate, resolved: &HashMap<Uuid, Resolved>, outcome: Outcome) -> Explanation {
    let r = &resolved[&c.fact_id];
    Explanation {
        fact_id: c.fact_id,
        ledger_seq: r.ledger_seq,
        ann_rank: c.ann_rank,
        ann_distance: c.ann_distance,
        bm25_rank: c.bm25_rank,
        bm25_score: c.bm25_score,
        rrf_score: c.rrf_score,
        decay_weight: c.decay_weight,
        final_score: c.final_score,
        outcome,
        token_cost: r.token_cost,
    }
}

pub fn search(ledger: &Ledger, indexes: &Indexes, keyring: &Keyring, query: Query) -> Result<(Vec<Explanation>, Uuid), SearchError> {
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

        // Both indexes are shared by every namespace in a data directory,
        // so fusion hands back candidates the caller is not entitled to and
        // this is the only place that can tell. Dropped outright rather
        // than reported with an outcome: `explanations` is written verbatim
        // into a Retrieval record, so naming a foreign fact would persist
        // its id and ledger position inside the boundary being enforced.
        //
        // ponytail: filtering after fusion, so a busy namespace can crowd a
        // quiet one out of the fixed-size candidate pool and cost it recall
        // (never correctness -- nothing foreign survives either way).
        // Upgrade path: filter inside the indexes, which for tantivy is a
        // term field on the namespace and for usearch means either a
        // per-namespace index or over-fetching until k survive.
        if record.header.namespace != namespace {
            continue;
        }

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
            token_cost: token_cost(keyring, assert.fact_id, &assert.content)?,
        });
    }

    let candidates: Vec<FusedCandidate> = resolved.iter().map(|r| r.fused.clone()).collect();
    let resolved: HashMap<Uuid, Resolved> = resolved.into_iter().map(|r| (r.fused.fact_id, r)).collect();

    let cfg = DecayConfig { half_life_days: DEFAULT_HALF_LIFE_DAYS, floor: DEFAULT_DECAY_FLOOR };
    let mut scored = apply_decay(
        candidates,
        |id| resolved[&id].pinned,
        |id| resolved[&id].last_accessed_stand_in,
        now,
        &cfg,
    );
    scored.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).expect("final_score is never NaN"));

    let cut_by_k = if scored.len() > k { scored.split_off(k) } else { Vec::new() };
    for c in &cut_by_k {
        explanations.push(scored_explanation(c, &resolved, Outcome::CutByK));
    }

    let (packed, skipped) = pack_to_budget(scored, max_tokens, |id| resolved[&id].token_cost);
    for (candidates, outcome) in [(&packed, Outcome::Injected), (&skipped, Outcome::CutByBudget)] {
        for p in candidates {
            explanations.push(scored_explanation(&p.candidate, &resolved, outcome));
        }
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

/// The plaintext of one `Injected` candidate: what the agent actually gets
/// to read. Kept apart from `Explanation` on purpose -- that struct is
/// written verbatim into the ledger's Retrieval record, and content must
/// only ever live there encrypted under its own erasable key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectedFact {
    pub fact_id: Uuid,
    pub content: Vec<u8>,
}

/// Decrypts every `Injected` candidate of a search, in the order they were
/// packed (best `final_score` first). Any other outcome is skipped: a cut
/// or filtered fact was not handed to the agent, so its content isn't
/// either.
pub fn injected_contents(ledger: &Ledger, keyring: &Keyring, explanations: &[Explanation]) -> Result<Vec<InjectedFact>, SearchError> {
    let mut out = Vec::new();
    for e in explanations.iter().filter(|e| e.outcome == Outcome::Injected) {
        let record = ledger.read(e.ledger_seq)?.ok_or(LedgerError::Decode(crate::record::DecodeError::TrailingBytes))?;
        let Payload::Assert(assert) = record.payload else {
            unreachable!("an Injected candidate's ledger_seq always points at its Assert record");
        };
        let content = keyring
            .decrypt(e.fact_id, &assert.content)
            .map_err(|source| SearchError::Decrypt { fact_id: e.fact_id, source })?;
        out.push(InjectedFact { fact_id: e.fact_id, content });
    }
    Ok(out)
}

#[derive(Debug, thiserror::Error)]
pub enum ExplainError {
    #[error(transparent)]
    Ledger(LedgerError),
    #[error("no retrieval with that id in the ledger")]
    NotFound,
}

/// The provenance table's column widths, shared so the CLI can colour a
/// cell without recomputing the alignment underneath it.
pub const EXPLANATION_HEADER: &str = "fact_id                              ann_rank   ann_dist  bm25_rk bm25_score       rrf  decay_wt     final       outcome tokens";

/// One `Explanation` as a row under [`EXPLANATION_HEADER`], with the outcome
/// cell already padded to its column so a caller can wrap it in escape codes
/// without disturbing the alignment. No trailing newline.
pub fn explanation_row(e: &Explanation, outcome_cell: &str) -> String {
    format!(
        "{:<36} {:>8} {:>10} {:>8} {:>10} {:>9.4} {:>9.4} {:>9.4} {outcome_cell} {:>6}",
        e.fact_id,
        e.ann_rank.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
        e.ann_distance.map(|d| format!("{d:.4}")).unwrap_or_else(|| "-".into()),
        e.bm25_rank.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
        e.bm25_score.map(|s| format!("{s:.4}")).unwrap_or_else(|| "-".into()),
        e.rrf_score,
        e.decay_weight,
        e.final_score,
        e.token_cost,
    )
}

/// The outcome cell padded to its column width, uncoloured.
pub fn outcome_cell(e: &Explanation) -> String {
    format!("{:>13}", format!("{:?}", e.outcome))
}

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
