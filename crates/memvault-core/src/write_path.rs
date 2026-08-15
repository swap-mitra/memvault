//! The write path (product doc §6.3):
//! validate boundary -> detect supersession -> encrypt+hash -> append
//! (Assert [+ Supersede]) in one txn -> commit -> post-commit index insert.
//!
//! Steps 2 (detect supersession) and 4-5 (append + commit) collapse into
//! `Ledger::write_assert`, which holds the ledger's fact-view lock for the
//! whole operation so the two happen atomically -- see ledger.rs's module
//! doc comment for why that matters.
//!
//! Step 6 (index insert) is deliberately after commit and outside the
//! ledger transaction (product doc §6.3): a crash here leaves the ledger
//! correct and the index behind, which recovery (a later task) reconciles
//! via the watermark. This module's job is only to not corrupt anything if
//! interrupted, not to close that gap.

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::crypto::{self, Keyring, KeyringError};
use crate::index::{IndexError, Indexes};
use crate::ledger::{Ledger, LedgerError};
use crate::record::{Assert, ModelFingerprint, NamespaceId, SourceRef};

/// Default from product doc §6.9's example config. Becomes a real
/// per-namespace setting once namespace config loading exists; no such
/// loader exists yet, so this is the one value used everywhere for now.
pub const DEFAULT_MAX_CONTENT_BYTES: usize = 65536;

pub struct WriteInput {
    pub namespace: NamespaceId,
    pub content: Vec<u8>,
    pub embedding: Option<Vec<f32>>,
    /// The model that produced `embedding`. Required even when `embedding`
    /// is `None`, matching `Assert::embedding_model` (product doc §6.1) --
    /// not in the plan's original sketch, but there's no other source for
    /// it until namespace config loading exists.
    pub embedding_model: ModelFingerprint,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    /// `Some(id)`: continue/update the fact lineage `id`, superseding
    /// whichever Assert is currently open for it, if any. `None`: a brand
    /// new fact, assigned a fresh fact_id.
    pub fact_id: Option<Uuid>,
    pub keywords: Vec<String>,
    pub pinned: bool,
    pub source: SourceRef,
}

#[derive(Debug)]
pub enum WriteError {
    InvalidInterval,
    ContentTooLarge { max: usize, actual: usize },
    EmbeddingDimensionMismatch { expected: u32, actual: usize },
    Ledger(LedgerError),
    Keyring(KeyringError),
    Index(IndexError),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::InvalidInterval => write!(f, "valid_to must be after valid_from"),
            WriteError::ContentTooLarge { max, actual } => {
                write!(f, "content is {actual} bytes, over the {max}-byte limit")
            }
            WriteError::EmbeddingDimensionMismatch { expected, actual } => {
                write!(f, "embedding has {actual} dimensions, model expects {expected}")
            }
            WriteError::Ledger(e) => write!(f, "{e}"),
            WriteError::Keyring(e) => write!(f, "{e}"),
            WriteError::Index(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for WriteError {}

impl From<LedgerError> for WriteError {
    fn from(e: LedgerError) -> Self {
        WriteError::Ledger(e)
    }
}
impl From<KeyringError> for WriteError {
    fn from(e: KeyringError) -> Self {
        WriteError::Keyring(e)
    }
}
impl From<IndexError> for WriteError {
    fn from(e: IndexError) -> Self {
        WriteError::Index(e)
    }
}

fn validate(input: &WriteInput) -> Result<(), WriteError> {
    if let Some(valid_to) = input.valid_to {
        if valid_to <= input.valid_from {
            return Err(WriteError::InvalidInterval);
        }
    }
    if input.content.len() > DEFAULT_MAX_CONTENT_BYTES {
        return Err(WriteError::ContentTooLarge {
            max: DEFAULT_MAX_CONTENT_BYTES,
            actual: input.content.len(),
        });
    }
    if let Some(embedding) = &input.embedding {
        if embedding.len() != input.embedding_model.dimensions as usize {
            return Err(WriteError::EmbeddingDimensionMismatch {
                expected: input.embedding_model.dimensions,
                actual: embedding.len(),
            });
        }
    }
    Ok(())
}

pub fn write_fact(ledger: &Ledger, indexes: &mut Indexes, keyring: &mut Keyring, input: WriteInput) -> Result<Uuid, WriteError> {
    validate(&input)?;

    let fact_id = input.fact_id.unwrap_or_else(Uuid::new_v4);
    let content_hash = crypto::content_hash(&input.content);
    let encrypted = keyring.encrypt(fact_id, &input.content)?;

    let assert = Assert {
        fact_id,
        valid_from: input.valid_from,
        valid_to: input.valid_to,
        content: encrypted,
        content_hash,
        embedding: input.embedding.clone(),
        embedding_model: input.embedding_model,
        keywords: input.keywords.clone(),
        pinned: input.pinned,
        source: input.source,
    };

    ledger.write_assert(input.namespace, Utc::now(), assert)?;

    // Post-commit, outside the ledger transaction (see module doc). Remove
    // first: idempotent no-op for a brand new fact_id, and correctly
    // retires the stale version's entries when this write superseded one,
    // since fact_id is stable across a supersession chain.
    indexes.vector.remove(fact_id)?;
    indexes.keyword.remove(fact_id)?;

    if let Some(embedding) = &input.embedding {
        indexes.vector.insert(fact_id, embedding)?;
    }
    let content_text = String::from_utf8_lossy(&input.content);
    indexes.keyword.insert(fact_id, &content_text, &input.keywords)?;
    indexes.keyword.commit()?;

    Ok(fact_id)
}
