//! Recovery and index replay (product doc §6.6): compare each index's
//! watermark against the ledger head, replay forward on the common case,
//! reset-and-replay-from-0 when an index can't be trusted incrementally
//! (a fingerprint mismatch, or a watermark past the ledger head -- an
//! impossible state).
//!
//! "Missing" from the doc's trigger list needs no separate code path
//! here: a freshly-created index already has watermark 0, so replaying
//! from its watermark forward *is* a full rebuild in that case. "Corrupt"
//! at the file level surfaces earlier, as an error from
//! `VectorIndex`/`KeywordIndex::open_or_create` itself -- an index that
//! won't even open can't be recovered by this function, only by the
//! caller deciding to delete and recreate it before calling `recover`.

use crate::chain;
use crate::crypto::{DecryptError, Keyring};
use crate::index::{IndexError, Indexes};
use crate::ledger::{Ledger, LedgerError, VerifyError};
use crate::record::{ModelFingerprint, Payload};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    Vector,
    Keyword,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RecoveryConfig {
    pub verify_chain: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryReport {
    /// The lowest seq replay started from, if any index needed it.
    pub replayed_from: Option<u64>,
    pub rebuilt: Vec<IndexKind>,
    pub verified: bool,
}

#[derive(Debug)]
pub enum RecoveryError {
    Ledger(LedgerError),
    Index(IndexError),
    Chain(chain::ChainError),
    Decrypt(DecryptError),
}

impl std::fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoveryError::Ledger(e) => write!(f, "{e}"),
            RecoveryError::Index(e) => write!(f, "{e}"),
            RecoveryError::Chain(e) => write!(f, "{e}"),
            RecoveryError::Decrypt(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RecoveryError {}

impl From<LedgerError> for RecoveryError {
    fn from(e: LedgerError) -> Self {
        RecoveryError::Ledger(e)
    }
}
impl From<IndexError> for RecoveryError {
    fn from(e: IndexError) -> Self {
        RecoveryError::Index(e)
    }
}
impl From<VerifyError> for RecoveryError {
    fn from(e: VerifyError) -> Self {
        match e {
            VerifyError::Ledger(le) => RecoveryError::Ledger(le),
            VerifyError::Chain(ce) => RecoveryError::Chain(ce),
        }
    }
}

pub fn recover(
    ledger: &Ledger,
    indexes: &mut Indexes,
    keyring: &Keyring,
    expected_fingerprint: &ModelFingerprint,
    cfg: RecoveryConfig,
) -> Result<RecoveryReport, RecoveryError> {
    let head = ledger.head()?;

    let verified = if cfg.verify_chain {
        ledger.verify()?;
        true
    } else {
        false
    };

    let mut rebuilt = Vec::new();

    if indexes.vector.fingerprint() != expected_fingerprint || indexes.vector.watermark() > head {
        indexes.vector.reset(expected_fingerprint)?;
        rebuilt.push(IndexKind::Vector);
    }
    if indexes.keyword.watermark() > head {
        indexes.keyword.reset()?;
        rebuilt.push(IndexKind::Keyword);
    }

    let vector_from = indexes.vector.watermark();
    let keyword_from = indexes.keyword.watermark();
    let replay_start = vector_from.min(keyword_from);

    let replayed_from = if replay_start < head {
        for record in ledger.scan_from(replay_start)? {
            let record = record?;
            let seq = record.header.seq;
            apply_record(&record.payload, seq, vector_from, keyword_from, indexes, keyring)?;
        }
        indexes.keyword.commit()?;
        indexes.vector.set_watermark(head)?;
        indexes.keyword.set_watermark(head)?;
        Some(replay_start)
    } else {
        None
    };

    Ok(RecoveryReport { replayed_from, rebuilt, verified })
}

fn apply_record(
    payload: &Payload,
    seq: u64,
    vector_from: u64,
    keyword_from: u64,
    indexes: &mut Indexes,
    keyring: &Keyring,
) -> Result<(), RecoveryError> {
    match payload {
        Payload::Assert(a) => {
            if seq >= vector_from {
                indexes.vector.remove(a.fact_id)?;
                if let Some(embedding) = &a.embedding {
                    indexes.vector.insert(a.fact_id, embedding)?;
                }
            }
            if seq >= keyword_from {
                indexes.keyword.remove(a.fact_id)?;
                match keyring.decrypt(a.fact_id, &a.content) {
                    Ok(plaintext) => {
                        let content_text = String::from_utf8_lossy(&plaintext);
                        indexes.keyword.insert(a.fact_id, &content_text, &a.keywords)?;
                    }
                    // Erased: the plaintext is gone forever (product doc
                    // §6.6/P7), so it can't be re-indexed. Correct, not a
                    // failure -- nothing to propagate.
                    Err(DecryptError::KeyDestroyed) => {}
                    Err(e) => return Err(RecoveryError::Decrypt(e)),
                }
            }
        }
        Payload::Supersede(s) => {
            if seq >= vector_from {
                indexes.vector.remove(s.fact_id)?;
            }
            if seq >= keyword_from {
                indexes.keyword.remove(s.fact_id)?;
            }
        }
        Payload::Erase(e) => {
            if seq >= vector_from {
                indexes.vector.remove(e.fact_id)?;
            }
            if seq >= keyword_from {
                indexes.keyword.remove(e.fact_id)?;
            }
        }
        Payload::Retrieval(_) | Payload::Checkpoint(_) => {}
    }
    Ok(())
}
