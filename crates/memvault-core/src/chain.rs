//! BLAKE3 hash-chain construction and verification over records (product
//! doc §6.2).

use crate::record::{canonical_bytes, Record};

/// `seq = 0` must chain from this sentinel; there is no record before it.
pub const GENESIS_PREV_HASH: [u8; 32] = [0u8; 32];

/// BLAKE3 of a record's canonical bytes. This is what the next record's
/// `prev_hash` must equal for the chain to be intact.
pub fn record_hash(record: &Record) -> [u8; 32] {
    blake3::hash(&canonical_bytes(record)).into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainError {
    /// `records[i].header.prev_hash` doesn't equal the recomputed hash of
    /// the record at `seq` — i.e. `seq` is the record whose stored bytes
    /// no longer match what its successor committed to when the chain was
    /// built. `seq` is what changed, not the record that noticed.
    Diverged { seq: u64 },
    /// Records did not arrive as a gapless `0, 1, 2, ...` sequence.
    NonSequential { expected: u64, found: u64 },
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainError::Diverged { seq } => {
                write!(f, "chain diverged: record at seq {seq} does not match what its successor committed to")
            }
            ChainError::NonSequential { expected, found } => {
                write!(f, "non-sequential ledger: expected seq {expected}, found {found}")
            }
        }
    }
}

impl std::error::Error for ChainError {}

/// Walks records in `seq` order and verifies the chain. Cost is linear in
/// ledger size (product doc §6.2). An empty ledger is trivially valid.
///
/// This detects tampering with any record that has a successor. It cannot
/// detect tampering confined to the single most recent record, since
/// nothing yet commits to its hash — that gap is what external checkpoint
/// anchoring closes (product doc §6.2), not this function.
pub fn verify_chain(records: impl Iterator<Item = Record>) -> Result<(), ChainError> {
    let mut prev: Option<Record> = None;
    let mut expected_seq = 0u64;

    for record in records {
        if record.header.seq != expected_seq {
            return Err(ChainError::NonSequential {
                expected: expected_seq,
                found: record.header.seq,
            });
        }

        let expected_prev_hash = match &prev {
            None => GENESIS_PREV_HASH,
            Some(p) => record_hash(p),
        };
        if record.header.prev_hash != expected_prev_hash {
            // The mismatch surfaces here, one record late: it means the
            // *previous* record's committed-to hash no longer matches what
            // this record expects, so the previous record is what changed.
            let culprit_seq = prev.as_ref().map(|p| p.header.seq).unwrap_or(record.header.seq);
            return Err(ChainError::Diverged { seq: culprit_seq });
        }

        expected_seq += 1;
        prev = Some(record);
    }

    Ok(())
}
