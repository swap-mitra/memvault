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
    verify_chain_from(records, 0)
}

/// Like [`verify_chain`], but starts at `start_seq` instead of the
/// genesis. For `start_seq == 0` this is identical to `verify_chain`
/// (the first record's `prev_hash` is still checked against
/// `GENESIS_PREV_HASH`). For `start_seq > 0`, the record at `start_seq` is
/// trusted as an already-verified resume point -- there is no predecessor
/// in `records` to re-derive its `prev_hash` from -- and only the chain
/// *from* there forward is confirmed. Product doc §6.2's `verify --from`.
pub fn verify_chain_from(records: impl Iterator<Item = Record>, start_seq: u64) -> Result<(), ChainError> {
    let mut prev: Option<Record> = None;
    let mut expected_seq = start_seq;

    for record in records {
        if record.header.seq != expected_seq {
            return Err(ChainError::NonSequential {
                expected: expected_seq,
                found: record.header.seq,
            });
        }

        match &prev {
            Some(p) => {
                let expected_prev_hash = record_hash(p);
                if record.header.prev_hash != expected_prev_hash {
                    // The mismatch surfaces here, one record late: it means
                    // the *previous* record's committed-to hash no longer
                    // matches what this record expects, so the previous
                    // record is what changed.
                    return Err(ChainError::Diverged { seq: p.header.seq });
                }
            }
            None if expected_seq == 0 => {
                if record.header.prev_hash != GENESIS_PREV_HASH {
                    return Err(ChainError::Diverged { seq: 0 });
                }
            }
            // Resuming mid-chain: nothing to check the first record's
            // prev_hash against, by design -- it's the trusted resume point.
            None => {}
        }

        expected_seq += 1;
        prev = Some(record);
    }

    Ok(())
}
