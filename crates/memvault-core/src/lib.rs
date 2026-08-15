pub mod chain;
pub mod record;

#[cfg(test)]
mod chain_tests;

pub use chain::{record_hash, verify_chain, ChainError};
pub use record::{
    Assert, Checkpoint, Encrypted, Erase, Explanation, ModelFingerprint, NamespaceId, Outcome,
    Payload, Record, RecordHeader, RecordKind, Retrieval, SourceRef, Supersede,
};
