pub mod chain;
pub mod crypto;
pub mod index;
pub mod ledger;
pub mod read_path;
pub mod record;
pub mod write_path;

#[cfg(test)]
mod chain_tests;
#[cfg(test)]
mod ledger_tests;
#[cfg(test)]
mod read_path_tests;
#[cfg(test)]
mod write_path_tests;

pub use chain::{record_hash, verify_chain, ChainError};
pub use crypto::{content_hash, DecryptError, Keyring, KeyringError};
pub use index::{IndexError, Indexes, KeywordIndex, VectorIndex};
pub use ledger::{Ledger, LedgerError, VerifyError, WriteAssertOutcome};
pub use read_path::{hybrid_search, FusedCandidate, Query, SearchError};
pub use record::{
    Assert, Checkpoint, DecodeError, Encrypted, Erase, Explanation, ModelFingerprint, NamespaceId,
    Outcome, Payload, Record, RecordHeader, RecordKind, Retrieval, SourceRef, Supersede,
};
pub use write_path::{write_fact, WriteError, WriteInput};
