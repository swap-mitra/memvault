/// redb splits failure into five concrete error types that all mean one
/// thing to us. thiserror's `#[from]` maps a single source type per variant,
/// so this fans all five into whichever enum's `Redb` variant is named.
macro_rules! redb_error {
    ($enum:ty, $variant:path) => {
        impl From<redb::DatabaseError> for $enum {
            fn from(e: redb::DatabaseError) -> Self { $variant(e.into()) }
        }
        impl From<redb::TransactionError> for $enum {
            fn from(e: redb::TransactionError) -> Self { $variant(e.into()) }
        }
        impl From<redb::TableError> for $enum {
            fn from(e: redb::TableError) -> Self { $variant(e.into()) }
        }
        impl From<redb::StorageError> for $enum {
            fn from(e: redb::StorageError) -> Self { $variant(e.into()) }
        }
        impl From<redb::CommitError> for $enum {
            fn from(e: redb::CommitError) -> Self { $variant(e.into()) }
        }
    };
}
pub(crate) use redb_error;

pub mod bitemporal;
pub mod budget;
pub mod chain;
pub mod crypto;
pub mod decay;
pub mod embedding;
pub mod erase;
pub mod explain;
pub mod index;
pub mod ledger;
pub mod read_path;
pub mod record;
pub mod recovery;
pub mod write_path;

#[cfg(test)]
mod chain_tests;
#[cfg(test)]
mod explain_tests;
#[cfg(test)]
mod ledger_tests;
#[cfg(test)]
mod read_path_tests;
#[cfg(test)]
mod recovery_tests;
#[cfg(test)]
mod write_path_tests;

pub use bitemporal::{memory_as_of, AsOfError, AsOfFact, AsOfQuery};
pub use budget::{pack_to_budget, PricedCandidate};
pub use chain::{record_hash, verify_chain_from, ChainError};
pub use crypto::{content_hash, DecryptError, Keyring, KeyringError};
pub use decay::{apply_decay, decay_weight, DecayConfig, ScoredCandidate};
pub use embedding::{placeholder_embedding, PLACEHOLDER_EMBEDDING_NAME};
pub use erase::{erase, EraseError};
pub use explain::{explain, search, ExplainError};
pub use index::{IndexError, Indexes, KeywordIndex, VectorIndex};
pub use ledger::{Ledger, LedgerError, VerifyError, WriteAssertOutcome, WriteEraseOutcome, WriteSupersedeOutcome};
pub use read_path::{hybrid_search, FusedCandidate, Query, SearchError};
pub use recovery::{recover, IndexKind, RecoveryConfig, RecoveryError, RecoveryReport};
pub use record::{
    default_fingerprint, Assert, Checkpoint, DecodeError, Encrypted, Erase, Explanation,
    ModelFingerprint, NamespaceId, Outcome, Payload, Record, RecordHeader, RecordKind, Retrieval,
    SourceRef, Supersede,
};
pub use write_path::{supersede_fact, write_fact, SupersedeError, WriteError, WriteInput};
