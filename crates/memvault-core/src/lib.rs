pub mod bitemporal;
pub mod budget;
pub mod chain;
pub mod crypto;
pub mod decay;
pub mod erase;
pub mod explain;
pub mod index;
pub mod ledger;
pub mod read_path;
pub mod record;
pub mod recovery;
pub mod working_set;
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
pub use budget::{pack_to_budget, PackedCandidate, SkippedCandidate};
pub use chain::{record_hash, verify_chain, ChainError};
pub use crypto::{content_hash, DecryptError, Keyring, KeyringError};
pub use decay::{apply_decay, decay_weight, DecayConfig, ScoredCandidate};
pub use erase::{erase, EraseError};
pub use explain::{explain, search, ExplainError};
pub use index::{IndexError, Indexes, KeywordIndex, VectorIndex};
pub use ledger::{Ledger, LedgerError, VerifyError, WriteAssertOutcome, WriteEraseOutcome, WriteSupersedeOutcome};
pub use read_path::{hybrid_search, FusedCandidate, Query, SearchError};
pub use recovery::{recover, IndexKind, RecoveryConfig, RecoveryError, RecoveryReport};
pub use record::{
    Assert, Checkpoint, DecodeError, Encrypted, Erase, Explanation, ModelFingerprint, NamespaceId,
    Outcome, Payload, Record, RecordHeader, RecordKind, Retrieval, SourceRef, Supersede,
};
pub use working_set::{SessionHandle, SessionId, SessionState, WorkingSet};
pub use write_path::{supersede_fact, write_fact, SupersedeError, WriteError, WriteInput};
