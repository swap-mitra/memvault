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
mod test_support;

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

/// Opens (creating if absent) the ledger, keyring and both indexes that make
/// up a MemVault data directory. Every surface -- CLI, MCP server, FFI,
/// benchmarks -- needs exactly these four over exactly this layout, and a
/// second opinion about the file names would be a data-corrupting one.
/// Recovery is the caller's next step, not this function's: the CLI defers
/// it to an explicit `replay`, the servers run it at startup.
///
/// `requested` is the embedding fingerprint a *new* directory should get;
/// `None` means `default_fingerprint()`. An existing directory keeps the
/// fingerprint it was created with (read back from the vector index's
/// sidecar), and a `requested` width that disagrees with it is an error
/// rather than a silently mismatched index. Callers take
/// `indexes.vector.fingerprint()` as the truth from here on.
pub fn open_stores(
    data_dir: &std::path::Path,
    requested: Option<&ModelFingerprint>,
) -> Result<(Ledger, Keyring, Indexes), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(data_dir)?;
    let ledger = Ledger::open(&data_dir.join("ledger.redb"))?;
    let keyring = Keyring::open(&data_dir.join("keys.redb"))?;
    let vector = VectorIndex::open_or_create(&data_dir.join("vectors.usearch"), requested.unwrap_or(&default_fingerprint()))?;
    if let Some(requested) = requested {
        if vector.fingerprint().dimensions != requested.dimensions {
            return Err(format!(
                "requested {} embedding dimensions, but {} already holds {}-dimensional embeddings",
                requested.dimensions,
                data_dir.display(),
                vector.fingerprint().dimensions
            )
            .into());
        }
    }
    let keyword = KeywordIndex::open_or_create(&data_dir.join("keyword"))?;
    Ok((ledger, keyring, Indexes { vector, keyword }))
}

/// Locks `mutex`, taking the guard back if a previous holder panicked. A
/// panic while a guard is held poisons the mutex, and `unwrap()` on every
/// later lock turns one failed request into a dead process: a server that
/// refuses every call after the first bad one. The ledger is the durable
/// truth and recovery reconciles the indexes at the next start, so serving
/// on is the right call; the panic itself was already reported.
pub fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod lock_tests {
    #[test]
    fn a_poisoned_mutex_still_hands_out_its_value() {
        let m = std::sync::Mutex::new(7);
        let poison = std::panic::catch_unwind(|| {
            let _guard = m.lock().unwrap();
            panic!("holder dies");
        });
        assert!(poison.is_err() && m.is_poisoned());
        assert_eq!(*super::lock(&m), 7);
    }
}

pub use bitemporal::{get_fact, memory_as_of, AsOfError, AsOfFact, AsOfQuery};
pub use budget::{pack_to_budget, PricedCandidate};
pub use chain::{record_hash, verify_chain_from, ChainError};
pub use crypto::{content_hash, DecryptError, Keyring, KeyringError};
pub use decay::{apply_decay, decay_weight, DecayConfig, ScoredCandidate};
pub use embedding::{placeholder_embedding, PLACEHOLDER_EMBEDDING_NAME};
pub use erase::{erase, EraseError};
pub use explain::{explain, explanation_row, injected_contents, outcome_cell, search, ExplainError, InjectedFact, EXPLANATION_HEADER};
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
