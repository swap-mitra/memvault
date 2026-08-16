//! Cryptographic erase (product doc §6.5, P7): destroy the fact's key so
//! its content becomes permanently unreadable everywhere, append an
//! `Erase` record, and remove it from both indexes. The `Assert` record
//! itself is never touched -- it stays in the chain forever, ciphertext
//! intact and `content_hash` independently verifiable, just undecryptable.

use chrono::Utc;
use uuid::Uuid;

use crate::crypto::{Keyring, KeyringError};
use crate::index::{IndexError, Indexes};
use crate::ledger::{Ledger, LedgerError};

#[derive(Debug)]
pub enum EraseError {
    /// `fact_id` has no currently-open Assert -- nothing to erase.
    /// ponytail: mirrors `supersede_fact`'s scope -- an already-closed
    /// fact_id (closed via `supersede_fact` rather than erased) can't be
    /// erased this way yet. Nothing in the plan's exit tests needs that;
    /// upgrade path is keying off the keyring entry's presence instead of
    /// `open_facts` if closed-fact erasure is needed later.
    NotFound,
    Ledger(LedgerError),
    Keyring(KeyringError),
    Index(IndexError),
}

impl std::fmt::Display for EraseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EraseError::NotFound => write!(f, "no open fact with that id"),
            EraseError::Ledger(e) => write!(f, "{e}"),
            EraseError::Keyring(e) => write!(f, "{e}"),
            EraseError::Index(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for EraseError {}

impl From<LedgerError> for EraseError {
    fn from(e: LedgerError) -> Self {
        EraseError::Ledger(e)
    }
}
impl From<KeyringError> for EraseError {
    fn from(e: KeyringError) -> Self {
        EraseError::Keyring(e)
    }
}
impl From<IndexError> for EraseError {
    fn from(e: IndexError) -> Self {
        EraseError::Index(e)
    }
}

/// Erases `fact_id`. Its whole supersession lineage shares one key (see
/// crypto.rs), so destroying it makes every past version unreadable too,
/// not only the current one.
///
/// The key is destroyed *before* the ledger records the erasure: if a
/// crash happens between the two, the worst case is a fact_id the ledger
/// still shows as open whose content is already unreadable -- a call to
/// `erase` finds it still open and finishes the job. The alternative
/// order risks the reverse: the ledger claiming a fact was erased while
/// its key is still live and the content still recoverable, which is the
/// one state this function must never leave behind.
pub fn erase(ledger: &Ledger, keyring: &mut Keyring, indexes: &mut Indexes, fact_id: Uuid, reason: String) -> Result<(), EraseError> {
    if ledger.open_assert_seq(fact_id).is_none() {
        return Err(EraseError::NotFound);
    }

    keyring.destroy_key(fact_id)?;

    let outcome = ledger.write_erase(Utc::now(), fact_id, reason)?.ok_or(EraseError::NotFound)?;

    indexes.vector.remove(fact_id)?;
    indexes.keyword.remove(fact_id)?;
    indexes.keyword.commit()?;

    indexes.vector.set_watermark(outcome.erase_seq + 1)?;
    indexes.keyword.set_watermark(outcome.erase_seq + 1)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::content_hash;
    use crate::index::{KeywordIndex, VectorIndex};
    use crate::record::{ModelFingerprint, NamespaceId, Payload, SourceRef};
    use crate::write_path::{write_fact, WriteInput};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn fingerprint() -> ModelFingerprint {
        ModelFingerprint { name: "test-model".into(), dimensions: 4, revision_hash: [1u8; 32] }
    }

    struct Harness {
        dir: PathBuf,
        ledger: Ledger,
        keyring: Keyring,
        indexes: Indexes,
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn harness(tag: &str) -> Harness {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("memvault-erase-test-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let ledger = Ledger::open(&dir.join("ledger.redb")).unwrap();
        let keyring = Keyring::open(&dir.join("keys.redb")).unwrap();
        let vector = VectorIndex::open_or_create(&dir.join("vectors.usearch"), &fingerprint()).unwrap();
        let keyword = KeywordIndex::open_or_create(&dir.join("keyword")).unwrap();

        Harness { dir, ledger, keyring, indexes: Indexes { vector, keyword } }
    }

    fn input(content: &str) -> WriteInput {
        WriteInput {
            namespace: NamespaceId("default".into()),
            content: content.as_bytes().to_vec(),
            embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
            embedding_model: fingerprint(),
            valid_from: chrono::Utc::now(),
            valid_to: None,
            fact_id: None,
            keywords: vec![],
            pinned: false,
            source: SourceRef::default(),
        }
    }

    /// The plan's acceptance test: write, search (found), erase,
    /// verify_chain (passes), search again (absent), raw ledger read of
    /// the Assert (present, undecryptable).
    #[test]
    fn test_erase_preserves_chain_removes_from_search() {
        let mut h = harness("acceptance");
        let plaintext = b"the plaintext that must eventually be forgotten";
        let hash_before = content_hash(plaintext);
        let fact_id = write_fact(&h.ledger, &mut h.indexes, &mut h.keyring, input("the plaintext that must eventually be forgotten")).unwrap();

        assert_eq!(h.indexes.keyword.search("forgotten", 5).unwrap()[0].0, fact_id);

        erase(&h.ledger, &mut h.keyring, &mut h.indexes, fact_id, "gdpr request".into()).unwrap();

        h.ledger.verify().unwrap();

        assert!(h.indexes.keyword.search("forgotten", 5).unwrap().is_empty());
        assert!(h.ledger.open_assert_seq(fact_id).is_none());

        let record = h.ledger.read(0).unwrap().unwrap();
        match record.payload {
            Payload::Assert(a) => {
                assert_eq!(a.fact_id, fact_id);
                assert_eq!(a.content_hash, hash_before);
                let result = h.keyring.decrypt(fact_id, &a.content);
                assert_eq!(result, Err(crate::crypto::DecryptError::KeyDestroyed));
            }
            other => panic!("expected the original Assert to still be present, got {other:?}"),
        }

        match h.ledger.read(1).unwrap().unwrap().payload {
            Payload::Erase(e) => {
                assert_eq!(e.fact_id, fact_id);
                assert_eq!(e.target_seq, 0);
                assert_eq!(e.reason, "gdpr request");
            }
            other => panic!("expected an Erase record at seq 1, got {other:?}"),
        }
    }

    #[test]
    fn erase_unknown_fact_id_is_not_found() {
        let mut h = harness("unknown");
        let result = erase(&h.ledger, &mut h.keyring, &mut h.indexes, Uuid::from_u128(999), "n/a".into());
        assert!(matches!(result, Err(EraseError::NotFound)));
    }
}
