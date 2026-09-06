//! Scaffolding every test module was writing for itself: a temp directory
//! that cleans itself up, the stock fingerprint, and the four stores opened
//! over one directory.

use tempfile::TempDir;

use crate::crypto::Keyring;
use crate::index::{Indexes, KeywordIndex, VectorIndex};
use crate::ledger::Ledger;
use crate::record::ModelFingerprint;

/// A directory removed when the returned handle drops.
pub fn tmp() -> TempDir {
    tempfile::tempdir().expect("temp dir")
}

/// The fingerprint tests write under unless they are testing fingerprints.
pub fn fingerprint() -> ModelFingerprint {
    ModelFingerprint { name: "test-model".into(), dimensions: 4, revision_hash: [1u8; 32] }
}

/// Ledger, keyring and both indexes over one temp directory. `dir` is last
/// to drop, so the stores close before it is removed.
pub struct Harness {
    pub ledger: Ledger,
    pub keyring: Keyring,
    pub indexes: Indexes,
    /// Held for its Drop, which removes the directory.
    #[allow(dead_code)]
    pub dir: TempDir,
}

pub fn harness() -> Harness {
    let dir = tmp();
    let ledger = Ledger::open(&dir.path().join("ledger.redb")).unwrap();
    let keyring = Keyring::open(&dir.path().join("keys.redb")).unwrap();
    let vector = VectorIndex::open_or_create(&dir.path().join("vectors.usearch"), &fingerprint()).unwrap();
    let keyword = KeywordIndex::open_or_create(&dir.path().join("keyword")).unwrap();
    Harness { ledger, keyring, indexes: Indexes { vector, keyword }, dir }
}
