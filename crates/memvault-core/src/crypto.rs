//! Payload encryption and the keyring (product doc §6.1, §6.5, P7).
//!
//! The keyring is a separate redb database from the ledger, at a separate
//! path with its own lifecycle: key destruction must not be a mutation of
//! the append-only ledger itself (product doc §6.5). Erasing a fact means
//! destroying its key here, not touching the `Assert` record's ciphertext,
//! which stays in the chain forever, permanently unreadable.

use std::path::Path;

use chacha20poly1305::aead::{Aead, Generate, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use redb::{ReadableDatabase, TableDefinition};
use uuid::Uuid;

use crate::record::Encrypted;

const KEYS_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("keys");

#[derive(Debug)]
pub enum KeyringError {
    Redb(redb::Error),
}

impl std::fmt::Display for KeyringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyringError::Redb(e) => write!(f, "keyring storage error: {e}"),
        }
    }
}

impl std::error::Error for KeyringError {}

macro_rules! redb_error {
    ($t:ty) => {
        impl From<$t> for KeyringError {
            fn from(e: $t) -> Self {
                KeyringError::Redb(e.into())
            }
        }
    };
}

redb_error!(redb::DatabaseError);
redb_error!(redb::TransactionError);
redb_error!(redb::TableError);
redb_error!(redb::StorageError);
redb_error!(redb::CommitError);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecryptError {
    /// No key on file for this fact_id: either it was erased (product doc
    /// §6.5) or never existed. The ciphertext alone can't distinguish
    /// those, and callers don't need it to -- either way it can't be read.
    KeyDestroyed,
    /// AEAD authentication failed: wrong key, corrupted ciphertext, or a
    /// tampered nonce.
    InvalidCiphertext,
    Storage(String),
}

impl std::fmt::Display for DecryptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecryptError::KeyDestroyed => write!(f, "key destroyed or never existed"),
            DecryptError::InvalidCiphertext => write!(f, "ciphertext failed authentication"),
            DecryptError::Storage(e) => write!(f, "keyring storage error: {e}"),
        }
    }
}

impl std::error::Error for DecryptError {}

pub struct Keyring {
    db: redb::Database,
}

impl Keyring {
    pub fn open(path: &Path) -> Result<Self, KeyringError> {
        let db = redb::Database::create(path)?;
        let txn = db.begin_write()?;
        txn.open_table(KEYS_TABLE)?;
        txn.commit()?;
        Ok(Keyring { db })
    }

    /// Generates a fresh key for `fact_id`, stores it, and encrypts
    /// `plaintext` under it in one step. There is no separate
    /// generate-then-encrypt sequence a caller could get out of order or
    /// skip half of.
    pub fn encrypt(&mut self, fact_id: Uuid, plaintext: &[u8]) -> Result<Encrypted, KeyringError> {
        let key = Key::generate();
        let nonce = Nonce::generate();
        let cipher = ChaCha20Poly1305::new(&key);
        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .expect("chacha20poly1305 encryption of an in-memory buffer cannot fail");

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(KEYS_TABLE)?;
            table.insert(fact_id.as_bytes().as_slice(), key.as_slice())?;
        }
        write_txn.commit()?;

        Ok(Encrypted {
            nonce: nonce.into(),
            ciphertext,
        })
    }

    pub fn decrypt(&self, fact_id: Uuid, encrypted: &Encrypted) -> Result<Vec<u8>, DecryptError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| DecryptError::Storage(e.to_string()))?;
        let table = read_txn
            .open_table(KEYS_TABLE)
            .map_err(|e| DecryptError::Storage(e.to_string()))?;
        let key_bytes = table
            .get(fact_id.as_bytes().as_slice())
            .map_err(|e| DecryptError::Storage(e.to_string()))?
            .ok_or(DecryptError::KeyDestroyed)?
            .value()
            .to_vec();
        let key = Key::try_from(key_bytes.as_slice())
            .map_err(|_| DecryptError::Storage("stored key has the wrong length".into()))?;

        let cipher = ChaCha20Poly1305::new(&key);
        let nonce = Nonce::from(encrypted.nonce);
        cipher
            .decrypt(&nonce, encrypted.ciphertext.as_ref())
            .map_err(|_| DecryptError::InvalidCiphertext)
    }

    /// Cryptographic erase (product doc §6.5, P7): removes the key so the
    /// corresponding ciphertext becomes permanently unreadable. Does not
    /// touch the ledger.
    pub fn destroy_key(&mut self, fact_id: Uuid) -> Result<(), KeyringError> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(KEYS_TABLE)?;
            table.remove(fact_id.as_bytes().as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }
}

/// BLAKE3 over the plaintext, computed independently of the keyring so it
/// survives key destruction (product doc §6.1, §6.5): the chain stays
/// verifiable, and a specific known value's prior presence stays provable,
/// even after the value itself is gone.
pub fn content_hash(plaintext: &[u8]) -> [u8; 32] {
    blake3::hash(plaintext).into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_keyring_path(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "memvault-keyring-test-{tag}-{}-{n}.redb",
            std::process::id()
        ))
    }

    struct TempPath(std::path::PathBuf);
    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn encrypt_then_decrypt_round_trips() {
        let path = TempPath(temp_keyring_path("roundtrip"));
        let mut keyring = Keyring::open(&path.0).unwrap();
        let fact_id = Uuid::from_u128(1);

        let encrypted = keyring.encrypt(fact_id, b"hello memvault").unwrap();
        let plaintext = keyring.decrypt(fact_id, &encrypted).unwrap();

        assert_eq!(plaintext, b"hello memvault");
    }

    #[test]
    fn wrong_fact_id_cannot_decrypt() {
        let path = TempPath(temp_keyring_path("wrong-key"));
        let mut keyring = Keyring::open(&path.0).unwrap();

        let encrypted = keyring.encrypt(Uuid::from_u128(1), b"secret").unwrap();
        let result = keyring.decrypt(Uuid::from_u128(2), &encrypted);

        assert_eq!(result, Err(DecryptError::KeyDestroyed));
    }

    /// The acceptance test from the implementation plan: destroying a key
    /// makes the ciphertext permanently unreadable, while content_hash
    /// (computed independently, over the plaintext) still verifies.
    #[test]
    fn test_erase_makes_content_unrecoverable() {
        let path = TempPath(temp_keyring_path("erase"));
        let mut keyring = Keyring::open(&path.0).unwrap();
        let fact_id = Uuid::from_u128(42);
        let plaintext = b"the plaintext that must eventually be forgotten";

        let hash_before = content_hash(plaintext);
        let encrypted = keyring.encrypt(fact_id, plaintext).unwrap();
        assert_eq!(keyring.decrypt(fact_id, &encrypted).unwrap(), plaintext);

        keyring.destroy_key(fact_id).unwrap();

        assert_eq!(keyring.decrypt(fact_id, &encrypted), Err(DecryptError::KeyDestroyed));
        // content_hash is independent of the keyring entirely -- it survives.
        assert_eq!(content_hash(plaintext), hash_before);
    }

    #[test]
    fn tampered_ciphertext_fails_authentication_not_silently() {
        let path = TempPath(temp_keyring_path("tamper"));
        let mut keyring = Keyring::open(&path.0).unwrap();
        let fact_id = Uuid::from_u128(7);

        let mut encrypted = keyring.encrypt(fact_id, b"authentic").unwrap();
        encrypted.ciphertext[0] ^= 0xFF;

        assert_eq!(keyring.decrypt(fact_id, &encrypted), Err(DecryptError::InvalidCiphertext));
    }
}
