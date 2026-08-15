//! Append-only ledger store on redb (product doc §6.2, §6.7).
//!
//! `append` is the sole write entry point: callers hand over a payload, not
//! a fully-formed `Record`, so there is no way to construct a record with a
//! wrong `seq` or `prev_hash` -- the ledger assigns both from its own state
//! inside the write transaction that persists it. Writes are serialized by
//! redb's single-writer model, which is what gives the chain its required
//! total order; reads use redb's MVCC snapshots and never block on a
//! writer.

use std::path::Path;

use chrono::{DateTime, Utc};
use redb::{ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};

use crate::chain;
use crate::record::{self, DecodeError, NamespaceId, Payload, Record};

const RECORDS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("records");

#[derive(Debug)]
pub enum LedgerError {
    Redb(redb::Error),
    Decode(DecodeError),
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedgerError::Redb(e) => write!(f, "ledger storage error: {e}"),
            LedgerError::Decode(e) => write!(f, "ledger record corrupt: {e}"),
        }
    }
}

impl std::error::Error for LedgerError {}

impl From<DecodeError> for LedgerError {
    fn from(e: DecodeError) -> Self {
        LedgerError::Decode(e)
    }
}

macro_rules! redb_error {
    ($t:ty) => {
        impl From<$t> for LedgerError {
            fn from(e: $t) -> Self {
                LedgerError::Redb(e.into())
            }
        }
    };
}

redb_error!(redb::DatabaseError);
redb_error!(redb::TransactionError);
redb_error!(redb::TableError);
redb_error!(redb::StorageError);
redb_error!(redb::CommitError);

pub struct Ledger {
    db: redb::Database,
}

impl Ledger {
    pub fn open(path: &Path) -> Result<Self, LedgerError> {
        let db = redb::Database::create(path)?;

        // Ensure the table exists so every other method can assume it does,
        // rather than special-casing "never written to" everywhere.
        let txn = db.begin_write()?;
        txn.open_table(RECORDS_TABLE)?;
        txn.commit()?;

        Ok(Ledger { db })
    }

    /// Number of records in the ledger, i.e. the seq that will be assigned
    /// to the next `append`. Zero for a freshly-opened, empty ledger.
    pub fn head(&self) -> Result<u64, LedgerError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS_TABLE)?;
        Ok(table.len()?)
    }

    /// Appends a new record built from `payload`, assigning it the next
    /// `seq` and chaining it from the current head, inside one write
    /// transaction. Returns the assigned `seq`.
    pub fn append(&self, namespace: NamespaceId, recorded_at: DateTime<Utc>, payload: Payload) -> Result<u64, LedgerError> {
        let write_txn = self.db.begin_write()?;
        let seq;
        {
            let mut table = write_txn.open_table(RECORDS_TABLE)?;
            seq = table.len()?;

            let prev_hash = if seq == 0 {
                chain::GENESIS_PREV_HASH
            } else {
                let prev_bytes = table
                    .get(seq - 1)?
                    .expect("append: previous seq must exist below the current table length")
                    .value()
                    .to_vec();
                blake3::hash(&prev_bytes).into()
            };

            let record = Record::new(seq, prev_hash, recorded_at, namespace, payload);
            table.insert(seq, record::canonical_bytes(&record).as_slice())?;
        }
        write_txn.commit()?;
        Ok(seq)
    }

    pub fn read(&self, seq: u64) -> Result<Option<Record>, LedgerError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS_TABLE)?;
        match table.get(seq)? {
            Some(guard) => Ok(Some(record::decode_record(guard.value())?)),
            None => Ok(None),
        }
    }

    /// Streams records from `seq` (inclusive) to the current head. The
    /// returned iterator owns its snapshot (redb's reference-counted
    /// `range`, not the transaction-borrowed one) so it outlives this call.
    pub fn scan_from(&self, seq: u64) -> Result<impl Iterator<Item = Result<Record, LedgerError>>, LedgerError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS_TABLE)?;
        let range = table.range(seq..)?;

        Ok(range.map(|entry| {
            let (_, value) = entry.map_err(|e| LedgerError::Redb(e.into()))?;
            Ok(record::decode_record(value.value())?)
        }))
    }

    /// Verifies the chain over the current contents of the ledger.
    pub fn verify(&self) -> Result<(), VerifyError> {
        let records = self.scan_from(0).map_err(VerifyError::Ledger)?;
        let mut collected = Vec::new();
        for record in records {
            collected.push(record.map_err(VerifyError::Ledger)?);
        }
        chain::verify_chain(collected.into_iter()).map_err(VerifyError::Chain)
    }
}

#[derive(Debug)]
pub enum VerifyError {
    Ledger(LedgerError),
    Chain(chain::ChainError),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::Ledger(e) => write!(f, "{e}"),
            VerifyError::Chain(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for VerifyError {}
