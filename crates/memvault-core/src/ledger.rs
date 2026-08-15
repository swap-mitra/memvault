//! Append-only ledger store on redb (product doc §6.2, §6.7).
//!
//! `append`/`append_batch` are the sole low-level write entry points:
//! callers hand over payloads, not fully-formed `Record`s, so there is no
//! way to construct a record with a wrong `seq` or `prev_hash` -- the
//! ledger assigns both from its own state inside the write transaction
//! that persists them. Writes are serialized by redb's single-writer
//! model, which is what gives the chain its required total order; reads
//! use redb's MVCC snapshots and never block on a writer.
//!
//! `write_assert` is a higher-level compound operation on top of those:
//! it holds `open_facts` (see below) for its whole duration, so
//! "is there already an open Assert for this fact_id" and "append its
//! Supersede plus the new Assert" happen as one atomic step -- otherwise
//! two concurrent writers superseding the same fact_id could both see "no
//! open assert" and leave two simultaneously-open Asserts behind, which
//! is exactly the bitemporal invariant (product doc §3 P2) this exists to
//! protect.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use redb::{ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use uuid::Uuid;

use crate::chain;
use crate::record::{self, Assert, DecodeError, NamespaceId, Payload, Record, Supersede};

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
    /// fact_id -> ledger seq of its currently-open Assert. A pure derived
    /// cache: rebuilt by a full replay on every `open()`, kept in sync by
    /// `write_assert` on every write. Not persisted separately -- there is
    /// nothing here that isn't already recoverable from the ledger itself.
    open_facts: Mutex<HashMap<Uuid, u64>>,
}

impl Ledger {
    pub fn open(path: &Path) -> Result<Self, LedgerError> {
        let db = redb::Database::create(path)?;

        // Ensure the table exists so every other method can assume it does,
        // rather than special-casing "never written to" everywhere.
        let txn = db.begin_write()?;
        txn.open_table(RECORDS_TABLE)?;
        txn.commit()?;

        let ledger = Ledger { db, open_facts: Mutex::new(HashMap::new()) };

        let mut open_facts = HashMap::new();
        for record in ledger.scan_from(0)? {
            let record = record?;
            match record.payload {
                Payload::Assert(a) => {
                    open_facts.insert(a.fact_id, record.header.seq);
                }
                Payload::Supersede(s) => {
                    open_facts.remove(&s.fact_id);
                }
                Payload::Erase(e) => {
                    open_facts.remove(&e.fact_id);
                }
                Payload::Retrieval(_) | Payload::Checkpoint(_) => {}
            }
        }
        *ledger.open_facts.lock().unwrap() = open_facts;

        Ok(ledger)
    }

    /// The seq of `fact_id`'s currently-open `Assert`, if any.
    pub fn open_assert_seq(&self, fact_id: Uuid) -> Option<u64> {
        self.open_facts.lock().unwrap().get(&fact_id).copied()
    }

    /// Number of records in the ledger, i.e. the seq that will be assigned
    /// to the next `append`. Zero for a freshly-opened, empty ledger.
    pub fn head(&self) -> Result<u64, LedgerError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS_TABLE)?;
        Ok(table.len()?)
    }

    /// Appends a single record. Convenience wrapper over `append_batch`.
    pub fn append(&self, namespace: NamespaceId, recorded_at: DateTime<Utc>, payload: Payload) -> Result<u64, LedgerError> {
        Ok(self.append_batch(namespace, recorded_at, vec![payload])?[0])
    }

    /// Appends `payloads` as consecutive records inside one write
    /// transaction, chained from the current head. Returns their assigned
    /// seqs in order. This is what lets `write_assert` commit a
    /// `Supersede` and its `Assert` atomically (product doc §6.3 step 4).
    pub fn append_batch(&self, namespace: NamespaceId, recorded_at: DateTime<Utc>, payloads: Vec<Payload>) -> Result<Vec<u64>, LedgerError> {
        let write_txn = self.db.begin_write()?;
        let mut seqs = Vec::with_capacity(payloads.len());
        {
            let mut table = write_txn.open_table(RECORDS_TABLE)?;
            let mut next_seq = table.len()?;

            let mut prev_hash = if next_seq == 0 {
                chain::GENESIS_PREV_HASH
            } else {
                let prev_bytes = table
                    .get(next_seq - 1)?
                    .expect("append_batch: previous seq must exist below the current table length")
                    .value()
                    .to_vec();
                blake3::hash(&prev_bytes).into()
            };

            for payload in payloads {
                let record = Record::new(next_seq, prev_hash, recorded_at, namespace.clone(), payload);
                let bytes = record::canonical_bytes(&record);
                table.insert(next_seq, bytes.as_slice())?;
                prev_hash = blake3::hash(&bytes).into();
                seqs.push(next_seq);
                next_seq += 1;
            }
        }
        write_txn.commit()?;
        Ok(seqs)
    }

    /// Writes `assert`, first closing any currently-open Assert for the
    /// same `fact_id` with a `Supersede`, both in one transaction. Holds
    /// `open_facts` for the whole operation -- see the module doc comment
    /// for why that matters. Returns the new Assert's seq and, if this
    /// write superseded a prior one, that prior Assert's seq.
    pub fn write_assert(&self, namespace: NamespaceId, recorded_at: DateTime<Utc>, assert: Assert) -> Result<WriteAssertOutcome, LedgerError> {
        let mut open_facts = self.open_facts.lock().unwrap();
        let fact_id = assert.fact_id;
        let superseded_seq = open_facts.get(&fact_id).copied();

        let mut payloads = Vec::with_capacity(2);
        if let Some(target_seq) = superseded_seq {
            payloads.push(Payload::Supersede(Supersede {
                fact_id,
                target_seq,
                valid_to: assert.valid_from,
                reason: None,
            }));
        }
        payloads.push(Payload::Assert(assert));

        let seqs = self.append_batch(namespace, recorded_at, payloads)?;
        let assert_seq = *seqs.last().expect("append_batch always returns at least one seq");

        open_facts.insert(fact_id, assert_seq);

        Ok(WriteAssertOutcome { assert_seq, superseded_seq })
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteAssertOutcome {
    pub assert_seq: u64,
    pub superseded_seq: Option<u64>,
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
