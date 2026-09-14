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
//!
//! Two chains live here. The *facts* chain (`ledger.redb`) holds Assert,
//! Supersede, Erase and Checkpoint records and grows only when memory
//! changes; it is never pruned. The *retrievals* chain (`retrievals.redb`,
//! alongside) holds one Retrieval record per search, indexed by
//! retrieval_id so `explain` is a lookup rather than a scan, and it can be
//! pruned from the front by age: the oldest retained record's `prev_hash`
//! still commits to everything pruned before it, so verification from that
//! point stays sound while the history stops growing without bound. An
//! agent that searches on every turn writes far more retrievals than
//! facts, and without this split the facts chain paid for all of them.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use redb::{ReadableDatabase, ReadableTable, TableDefinition};
use uuid::Uuid;

use crate::chain;
use crate::record::{self, Assert, DecodeError, Erase, NamespaceId, Payload, Record, Supersede};

const RECORDS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("records");
/// retrieval_id bytes -> seq. Only the retrievals chain ever has entries;
/// written in the same transaction as the record it points at.
const RETRIEVAL_INDEX: TableDefinition<&[u8], u64> = TableDefinition::new("retrieval_index");

/// The retrievals chain's file, next to `ledger.redb` in a data directory.
pub const RETRIEVALS_FILE: &str = "retrievals.redb";

#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    #[error("ledger storage error: {0}")]
    Redb(redb::Error),
    #[error("ledger record corrupt: {0}")]
    Decode(#[from] DecodeError),
    /// A write supplied a `fact_id` that is open in a different namespace.
    /// Unlike the other two this is the caller's mistake, not broken
    /// storage -- see `write_assert` for why it can't be honoured.
    #[error("fact {fact_id} belongs to namespace {}, so a write to namespace {} cannot supersede it", .fact_namespace.0, .write_namespace.0)]
    CrossNamespaceSupersede {
        fact_id: Uuid,
        fact_namespace: NamespaceId,
        write_namespace: NamespaceId,
    },
}

crate::redb_error!(LedgerError, LedgerError::Redb);

/// One hash chain in one redb file. `Ledger` owns two.
struct Chain {
    db: redb::Database,
}

impl Chain {
    fn open(path: &Path) -> Result<Self, LedgerError> {
        let db = redb::Database::create(path)?;

        // Ensure the tables exist so every other method can assume they do,
        // rather than special-casing "never written to" everywhere.
        let txn = db.begin_write()?;
        txn.open_table(RECORDS_TABLE)?;
        txn.open_table(RETRIEVAL_INDEX)?;
        txn.commit()?;

        Ok(Chain { db })
    }

    /// The oldest seq still present. Zero for an unpruned chain, higher
    /// after `prune_before`, `None` when empty.
    fn first_seq(&self) -> Result<Option<u64>, LedgerError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS_TABLE)?;
        Ok(table.first()?.map(|(k, _)| k.value()))
    }

    /// The seq the next `append` gets: one past the newest record. Not the
    /// record count, since pruning removes records from the front without
    /// renumbering what remains.
    fn head(&self) -> Result<u64, LedgerError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS_TABLE)?;
        Ok(table.last()?.map(|(k, _)| k.value() + 1).unwrap_or(0))
    }

    fn append_batch(&self, namespace: NamespaceId, recorded_at: DateTime<Utc>, payloads: Vec<Payload>) -> Result<Vec<u64>, LedgerError> {
        let write_txn = self.db.begin_write()?;
        let mut seqs = Vec::with_capacity(payloads.len());
        {
            let mut table = write_txn.open_table(RECORDS_TABLE)?;
            let mut index = write_txn.open_table(RETRIEVAL_INDEX)?;

            let (mut next_seq, mut prev_hash): (u64, [u8; 32]) = match table.last()? {
                Some((seq, bytes)) => (seq.value() + 1, blake3::hash(bytes.value()).into()),
                None => (0, chain::GENESIS_PREV_HASH),
            };

            for payload in payloads {
                let record = Record::new(next_seq, prev_hash, recorded_at, namespace.clone(), payload);
                if let Payload::Retrieval(r) = &record.payload {
                    index.insert(r.retrieval_id.as_bytes().as_slice(), next_seq)?;
                }
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

    fn read(&self, seq: u64) -> Result<Option<Record>, LedgerError> {
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
    fn scan_from(&self, seq: u64) -> Result<impl Iterator<Item = Result<Record, LedgerError>>, LedgerError> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(RECORDS_TABLE)?;
        let range = table.range(seq..)?;

        Ok(range.map(|entry| {
            let (_, value) = entry.map_err(|e| LedgerError::Redb(e.into()))?;
            Ok(record::decode_record(value.value())?)
        }))
    }

    /// Verifies from `seq` onward, trusting the record at `seq` as an
    /// already-verified resume point (see `chain::verify_chain_from`).
    fn verify_from(&self, seq: u64) -> Result<(), VerifyError> {
        let records = self.scan_from(seq).map_err(VerifyError::Ledger)?;
        let mut collected = Vec::new();
        for record in records {
            collected.push(record.map_err(VerifyError::Ledger)?);
        }
        chain::verify_chain_from(collected.into_iter(), seq).map_err(VerifyError::Chain)
    }

    /// Verifies everything still present: from the genesis for an
    /// unpruned chain, from the oldest retained record otherwise.
    fn verify(&self) -> Result<(), VerifyError> {
        match self.first_seq().map_err(VerifyError::Ledger)? {
            Some(first) => self.verify_from(first),
            None => Ok(()),
        }
    }

    fn find_retrieval(&self, retrieval_id: Uuid) -> Result<Option<Record>, LedgerError> {
        let txn = self.db.begin_read()?;
        let index = txn.open_table(RETRIEVAL_INDEX)?;
        let Some(seq) = index.get(retrieval_id.as_bytes().as_slice())?.map(|g| g.value()) else {
            return Ok(None);
        };
        let table = txn.open_table(RECORDS_TABLE)?;
        match table.get(seq)? {
            Some(guard) => Ok(Some(record::decode_record(guard.value())?)),
            None => Ok(None),
        }
    }

    /// Removes every record recorded before `cutoff`, except the newest
    /// record, which always stays: it is the anchor the next append chains
    /// from, and its `prev_hash` is what still commits to everything
    /// removed. Records are appended in time order, so the walk stops at
    /// the first one at or after the cutoff.
    fn prune_before(&self, cutoff: DateTime<Utc>) -> Result<PruneOutcome, LedgerError> {
        let mut doomed: Vec<(u64, Option<Uuid>)> = Vec::new();
        {
            let txn = self.db.begin_read()?;
            let table = txn.open_table(RECORDS_TABLE)?;
            let newest = table.last()?.map(|(k, _)| k.value());
            for entry in table.iter()? {
                let (seq, bytes) = entry.map_err(|e| LedgerError::Redb(e.into()))?;
                let seq = seq.value();
                if Some(seq) == newest {
                    break;
                }
                let record = record::decode_record(bytes.value())?;
                if record.header.recorded_at >= cutoff {
                    break;
                }
                let retrieval_id = match &record.payload {
                    Payload::Retrieval(r) => Some(r.retrieval_id),
                    _ => None,
                };
                doomed.push((seq, retrieval_id));
            }
        }

        if !doomed.is_empty() {
            let write_txn = self.db.begin_write()?;
            {
                let mut table = write_txn.open_table(RECORDS_TABLE)?;
                let mut index = write_txn.open_table(RETRIEVAL_INDEX)?;
                for (seq, retrieval_id) in &doomed {
                    table.remove(*seq)?;
                    if let Some(id) = retrieval_id {
                        index.remove(id.as_bytes().as_slice())?;
                    }
                }
            }
            write_txn.commit()?;
        }

        Ok(PruneOutcome { pruned: doomed.len() as u64, first_seq: self.first_seq()? })
    }
}

pub struct Ledger {
    facts: Chain,
    retrievals: Chain,
    /// fact_id -> ledger seq of its currently-open Assert. A pure derived
    /// cache: rebuilt by a full replay on every `open()`, kept in sync by
    /// `write_assert` on every write. Not persisted separately -- there is
    /// nothing here that isn't already recoverable from the ledger itself.
    open_facts: Mutex<HashMap<Uuid, u64>>,
}

impl Ledger {
    /// Opens the facts chain at `path` and the retrievals chain in
    /// [`RETRIEVALS_FILE`] beside it, creating either if absent.
    pub fn open(path: &Path) -> Result<Self, LedgerError> {
        let facts = Chain::open(path)?;
        let retrievals = Chain::open(&path.with_file_name(RETRIEVALS_FILE))?;
        let ledger = Ledger { facts, retrievals, open_facts: Mutex::new(HashMap::new()) };

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
        *crate::lock(&ledger.open_facts) = open_facts;

        Ok(ledger)
    }

    /// The seq of `fact_id`'s currently-open `Assert`, if any.
    pub fn open_assert_seq(&self, fact_id: Uuid) -> Option<u64> {
        crate::lock(&self.open_facts).get(&fact_id).copied()
    }

    /// Number of records in the facts chain, i.e. the seq that will be
    /// assigned to the next facts-chain `append`. Zero when empty.
    pub fn head(&self) -> Result<u64, LedgerError> {
        self.facts.head()
    }

    /// Appends a single record to the chain it belongs in: a `Retrieval`
    /// to the retrievals chain, anything else to the facts chain.
    pub fn append(&self, namespace: NamespaceId, recorded_at: DateTime<Utc>, payload: Payload) -> Result<u64, LedgerError> {
        let chain = match payload {
            Payload::Retrieval(_) => &self.retrievals,
            _ => &self.facts,
        };
        Ok(chain.append_batch(namespace, recorded_at, vec![payload])?[0])
    }

    /// Appends `payloads` as consecutive facts-chain records inside one
    /// write transaction, chained from the current head. Returns their
    /// assigned seqs in order. This is what lets `write_assert` commit a
    /// `Supersede` and its `Assert` atomically (product doc §6.3 step 4).
    /// Retrievals go through `append`, one record each.
    pub fn append_batch(&self, namespace: NamespaceId, recorded_at: DateTime<Utc>, payloads: Vec<Payload>) -> Result<Vec<u64>, LedgerError> {
        debug_assert!(!payloads.iter().any(|p| matches!(p, Payload::Retrieval(_))), "retrievals belong in their own chain; use `append`");
        self.facts.append_batch(namespace, recorded_at, payloads)
    }

    /// Writes `assert`, first closing any currently-open Assert for the
    /// same `fact_id` with a `Supersede`, both in one transaction. Holds
    /// `open_facts` for the whole operation -- see the module doc comment
    /// for why that matters. Returns the new Assert's seq and, if this
    /// write superseded a prior one, that prior Assert's seq.
    pub fn write_assert(&self, namespace: NamespaceId, recorded_at: DateTime<Utc>, assert: Assert) -> Result<WriteAssertOutcome, LedgerError> {
        let mut open_facts = crate::lock(&self.open_facts);
        let fact_id = assert.fact_id;
        let superseded_seq = open_facts.get(&fact_id).copied();

        // A fact_id is a global handle -- `supersede` and `erase` take one
        // with no namespace and close the fact wherever it lives -- so a
        // fact_id open elsewhere is a caller mistake, not a request to move
        // the fact. Refused rather than honoured: the Supersede would be
        // filed under the writing namespace, where the owning namespace's
        // as_of replay could never see it, leaving the fact closed to the
        // fact view and open to its owner with no way to reconcile the two.
        //
        // Checked under the same lock as the write it guards, so a
        // concurrent writer can't move the fact between the check and the
        // append.
        if let Some(target_seq) = superseded_seq {
            let target = self.read(target_seq)?.expect("open_facts points at a seq that must exist in the ledger");
            if target.header.namespace != namespace {
                return Err(LedgerError::CrossNamespaceSupersede {
                    fact_id,
                    fact_namespace: target.header.namespace,
                    write_namespace: namespace,
                });
            }
        }

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

    /// Shared by `write_supersede`/`write_erase`: both close `fact_id`'s
    /// currently-open `Assert` by appending one record that references it
    /// (a `Supersede` or an `Erase`), atomically under the fact-view lock
    /// -- see the module doc comment. `None` if `fact_id` has no open
    /// Assert. The closed record's own namespace is reused (a caller
    /// closing an existing fact has no independent namespace to supply).
    /// Returns the closed Assert's seq and the new record's seq.
    fn close_open_fact(&self, recorded_at: DateTime<Utc>, fact_id: Uuid, build_payload: impl FnOnce(u64) -> Payload) -> Result<Option<(u64, u64)>, LedgerError> {
        let mut open_facts = crate::lock(&self.open_facts);
        let Some(target_seq) = open_facts.get(&fact_id).copied() else {
            return Ok(None);
        };

        let target = self
            .read(target_seq)?
            .expect("open_facts points at a seq that must exist in the ledger");

        let seqs = self.append_batch(target.header.namespace, recorded_at, vec![build_payload(target_seq)])?;
        let closing_seq = seqs[0];

        open_facts.remove(&fact_id);

        Ok(Some((target_seq, closing_seq)))
    }

    /// Closes `fact_id`'s currently-open `Assert` with a `Supersede`,
    /// without writing a new `Assert` -- product doc §9's mitigation for
    /// caller-directed supersession, exposed standalone rather than only
    /// as a side effect of `write_assert`.
    pub fn write_supersede(&self, recorded_at: DateTime<Utc>, fact_id: Uuid, valid_to: DateTime<Utc>, reason: Option<String>) -> Result<Option<WriteSupersedeOutcome>, LedgerError> {
        let outcome = self.close_open_fact(recorded_at, fact_id, |target_seq| Payload::Supersede(Supersede { fact_id, target_seq, valid_to, reason }))?;
        Ok(outcome.map(|(target_seq, supersede_seq)| WriteSupersedeOutcome { target_seq, supersede_seq }))
    }

    /// Appends an `Erase` record for `fact_id`'s currently-open `Assert`.
    /// Product doc §6.5: recording the erasure is the ledger's part of the
    /// job; destroying the key (making the ciphertext this record still
    /// points at permanently unreadable) is the caller's, via `Keyring`.
    pub fn write_erase(&self, recorded_at: DateTime<Utc>, fact_id: Uuid, reason: String) -> Result<Option<WriteEraseOutcome>, LedgerError> {
        let outcome = self.close_open_fact(recorded_at, fact_id, |target_seq| Payload::Erase(Erase { fact_id, target_seq, reason }))?;
        Ok(outcome.map(|(target_seq, erase_seq)| WriteEraseOutcome { target_seq, erase_seq }))
    }

    /// One facts-chain record by seq.
    pub fn read(&self, seq: u64) -> Result<Option<Record>, LedgerError> {
        self.facts.read(seq)
    }

    /// Streams facts-chain records from `seq` (inclusive) to the head.
    pub fn scan_from(&self, seq: u64) -> Result<impl Iterator<Item = Result<Record, LedgerError>>, LedgerError> {
        self.facts.scan_from(seq)
    }

    /// Verifies the facts chain from the genesis.
    pub fn verify(&self) -> Result<(), VerifyError> {
        self.verify_from(0)
    }

    /// Verifies the facts chain from `seq` onward, trusting the record at
    /// `seq` as an already-verified resume point rather than re-deriving it
    /// from a predecessor. `memvault verify --from`.
    pub fn verify_from(&self, seq: u64) -> Result<(), VerifyError> {
        self.facts.verify_from(seq)
    }

    /// The `Retrieval` record with this id, by index lookup. `None` if it
    /// was never written here or has been pruned.
    pub fn find_retrieval(&self, retrieval_id: Uuid) -> Result<Option<Record>, LedgerError> {
        self.retrievals.find_retrieval(retrieval_id)
    }

    /// The seq the next retrieval gets. Grows for the life of the chain;
    /// pruning never renumbers.
    pub fn retrievals_head(&self) -> Result<u64, LedgerError> {
        self.retrievals.head()
    }

    /// The oldest retrieval still present: where `verify_retrievals` starts
    /// and the earliest search `explain` can still answer for.
    pub fn retrievals_first_seq(&self) -> Result<Option<u64>, LedgerError> {
        self.retrievals.first_seq()
    }

    /// Streams retrievals-chain records from `seq` (inclusive) to its head.
    pub fn scan_retrievals_from(&self, seq: u64) -> Result<impl Iterator<Item = Result<Record, LedgerError>>, LedgerError> {
        self.retrievals.scan_from(seq)
    }

    /// Verifies the retrievals chain over everything still present: from
    /// the genesis if nothing was ever pruned, otherwise from the oldest
    /// retained record, whose own `prev_hash` commits to what came before.
    pub fn verify_retrievals(&self) -> Result<(), VerifyError> {
        self.retrievals.verify()
    }

    /// Verifies the retrievals chain from `seq` onward.
    pub fn verify_retrievals_from(&self, seq: u64) -> Result<(), VerifyError> {
        self.retrievals.verify_from(seq)
    }

    /// Drops retrievals recorded before `cutoff` from the front of the
    /// retrievals chain, always keeping the newest record as the anchor.
    /// Pruned searches can no longer be explained; the chain still verifies
    /// from the first record kept. Facts are never pruned.
    pub fn prune_retrievals_before(&self, cutoff: DateTime<Utc>) -> Result<PruneOutcome, LedgerError> {
        self.retrievals.prune_before(cutoff)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteAssertOutcome {
    pub assert_seq: u64,
    pub superseded_seq: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteSupersedeOutcome {
    pub target_seq: u64,
    pub supersede_seq: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteEraseOutcome {
    pub target_seq: u64,
    pub erase_seq: u64,
}

/// What `prune_retrievals_before` did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PruneOutcome {
    pub pruned: u64,
    /// The oldest retained seq afterwards, where verification now starts.
    /// `None` only for a chain that was empty to begin with.
    pub first_seq: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error(transparent)]
    Ledger(LedgerError),
    #[error(transparent)]
    Chain(chain::ChainError),
}
