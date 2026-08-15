//! Vector index: a usearch HNSW graph plus a small redb-backed sidecar for
//! the bits usearch doesn't store -- the fact_id<->u64 key mapping (usearch
//! keys are `u64`, ours are `Uuid`), the watermark, and the model
//! fingerprint.
//!
//! The risk spike (see docs/IMPLEMENTATION_PLAN.md, and
//! examples/hnsw_supersede_spike.rs) found tombstone accumulation from
//! repeated supersession is not a first-order concern at moderate churn;
//! `remove` does not schedule periodic `compact()` here, since nothing yet
//! calls `remove` at volume to need it.

use std::path::{Path, PathBuf};

use redb::{ReadableDatabase, ReadableTable, TableDefinition};
use usearch::{Index as UsearchIndex, IndexOptions, MetricKind, ScalarKind};
use uuid::Uuid;

use super::IndexError;
use crate::record::ModelFingerprint;

const FORWARD_TABLE: TableDefinition<&[u8], u64> = TableDefinition::new("forward");
const REVERSE_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("reverse");
const META_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");

/// `<path>` gains a suffix rather than replacing its extension, so any
/// extension the caller chose for the usearch file survives.
fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(suffix);
    PathBuf::from(os)
}

fn encode_fingerprint(fp: &ModelFingerprint) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(4 + 32 + fp.name.len());
    bytes.extend_from_slice(&fp.dimensions.to_le_bytes());
    bytes.extend_from_slice(&fp.revision_hash);
    bytes.extend_from_slice(fp.name.as_bytes());
    bytes
}

fn decode_fingerprint(bytes: &[u8]) -> Result<ModelFingerprint, IndexError> {
    if bytes.len() < 36 {
        return Err(IndexError::Corrupt("fingerprint sidecar entry too short".into()));
    }
    let dimensions = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let revision_hash: [u8; 32] = bytes[4..36].try_into().unwrap();
    let name = String::from_utf8(bytes[36..].to_vec())
        .map_err(|_| IndexError::Corrupt("fingerprint name is not valid utf-8".into()))?;
    Ok(ModelFingerprint { name, dimensions, revision_hash })
}

pub struct VectorIndex {
    index: UsearchIndex,
    meta_db: redb::Database,
    fingerprint: ModelFingerprint,
    watermark: u64,
}

impl VectorIndex {
    pub fn open_or_create(path: &Path, fingerprint: &ModelFingerprint) -> Result<Self, IndexError> {
        let meta_db = redb::Database::create(sidecar_path(path, ".meta.redb"))?;
        {
            let txn = meta_db.begin_write()?;
            txn.open_table(FORWARD_TABLE)?;
            txn.open_table(REVERSE_TABLE)?;
            txn.open_table(META_TABLE)?;
            txn.commit()?;
        }

        let exists = path.exists();
        let index = if exists {
            let path_str = path.to_str().ok_or_else(|| IndexError::Corrupt("index path is not valid utf-8".into()))?;
            UsearchIndex::restore(path_str)?
        } else {
            let options = IndexOptions {
                dimensions: fingerprint.dimensions as usize,
                metric: MetricKind::Cos,
                quantization: ScalarKind::F32,
                connectivity: 0,
                expansion_add: 0,
                expansion_search: 0,
                multi: false,
            };
            let index = UsearchIndex::new(&options)?;
            index.reserve(1024)?;
            index
        };

        let (watermark, stored_fingerprint) = {
            let read_txn = meta_db.begin_read()?;
            let table = read_txn.open_table(META_TABLE)?;
            let watermark = match table.get("watermark")? {
                Some(g) => u64::from_le_bytes(
                    g.value()
                        .try_into()
                        .map_err(|_| IndexError::Corrupt("watermark sidecar entry has wrong length".into()))?,
                ),
                None => 0,
            };
            let fp = match table.get("fingerprint")? {
                Some(g) => decode_fingerprint(g.value())?,
                None => fingerprint.clone(),
            };
            (watermark, fp)
        };

        let this = VectorIndex { index, meta_db, fingerprint: stored_fingerprint, watermark };
        if !exists {
            this.persist_fingerprint()?;
        }
        Ok(this)
    }

    fn persist_fingerprint(&self) -> Result<(), IndexError> {
        let bytes = encode_fingerprint(&self.fingerprint);
        let write_txn = self.meta_db.begin_write()?;
        {
            let mut table = write_txn.open_table(META_TABLE)?;
            table.insert("fingerprint", bytes.as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Assigns a fresh usearch key, records the fact_id<->key mapping, then
    /// inserts into the graph. fact_ids are never reused (product doc: each
    /// Assert gets its own fact_id, supersession never rewrites one), so
    /// this is always a first insert for `fact_id`, never an update.
    pub fn insert(&mut self, fact_id: Uuid, embedding: &[f32]) -> Result<(), IndexError> {
        let write_txn = self.meta_db.begin_write()?;
        let key = {
            let mut forward = write_txn.open_table(FORWARD_TABLE)?;
            let mut reverse = write_txn.open_table(REVERSE_TABLE)?;

            // usearch keys are u64; ours are 128-bit Uuids. Derive a
            // candidate from the fact_id and linearly probe on the
            // (astronomically unlikely, but real) case of a collision --
            // silently aliasing two facts' vectors would be a correctness
            // bug, not just a performance one.
            let mut attempt: u64 = 0;
            let key = loop {
                let hash_input: Vec<u8> = if attempt == 0 {
                    fact_id.as_bytes().to_vec()
                } else {
                    let mut v = fact_id.as_bytes().to_vec();
                    v.extend_from_slice(&attempt.to_le_bytes());
                    v
                };
                let candidate = u64::from_le_bytes(blake3::hash(&hash_input).as_bytes()[0..8].try_into().unwrap());
                if reverse.get(candidate)?.is_none() {
                    break candidate;
                }
                attempt += 1;
            };

            forward.insert(fact_id.as_bytes().as_slice(), key)?;
            reverse.insert(key, fact_id.as_bytes().as_slice())?;
            key
        };
        write_txn.commit()?;

        self.index.add(key, embedding)?;
        Ok(())
    }

    pub fn remove(&mut self, fact_id: Uuid) -> Result<(), IndexError> {
        let write_txn = self.meta_db.begin_write()?;
        let key = {
            let mut forward = write_txn.open_table(FORWARD_TABLE)?;
            let mut reverse = write_txn.open_table(REVERSE_TABLE)?;
            let key = forward.get(fact_id.as_bytes().as_slice())?.map(|g| g.value());
            if let Some(key) = key {
                forward.remove(fact_id.as_bytes().as_slice())?;
                reverse.remove(key)?;
            }
            key
        };
        write_txn.commit()?;

        if let Some(key) = key {
            self.index.remove(key)?;
        }
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(Uuid, f32, u32)>, IndexError> {
        let matches = self.index.search(query, k)?;
        let read_txn = self.meta_db.begin_read()?;
        let reverse = read_txn.open_table(REVERSE_TABLE)?;

        let mut results = Vec::with_capacity(matches.keys.len());
        for (rank, (key, distance)) in matches.keys.iter().zip(matches.distances.iter()).enumerate() {
            let guard = reverse
                .get(*key)?
                .ok_or_else(|| IndexError::Corrupt(format!("usearch key {key} has no reverse mapping")))?;
            let fact_id_bytes: [u8; 16] = guard
                .value()
                .try_into()
                .map_err(|_| IndexError::Corrupt("reverse mapping value has wrong length".into()))?;
            results.push((Uuid::from_bytes(fact_id_bytes), *distance, rank as u32));
        }
        Ok(results)
    }

    pub fn watermark(&self) -> u64 {
        self.watermark
    }

    /// Records how far this index has been brought forward relative to the
    /// ledger (product doc §6.6's `last_applied_seq`). Not in the plan's
    /// original interface sketch, but recovery (a later task) needs it
    /// persisted, not just held in memory.
    pub fn set_watermark(&mut self, seq: u64) -> Result<(), IndexError> {
        self.watermark = seq;
        let write_txn = self.meta_db.begin_write()?;
        {
            let mut table = write_txn.open_table(META_TABLE)?;
            table.insert("watermark", seq.to_le_bytes().as_slice())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    pub fn fingerprint(&self) -> &ModelFingerprint {
        &self.fingerprint
    }
}
