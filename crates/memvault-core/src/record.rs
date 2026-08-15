//! The five ledger record types and their canonical byte encoding.
//!
//! Canonical encoding is hand-rolled rather than derived through a generic
//! serialization crate: the chain hash (see `chain.rs`) is computed over
//! these bytes and persisted forever, so the encoding is a wire format with
//! the same stability requirements as a file format, not an implementation
//! detail a derive macro's internals could silently change across a crate
//! upgrade. Field order below is the canonical order; changing it changes
//! every hash ever computed.

use chrono::{DateTime, Utc};
use uuid::Uuid;

/// A namespace groups facts under one decay/embedding-model configuration
/// (see product doc §6.9). Namespace names are caller-chosen strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NamespaceId(pub String);

/// Identifies the exact embedding model a vector was produced with, so a
/// query against a mismatched model is rejected rather than silently
/// searched (product doc §6.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFingerprint {
    pub name: String,
    pub dimensions: u32,
    pub revision_hash: [u8; 32],
}

/// Opaque caller-supplied provenance. MemVault stores and returns it
/// without interpreting it; callers decide what bytes go in.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceRef(pub Vec<u8>);

/// A ChaCha20-Poly1305-encrypted payload: nonce plus ciphertext-with-tag.
/// Encryption itself is not implemented here; this is the wire shape the
/// crypto layer fills in.
///
/// The product doc types the `Assert::content` field as the generic
/// `Encrypted<Vec<u8>>`, but every use in the spec is that one
/// instantiation — a type parameter with a single caller buys nothing, so
/// this is the concrete form instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encrypted {
    pub nonce: [u8; 12],
    pub ciphertext: Vec<u8>,
}

/// The five record kinds, stored redundantly as a header discriminant (see
/// `RecordHeader::kind`) so a scan can filter by kind without deserializing
/// the payload. Always derived from the payload by `Record::new` — never
/// set independently — so the two can't drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    Assert,
    Supersede,
    Erase,
    Retrieval,
    Checkpoint,
}

impl RecordKind {
    fn discriminant(self) -> u8 {
        match self {
            RecordKind::Assert => 0,
            RecordKind::Supersede => 1,
            RecordKind::Erase => 2,
            RecordKind::Retrieval => 3,
            RecordKind::Checkpoint => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordHeader {
    pub seq: u64,
    pub prev_hash: [u8; 32],
    pub recorded_at: DateTime<Utc>,
    pub namespace: NamespaceId,
    pub kind: RecordKind,
}

/// A new fact. Product doc §6.1.
#[derive(Debug, Clone, PartialEq)]
pub struct Assert {
    pub fact_id: Uuid,
    pub valid_from: DateTime<Utc>,
    pub valid_to: Option<DateTime<Utc>>,
    pub content: Encrypted,
    /// BLAKE3 over the plaintext, stored outside the encrypted envelope so
    /// it survives erasure (product doc §6.1, §6.5).
    pub content_hash: [u8; 32],
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: ModelFingerprint,
    pub keywords: Vec<String>,
    pub pinned: bool,
    pub source: SourceRef,
}

/// Closes a prior fact's open valid-time interval. Product doc §6.1
/// ("Supersede — closes a prior fact's valid interval") does not spell out
/// the payload; `target_seq` pins the exact `Assert` record being closed
/// (its `fact_id` alone isn't unique across supersession chains).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Supersede {
    pub fact_id: Uuid,
    pub target_seq: u64,
    pub valid_to: DateTime<Utc>,
    pub reason: Option<String>,
}

/// Records a cryptographic erase. Product doc §6.5: "append Erase record:
/// fact_id, ledger_seq of target, timestamp, reason" — timestamp is
/// `RecordHeader::recorded_at`, not duplicated here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Erase {
    pub fact_id: Uuid,
    pub target_seq: u64,
    pub reason: String,
}

/// Why a candidate did or didn't make it into a retrieval's results.
/// Product doc §6.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Injected,
    CutByBudget,
    CutByK,
    FilteredByTime,
}

impl Outcome {
    fn discriminant(self) -> u8 {
        match self {
            Outcome::Injected => 0,
            Outcome::CutByBudget => 1,
            Outcome::CutByK => 2,
            Outcome::FilteredByTime => 3,
        }
    }
}

/// Per-candidate scoring trail. Product doc §6.4, verbatim field set.
#[derive(Debug, Clone, PartialEq)]
pub struct Explanation {
    pub fact_id: Uuid,
    pub ledger_seq: u64,
    pub ann_rank: Option<u32>,
    pub ann_distance: Option<f32>,
    pub bm25_rank: Option<u32>,
    pub bm25_score: Option<f32>,
    pub rrf_score: f32,
    pub decay_weight: f32,
    pub final_score: f32,
    pub outcome: Outcome,
    pub token_cost: u32,
}

/// A past query and its full scoring trail, so `memvault explain
/// <retrieval_id>` (product doc §6.4) can reconstruct it exactly from the
/// ledger alone.
#[derive(Debug, Clone, PartialEq)]
pub struct Retrieval {
    pub retrieval_id: Uuid,
    pub query_text: Option<String>,
    pub query_embedding_model: Option<ModelFingerprint>,
    pub as_of: Option<DateTime<Utc>>,
    pub max_tokens: u32,
    pub k: u32,
    pub candidates: Vec<Explanation>,
}

/// A Merkle root over a contiguous seq range, for external anchoring
/// without exposing content (product doc §6.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub range_start_seq: u64,
    pub range_end_seq: u64,
    pub merkle_root: [u8; 32],
}

#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    Assert(Assert),
    Supersede(Supersede),
    Erase(Erase),
    Retrieval(Retrieval),
    Checkpoint(Checkpoint),
}

impl Payload {
    fn kind(&self) -> RecordKind {
        match self {
            Payload::Assert(_) => RecordKind::Assert,
            Payload::Supersede(_) => RecordKind::Supersede,
            Payload::Erase(_) => RecordKind::Erase,
            Payload::Retrieval(_) => RecordKind::Retrieval,
            Payload::Checkpoint(_) => RecordKind::Checkpoint,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub header: RecordHeader,
    pub payload: Payload,
}

impl Record {
    /// `header.kind` is derived from `payload`, not taken as a parameter,
    /// so the two can never disagree.
    pub fn new(
        seq: u64,
        prev_hash: [u8; 32],
        recorded_at: DateTime<Utc>,
        namespace: NamespaceId,
        payload: Payload,
    ) -> Self {
        let kind = payload.kind();
        Record {
            header: RecordHeader {
                seq,
                prev_hash,
                recorded_at,
                namespace,
                kind,
            },
            payload,
        }
    }
}

// --- canonical encoding -----------------------------------------------

struct Encoder(Vec<u8>);

impl Encoder {
    fn new() -> Self {
        Encoder(Vec::new())
    }

    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }

    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn bool(&mut self, v: bool) {
        self.u8(v as u8);
    }

    fn bytes(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.0.extend_from_slice(v);
    }

    fn str(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }

    fn fixed32(&mut self, v: &[u8; 32]) {
        self.0.extend_from_slice(v);
    }

    fn fixed12(&mut self, v: &[u8; 12]) {
        self.0.extend_from_slice(v);
    }

    fn uuid(&mut self, v: &Uuid) {
        self.0.extend_from_slice(v.as_bytes());
    }

    fn datetime(&mut self, v: &DateTime<Utc>) {
        self.i64(v.timestamp());
        self.u32(v.timestamp_subsec_nanos());
    }

    fn option<T>(&mut self, v: &Option<T>, write: impl FnOnce(&mut Self, &T)) {
        match v {
            Some(inner) => {
                self.bool(true);
                write(self, inner);
            }
            None => self.bool(false),
        }
    }

    fn model_fingerprint(&mut self, v: &ModelFingerprint) {
        self.str(&v.name);
        self.u32(v.dimensions);
        self.fixed32(&v.revision_hash);
    }

    fn finish(self) -> Vec<u8> {
        self.0
    }
}

/// Deterministic byte encoding of a record: fixed field order, explicit
/// little-endian widths, length-prefixed variable data. Same bytes in, same
/// bytes out, on any platform, forever — this is what `chain::record_hash`
/// hashes.
pub fn canonical_bytes(record: &Record) -> Vec<u8> {
    let mut e = Encoder::new();

    e.u64(record.header.seq);
    e.fixed32(&record.header.prev_hash);
    e.datetime(&record.header.recorded_at);
    e.str(&record.header.namespace.0);
    e.u8(record.header.kind.discriminant());

    match &record.payload {
        Payload::Assert(a) => {
            e.uuid(&a.fact_id);
            e.datetime(&a.valid_from);
            e.option(&a.valid_to, |e, v| e.datetime(v));
            e.fixed12(&a.content.nonce);
            e.bytes(&a.content.ciphertext);
            e.fixed32(&a.content_hash);
            e.option(&a.embedding, |e, v| {
                e.u64(v.len() as u64);
                for f in v {
                    e.f32(*f);
                }
            });
            e.model_fingerprint(&a.embedding_model);
            e.u64(a.keywords.len() as u64);
            for k in &a.keywords {
                e.str(k);
            }
            e.bool(a.pinned);
            e.bytes(&a.source.0);
        }
        Payload::Supersede(s) => {
            e.uuid(&s.fact_id);
            e.u64(s.target_seq);
            e.datetime(&s.valid_to);
            e.option(&s.reason, |e, v| e.str(v));
        }
        Payload::Erase(er) => {
            e.uuid(&er.fact_id);
            e.u64(er.target_seq);
            e.str(&er.reason);
        }
        Payload::Retrieval(r) => {
            e.uuid(&r.retrieval_id);
            e.option(&r.query_text, |e, v| e.str(v));
            e.option(&r.query_embedding_model, |e, v| e.model_fingerprint(v));
            e.option(&r.as_of, |e, v| e.datetime(v));
            e.u32(r.max_tokens);
            e.u32(r.k);
            e.u64(r.candidates.len() as u64);
            for c in &r.candidates {
                e.uuid(&c.fact_id);
                e.u64(c.ledger_seq);
                e.option(&c.ann_rank, |e, v| e.u32(*v));
                e.option(&c.ann_distance, |e, v| e.f32(*v));
                e.option(&c.bm25_rank, |e, v| e.u32(*v));
                e.option(&c.bm25_score, |e, v| e.f32(*v));
                e.f32(c.rrf_score);
                e.f32(c.decay_weight);
                e.f32(c.final_score);
                e.u8(c.outcome.discriminant());
                e.u32(c.token_cost);
            }
        }
        Payload::Checkpoint(cp) => {
            e.u64(cp.range_start_seq);
            e.u64(cp.range_end_seq);
            e.fixed32(&cp.merkle_root);
        }
    }

    e.finish()
}
