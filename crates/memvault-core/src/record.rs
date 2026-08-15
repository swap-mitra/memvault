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

    fn from_discriminant(v: u8) -> Result<Self, DecodeError> {
        match v {
            0 => Ok(RecordKind::Assert),
            1 => Ok(RecordKind::Supersede),
            2 => Ok(RecordKind::Erase),
            3 => Ok(RecordKind::Retrieval),
            4 => Ok(RecordKind::Checkpoint),
            other => Err(DecodeError::InvalidDiscriminant(other)),
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

    fn from_discriminant(v: u8) -> Result<Self, DecodeError> {
        match v {
            0 => Ok(Outcome::Injected),
            1 => Ok(Outcome::CutByBudget),
            2 => Ok(Outcome::CutByK),
            3 => Ok(Outcome::FilteredByTime),
            other => Err(DecodeError::InvalidDiscriminant(other)),
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

// --- canonical decoding -------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    UnexpectedEof,
    InvalidDiscriminant(u8),
    InvalidUtf8,
    InvalidTimestamp,
    TrailingBytes,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::UnexpectedEof => write!(f, "record bytes truncated"),
            DecodeError::InvalidDiscriminant(v) => write!(f, "invalid discriminant byte: {v}"),
            DecodeError::InvalidUtf8 => write!(f, "invalid utf-8 in record bytes"),
            DecodeError::InvalidTimestamp => write!(f, "timestamp out of range"),
            DecodeError::TrailingBytes => write!(f, "unconsumed bytes after decoding record"),
        }
    }
}

impl std::error::Error for DecodeError {}

struct Decoder<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Decoder { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::UnexpectedEof)?;
        let slice = self.bytes.get(self.pos..end).ok_or(DecodeError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn i64(&mut self) -> Result<i64, DecodeError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, DecodeError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn bool(&mut self) -> Result<bool, DecodeError> {
        Ok(self.u8()? != 0)
    }

    fn bytes(&mut self) -> Result<Vec<u8>, DecodeError> {
        let len = self.u64()? as usize;
        Ok(self.take(len)?.to_vec())
    }

    fn str(&mut self) -> Result<String, DecodeError> {
        String::from_utf8(self.bytes()?).map_err(|_| DecodeError::InvalidUtf8)
    }

    fn fixed32(&mut self) -> Result<[u8; 32], DecodeError> {
        Ok(self.take(32)?.try_into().unwrap())
    }

    fn fixed12(&mut self) -> Result<[u8; 12], DecodeError> {
        Ok(self.take(12)?.try_into().unwrap())
    }

    fn uuid(&mut self) -> Result<Uuid, DecodeError> {
        Ok(Uuid::from_bytes(self.take(16)?.try_into().unwrap()))
    }

    fn datetime(&mut self) -> Result<DateTime<Utc>, DecodeError> {
        let secs = self.i64()?;
        let nanos = self.u32()?;
        DateTime::from_timestamp(secs, nanos).ok_or(DecodeError::InvalidTimestamp)
    }

    fn option<T>(&mut self, read: impl FnOnce(&mut Self) -> Result<T, DecodeError>) -> Result<Option<T>, DecodeError> {
        if self.bool()? {
            Ok(Some(read(self)?))
        } else {
            Ok(None)
        }
    }

    fn model_fingerprint(&mut self) -> Result<ModelFingerprint, DecodeError> {
        Ok(ModelFingerprint {
            name: self.str()?,
            dimensions: self.u32()?,
            revision_hash: self.fixed32()?,
        })
    }
}

/// Inverse of [`canonical_bytes`]. Rejects any trailing bytes as
/// corruption rather than silently ignoring them (product doc §6.10:
/// malformed records are corruption, not something to shrug off).
pub fn decode_record(bytes: &[u8]) -> Result<Record, DecodeError> {
    let mut d = Decoder::new(bytes);

    let seq = d.u64()?;
    let prev_hash = d.fixed32()?;
    let recorded_at = d.datetime()?;
    let namespace = NamespaceId(d.str()?);
    let kind = RecordKind::from_discriminant(d.u8()?)?;

    let payload = match kind {
        RecordKind::Assert => Payload::Assert(Assert {
            fact_id: d.uuid()?,
            valid_from: d.datetime()?,
            valid_to: d.option(Decoder::datetime)?,
            content: Encrypted {
                nonce: d.fixed12()?,
                ciphertext: d.bytes()?,
            },
            content_hash: d.fixed32()?,
            embedding: d.option(|d| {
                let len = d.u64()? as usize;
                (0..len).map(|_| d.f32()).collect()
            })?,
            embedding_model: d.model_fingerprint()?,
            keywords: {
                let len = d.u64()? as usize;
                (0..len).map(|_| d.str()).collect::<Result<_, _>>()?
            },
            pinned: d.bool()?,
            source: SourceRef(d.bytes()?),
        }),
        RecordKind::Supersede => Payload::Supersede(Supersede {
            fact_id: d.uuid()?,
            target_seq: d.u64()?,
            valid_to: d.datetime()?,
            reason: d.option(Decoder::str)?,
        }),
        RecordKind::Erase => Payload::Erase(Erase {
            fact_id: d.uuid()?,
            target_seq: d.u64()?,
            reason: d.str()?,
        }),
        RecordKind::Retrieval => Payload::Retrieval(Retrieval {
            retrieval_id: d.uuid()?,
            query_text: d.option(Decoder::str)?,
            query_embedding_model: d.option(Decoder::model_fingerprint)?,
            as_of: d.option(Decoder::datetime)?,
            max_tokens: d.u32()?,
            k: d.u32()?,
            candidates: {
                let len = d.u64()? as usize;
                (0..len)
                    .map(|_| {
                        Ok(Explanation {
                            fact_id: d.uuid()?,
                            ledger_seq: d.u64()?,
                            ann_rank: d.option(Decoder::u32)?,
                            ann_distance: d.option(Decoder::f32)?,
                            bm25_rank: d.option(Decoder::u32)?,
                            bm25_score: d.option(Decoder::f32)?,
                            rrf_score: d.f32()?,
                            decay_weight: d.f32()?,
                            final_score: d.f32()?,
                            outcome: Outcome::from_discriminant(d.u8()?)?,
                            token_cost: d.u32()?,
                        })
                    })
                    .collect::<Result<_, DecodeError>>()?
            },
        }),
        RecordKind::Checkpoint => Payload::Checkpoint(Checkpoint {
            range_start_seq: d.u64()?,
            range_end_seq: d.u64()?,
            merkle_root: d.fixed32()?,
        }),
    };

    if d.pos != d.bytes.len() {
        return Err(DecodeError::TrailingBytes);
    }

    Ok(Record {
        header: RecordHeader {
            seq,
            prev_hash,
            recorded_at,
            namespace,
            kind,
        },
        payload,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(kind_source: &Payload) -> RecordHeader {
        RecordHeader {
            seq: 9,
            prev_hash: [3u8; 32],
            recorded_at: DateTime::from_timestamp(1_700_000_000, 123_456_789).unwrap(),
            namespace: NamespaceId("default".into()),
            kind: kind_source.kind(),
        }
    }

    fn round_trip(payload: Payload) {
        let record = Record {
            header: header(&payload),
            payload,
        };
        let bytes = canonical_bytes(&record);
        let decoded = decode_record(&bytes).expect("decode");
        assert_eq!(decoded, record);
    }

    #[test]
    fn assert_round_trips_with_embedding_and_keywords() {
        round_trip(Payload::Assert(Assert {
            fact_id: Uuid::from_u128(1),
            valid_from: DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
            valid_to: Some(DateTime::from_timestamp(1_700_100_000, 0).unwrap()),
            content: Encrypted {
                nonce: [9u8; 12],
                ciphertext: vec![1, 2, 3, 4, 5],
            },
            content_hash: [4u8; 32],
            embedding: Some(vec![0.1, -0.2, 0.3]),
            embedding_model: ModelFingerprint {
                name: "test-model".into(),
                dimensions: 3,
                revision_hash: [5u8; 32],
            },
            keywords: vec!["alpha".into(), "beta".into()],
            pinned: true,
            source: SourceRef(vec![7, 8, 9]),
        }));
    }

    #[test]
    fn assert_round_trips_without_embedding() {
        round_trip(Payload::Assert(Assert {
            fact_id: Uuid::from_u128(2),
            valid_from: DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
            valid_to: None,
            content: Encrypted {
                nonce: [0u8; 12],
                ciphertext: vec![],
            },
            content_hash: [0u8; 32],
            embedding: None,
            embedding_model: ModelFingerprint {
                name: "".into(),
                dimensions: 0,
                revision_hash: [0u8; 32],
            },
            keywords: vec![],
            pinned: false,
            source: SourceRef::default(),
        }));
    }

    #[test]
    fn supersede_round_trips() {
        round_trip(Payload::Supersede(Supersede {
            fact_id: Uuid::from_u128(3),
            target_seq: 4,
            valid_to: DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
            reason: Some("user correction".into()),
        }));
    }

    #[test]
    fn erase_round_trips() {
        round_trip(Payload::Erase(Erase {
            fact_id: Uuid::from_u128(5),
            target_seq: 6,
            reason: "gdpr request".into(),
        }));
    }

    #[test]
    fn retrieval_round_trips_with_all_outcome_variants() {
        let candidate = |outcome: Outcome, tag: u64| Explanation {
            fact_id: Uuid::from_u128(tag as u128),
            ledger_seq: tag,
            ann_rank: Some(1),
            ann_distance: Some(0.5),
            bm25_rank: None,
            bm25_score: None,
            rrf_score: 0.9,
            decay_weight: 1.0,
            final_score: 0.9,
            outcome,
            token_cost: 42,
        };

        round_trip(Payload::Retrieval(Retrieval {
            retrieval_id: Uuid::from_u128(10),
            query_text: Some("what changed".into()),
            query_embedding_model: Some(ModelFingerprint {
                name: "q-model".into(),
                dimensions: 8,
                revision_hash: [6u8; 32],
            }),
            as_of: Some(DateTime::from_timestamp(1_700_000_000, 0).unwrap()),
            max_tokens: 2048,
            k: 10,
            candidates: vec![
                candidate(Outcome::Injected, 1),
                candidate(Outcome::CutByBudget, 2),
                candidate(Outcome::CutByK, 3),
                candidate(Outcome::FilteredByTime, 4),
            ],
        }));
    }

    #[test]
    fn checkpoint_round_trips() {
        round_trip(Payload::Checkpoint(Checkpoint {
            range_start_seq: 0,
            range_end_seq: 999,
            merkle_root: [2u8; 32],
        }));
    }

    #[test]
    fn truncated_bytes_are_rejected() {
        let record = Record {
            header: header(&Payload::Checkpoint(Checkpoint {
                range_start_seq: 0,
                range_end_seq: 1,
                merkle_root: [0u8; 32],
            })),
            payload: Payload::Checkpoint(Checkpoint {
                range_start_seq: 0,
                range_end_seq: 1,
                merkle_root: [0u8; 32],
            }),
        };
        let mut bytes = canonical_bytes(&record);
        bytes.truncate(bytes.len() - 4);
        assert_eq!(decode_record(&bytes), Err(DecodeError::UnexpectedEof));
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let record = Record {
            header: header(&Payload::Checkpoint(Checkpoint {
                range_start_seq: 0,
                range_end_seq: 1,
                merkle_root: [0u8; 32],
            })),
            payload: Payload::Checkpoint(Checkpoint {
                range_start_seq: 0,
                range_end_seq: 1,
                merkle_root: [0u8; 32],
            }),
        };
        let mut bytes = canonical_bytes(&record);
        bytes.push(0xFF);
        assert_eq!(decode_record(&bytes), Err(DecodeError::TrailingBytes));
    }
}
