//! Integration-style tests for chain construction and verification.

use chrono::{TimeZone, Utc};
use uuid::Uuid;

use crate::chain::{record_hash, verify_chain, verify_chain_from, ChainError, GENESIS_PREV_HASH};
use crate::record::{
    Assert, Encrypted, ModelFingerprint, NamespaceId, Payload, Record, SourceRef,
};

fn fingerprint() -> ModelFingerprint {
    ModelFingerprint {
        name: "test-model".into(),
        dimensions: 4,
        revision_hash: [7u8; 32],
    }
}

fn assert_payload(tag: u8) -> Payload {
    Payload::Assert(Assert {
        fact_id: Uuid::from_u128(tag as u128),
        valid_from: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
        valid_to: None,
        content: Encrypted {
            nonce: [tag; 12],
            ciphertext: vec![tag; 8],
        },
        content_hash: [tag; 32],
        embedding: Some(vec![0.1, 0.2, 0.3, 0.4]),
        embedding_model: fingerprint(),
        keywords: vec!["alpha".into(), "beta".into()],
        pinned: tag % 2 == 0,
        source: SourceRef(vec![tag]),
    })
}

/// Builds a chain of `n` Assert records with correctly-computed prev_hash
/// links, as a ledger writer would.
fn build_chain(n: u64) -> Vec<Record> {
    let namespace = NamespaceId("default".into());
    let mut records = Vec::with_capacity(n as usize);
    let mut prev_hash = GENESIS_PREV_HASH;

    for seq in 0..n {
        let recorded_at = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let record = Record::new(seq, prev_hash, recorded_at, namespace.clone(), assert_payload(seq as u8));
        prev_hash = record_hash(&record);
        records.push(record);
    }

    records
}

#[test]
fn valid_chain_verifies() {
    let chain = build_chain(20);
    assert_eq!(verify_chain(chain.into_iter()), Ok(()));
}

#[test]
fn empty_chain_is_trivially_valid() {
    assert_eq!(verify_chain(std::iter::empty()), Ok(()));
}

#[test]
fn test_chain_detects_tampering() {
    let mut chain = build_chain(20);

    // Flip one byte in record 7's content, without recomputing anything
    // downstream -- simulating on-disk corruption discovered at read time.
    match &mut chain[7].payload {
        Payload::Assert(a) => a.content.ciphertext[0] ^= 0xFF,
        _ => unreachable!(),
    }

    assert_eq!(verify_chain(chain.into_iter()), Err(ChainError::Diverged { seq: 7 }));
}

#[test]
fn corrupting_genesis_record_itself_is_detected() {
    let mut chain = build_chain(5);
    chain[0].header.prev_hash = [0xAB; 32];
    assert_eq!(verify_chain(chain.into_iter()), Err(ChainError::Diverged { seq: 0 }));
}

#[test]
fn missing_record_is_reported_as_non_sequential() {
    let mut chain = build_chain(5);
    chain.remove(2);
    assert_eq!(
        verify_chain(chain.into_iter()),
        Err(ChainError::NonSequential { expected: 2, found: 3 })
    );
}

#[test]
fn verify_chain_from_resumes_after_a_trusted_prefix() {
    let chain = build_chain(20);
    // Skip the first 10 records entirely; the suffix alone still verifies.
    assert_eq!(verify_chain_from(chain.into_iter().skip(10), 10), Ok(()));
}

#[test]
fn verify_chain_from_still_detects_tampering_in_the_suffix() {
    let mut chain = build_chain(20);
    match &mut chain[15].payload {
        Payload::Assert(a) => a.content.ciphertext[0] ^= 0xFF,
        _ => unreachable!(),
    }
    assert_eq!(verify_chain_from(chain.into_iter().skip(10), 10), Err(ChainError::Diverged { seq: 15 }));
}

#[test]
fn verify_chain_from_zero_still_checks_the_genesis_record() {
    let mut chain = build_chain(5);
    chain[0].header.prev_hash = [0xAB; 32];
    assert_eq!(verify_chain_from(chain.into_iter(), 0), Err(ChainError::Diverged { seq: 0 }));
}

#[test]
fn canonical_bytes_are_deterministic() {
    use crate::record::canonical_bytes;

    let chain_a = build_chain(3);
    let chain_b = build_chain(3);

    for (a, b) in chain_a.iter().zip(chain_b.iter()) {
        assert_eq!(canonical_bytes(a), canonical_bytes(b));
    }
}
