use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use chrono::Utc;
use uuid::Uuid;

use crate::ledger::Ledger;
use crate::record::{Assert, Encrypted, ModelFingerprint, NamespaceId, Payload, SourceRef};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A fresh path per test/call, in the OS temp dir -- no `tempfile` dependency
/// needed for something this small.
fn temp_ledger_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "memvault-ledger-test-{tag}-{}-{n}.redb",
        std::process::id()
    ))
}

struct TempPath(PathBuf);

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn fingerprint() -> ModelFingerprint {
    ModelFingerprint {
        name: "test-model".into(),
        dimensions: 4,
        revision_hash: [1u8; 32],
    }
}

fn assert_payload(tag: u64) -> Payload {
    Payload::Assert(Assert {
        fact_id: Uuid::from_u128(tag as u128),
        valid_from: Utc::now(),
        valid_to: None,
        content: Encrypted {
            nonce: [0u8; 12],
            ciphertext: tag.to_le_bytes().to_vec(),
        },
        content_hash: [0u8; 32],
        embedding: None,
        embedding_model: fingerprint(),
        keywords: vec![],
        pinned: false,
        source: SourceRef::default(),
    })
}

#[test]
fn append_read_round_trips() {
    let path = TempPath(temp_ledger_path("roundtrip"));
    let ledger = Ledger::open(&path.0).unwrap();

    let seq = ledger
        .append(NamespaceId("default".into()), Utc::now(), assert_payload(42))
        .unwrap();
    assert_eq!(seq, 0);
    assert_eq!(ledger.head().unwrap(), 1);

    let read_back = ledger.read(0).unwrap().unwrap();
    match read_back.payload {
        Payload::Assert(a) => assert_eq!(a.fact_id, Uuid::from_u128(42)),
        other => panic!("wrong payload kind: {other:?}"),
    }

    assert!(ledger.read(1).unwrap().is_none());
}

#[test]
fn append_extends_and_verifies_chain() {
    let path = TempPath(temp_ledger_path("chain"));
    let ledger = Ledger::open(&path.0).unwrap();

    for i in 0..50u64 {
        let seq = ledger
            .append(NamespaceId("default".into()), Utc::now(), assert_payload(i))
            .unwrap();
        assert_eq!(seq, i);
    }

    ledger.verify().unwrap();
    assert_eq!(ledger.head().unwrap(), 50);
}

/// Spawns a writer appending 1000 records and 4 reader threads scanning
/// concurrently. Asserts no reader ever observes a torn/partial record
/// (decode_record fails loudly on truncated bytes -- see record.rs) and
/// that every reader eventually observes seq 999.
#[test]
fn test_concurrent_read_during_write() {
    const N: u64 = 1000;

    let path = TempPath(temp_ledger_path("concurrent"));
    let ledger = Arc::new(Ledger::open(&path.0).unwrap());

    let writer = {
        let ledger = Arc::clone(&ledger);
        thread::spawn(move || {
            for i in 0..N {
                ledger
                    .append(NamespaceId("default".into()), Utc::now(), assert_payload(i))
                    .unwrap();
            }
        })
    };

    let readers: Vec<_> = (0..4)
        .map(|_| {
            let ledger = Arc::clone(&ledger);
            thread::spawn(move || {
                let mut max_seen: Option<u64> = None;
                loop {
                    let head = ledger.head().unwrap();
                    for record in ledger.scan_from(0).unwrap() {
                        let record = record.expect("reader observed a torn or corrupt record");
                        max_seen = Some(max_seen.map_or(record.header.seq, |m| m.max(record.header.seq)));
                    }
                    if head >= N {
                        break;
                    }
                    thread::yield_now();
                }
                max_seen
            })
        })
        .collect();

    writer.join().unwrap();
    for reader in readers {
        let max_seen = reader.join().unwrap();
        assert_eq!(max_seen, Some(N - 1), "reader did not eventually observe the last written seq");
    }
}
