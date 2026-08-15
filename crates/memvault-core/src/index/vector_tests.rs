use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use uuid::Uuid;

use crate::index::VectorIndex;
use crate::record::ModelFingerprint;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_index_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("memvault-vector-test-{tag}-{}-{n}.usearch", std::process::id()))
}

struct TempPath(PathBuf);
impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let mut meta = self.0.as_os_str().to_owned();
        meta.push(".meta.redb");
        let _ = std::fs::remove_file(PathBuf::from(meta));
    }
}

fn fingerprint() -> ModelFingerprint {
    ModelFingerprint {
        name: "test-model".into(),
        dimensions: 8,
        revision_hash: [1u8; 32],
    }
}

/// Deterministic pseudo-random unit vector so distinct fact_ids get
/// distinct, reproducible embeddings without pulling in a `rand` dep.
fn vector_for(seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..8)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn open_or_create_then_reopen_preserves_fingerprint_and_watermark() {
    let path = TempPath(temp_index_path("reopen"));
    {
        let mut index = VectorIndex::open_or_create(&path.0, &fingerprint()).unwrap();
        index.set_watermark(7).unwrap();
        index.insert(Uuid::from_u128(1), &vector_for(1)).unwrap();
    }

    let index = VectorIndex::open_or_create(&path.0, &fingerprint()).unwrap();
    assert_eq!(index.watermark(), 7);
    assert_eq!(index.fingerprint().name, "test-model");
    assert_eq!(index.fingerprint().dimensions, 8);
}

/// The plan's acceptance test: insert 100 vectors, remove 10, search for a
/// removed vector's near-neighbor (itself, at distance 0), assert it's
/// absent from results.
#[test]
fn test_insert_remove_search_roundtrip() {
    let path = TempPath(temp_index_path("roundtrip"));
    let mut index = VectorIndex::open_or_create(&path.0, &fingerprint()).unwrap();

    let ids: Vec<Uuid> = (0..100).map(|i| Uuid::from_u128(i as u128)).collect();
    for (i, id) in ids.iter().enumerate() {
        index.insert(*id, &vector_for(i as u64)).unwrap();
    }

    let removed = &ids[0..10];
    for id in removed {
        index.remove(*id).unwrap();
    }

    for (i, id) in removed.iter().enumerate() {
        let results = index.search(&vector_for(i as u64), 5).unwrap();
        assert!(
            !results.iter().any(|(found, _, _)| found == id),
            "removed fact_id {id} still returned by search"
        );
    }

    // A fact that was never removed is still findable as its own top hit.
    let still_present = ids[50];
    let results = index.search(&vector_for(50), 1).unwrap();
    assert_eq!(results[0].0, still_present);
}

#[test]
fn reset_discards_everything_and_accepts_a_new_dimensionality() {
    let path = TempPath(temp_index_path("reset"));
    let mut index = VectorIndex::open_or_create(&path.0, &fingerprint()).unwrap();
    index.insert(Uuid::from_u128(1), &vector_for(1)).unwrap();
    index.set_watermark(9).unwrap();

    let new_fingerprint = ModelFingerprint {
        name: "different-model".into(),
        dimensions: 16,
        revision_hash: [2u8; 32],
    };
    index.reset(&new_fingerprint).unwrap();

    assert_eq!(index.watermark(), 0);
    assert_eq!(index.fingerprint(), &new_fingerprint);
    // The old, lower-dimensional vector is gone; a query at the new
    // dimensionality finds nothing (nothing was ever inserted since reset).
    let results = index.search(&vec![0.0; 16], 5).unwrap();
    assert!(results.is_empty());
    // And inserting at the new dimensionality works.
    index.insert(Uuid::from_u128(2), &vec![1.0; 16]).unwrap();
}
