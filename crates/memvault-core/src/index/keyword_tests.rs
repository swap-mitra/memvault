use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use uuid::Uuid;

use crate::index::KeywordIndex;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_index_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("memvault-keyword-test-{tag}-{}-{n}", std::process::id()))
}

struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
        let mut wm = self.0.as_os_str().to_owned();
        wm.push(".watermark");
        let _ = std::fs::remove_file(PathBuf::from(wm));
    }
}

/// The plan's acceptance test: index 3 short docs, one clearly on-topic,
/// assert it ranks first for a matching query.
#[test]
fn test_bm25_ranks_relevant_above_irrelevant() {
    let dir = TempDir(temp_index_dir("bm25"));
    let mut index = KeywordIndex::open_or_create(&dir.0).unwrap();

    let on_topic = Uuid::from_u128(1);
    let off_topic_a = Uuid::from_u128(2);
    let off_topic_b = Uuid::from_u128(3);

    index.insert(on_topic, "the ledger is hash-chained and append-only", &[]).unwrap();
    index.insert(off_topic_a, "bananas are a good source of potassium", &[]).unwrap();
    index.insert(off_topic_b, "the weather in march was unusually warm", &[]).unwrap();
    index.commit().unwrap();

    let results = index.search("hash-chained ledger", 3).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].0, on_topic, "on-topic document did not rank first");
}

#[test]
fn keywords_boost_ranking() {
    let dir = TempDir(temp_index_dir("boost"));
    let mut index = KeywordIndex::open_or_create(&dir.0).unwrap();

    let boosted = Uuid::from_u128(1);
    let plain = Uuid::from_u128(2);

    index.insert(boosted, "a document about something else entirely", &["memvault".into()]).unwrap();
    index.insert(plain, "memvault is mentioned once here in passing", &[]).unwrap();
    index.commit().unwrap();

    let results = index.search("memvault", 2).unwrap();
    assert_eq!(results[0].0, boosted, "keyword-boosted document did not outrank a plain content match");
}

#[test]
fn remove_then_commit_excludes_from_search() {
    let dir = TempDir(temp_index_dir("remove"));
    let mut index = KeywordIndex::open_or_create(&dir.0).unwrap();

    let fact_id = Uuid::from_u128(1);
    index.insert(fact_id, "a fact about removal semantics", &[]).unwrap();
    index.commit().unwrap();
    assert_eq!(index.search("removal semantics", 5).unwrap()[0].0, fact_id);

    index.remove(fact_id).unwrap();
    index.commit().unwrap();

    let results = index.search("removal semantics", 5).unwrap();
    assert!(results.is_empty(), "removed document still returned by search");
}

#[test]
fn watermark_persists_across_reopen() {
    let dir = TempDir(temp_index_dir("watermark"));
    {
        let mut index = KeywordIndex::open_or_create(&dir.0).unwrap();
        index.set_watermark(42).unwrap();
    }
    let index = KeywordIndex::open_or_create(&dir.0).unwrap();
    assert_eq!(index.watermark(), 42);
}
