//! A stand-in for the embedding the caller is supposed to supply.
//!
//! ponytail: a real embedding model is out of scope by design (product doc
//! P5: no model calls on the hot path, no embedding model in the default
//! build). This hashes overlapping trigrams into a fixed-width vector so the
//! CLI, demos, and benchmarks have *something* to run ANN search over -- it
//! is not semantically meaningful, only enough to exercise fusion, decay,
//! and budget mechanics end to end. Anything reporting numbers produced with
//! it must say so. Upgrade path: callers supply real embeddings, which the
//! engine already accepts everywhere.

use crate::record::default_fingerprint;

/// What to call this when naming the model behind a published measurement.
pub const PLACEHOLDER_EMBEDDING_NAME: &str = "hashed-trigram placeholder (not a semantic model)";

pub fn placeholder_embedding(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; default_fingerprint().dimensions as usize];
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return v;
    }
    let window_len = 3.min(bytes.len());
    for window in bytes.windows(window_len) {
        let h = blake3::hash(window);
        let raw = h.as_bytes();
        let bucket = (raw[0] as usize) % v.len();
        let sign = if raw[1] % 2 == 0 { 1.0 } else { -1.0 };
        v[bucket] += sign;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
