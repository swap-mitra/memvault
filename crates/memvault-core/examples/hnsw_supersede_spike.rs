//! P0-0 risk spike (see docs/IMPLEMENTATION_PLAN.md): does usearch's HNSW
//! tombstone accumulation degrade recall/latency under the supersession-heavy
//! churn a bitemporal store produces, and does periodic `compact()` fix it
//! cheaply enough to be a real mitigation?
//!
//! Throwaway: not part of the shipped crate, not wired into P0-4. Run with
//! `cargo run --release --example hnsw_supersede_spike -p memvault-core`.
//!
//! Method: two regimes share an identical random operation schedule --
//! insert an initial corpus, then repeatedly "supersede" a fraction of the
//! live set (remove old key, insert a new one), one regime never compacts,
//! the other compacts every few cycles. At each checkpoint, recall@k is
//! measured as agreement between the index's own approximate `search` and
//! its exact brute-force `exact_search` (no second ground-truth index
//! needed), plus mean approximate-search latency and raw graph stats.

use std::collections::HashSet;
use std::time::Instant;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

const DIMENSIONS: usize = 32;
const INITIAL_VECTORS: usize = 3000;
const CYCLES: usize = 25;
const MEASURE_EVERY: usize = 5;
const SUPERSEDE_FRACTION: f64 = 0.08;
const QUERY_SAMPLE: usize = 50;
const K: usize = 10;
const COMPACT_EVERY: usize = 5;

/// splitmix64 -- enough determinism for a reproducible spike, no `rand` dep.
struct Rng(u64);

impl Rng {
    fn seed(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn next_index(&mut self, len: usize) -> usize {
        (self.next_u64() % len as u64) as usize
    }

    fn vector(&mut self) -> Vec<f32> {
        (0..DIMENSIONS).map(|_| self.next_f32() * 2.0 - 1.0).collect()
    }
}

fn options() -> IndexOptions {
    IndexOptions {
        dimensions: DIMENSIONS,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: 0,
        expansion_add: 0,
        expansion_search: 0,
        multi: false,
    }
}

/// recall@K = mean fraction of the exact top-K present in the approximate
/// top-K, averaged over `QUERY_SAMPLE` fresh random queries.
fn measure(index: &Index, rng: &mut Rng) -> (f64, f64) {
    let mut recall_sum = 0.0;
    let mut latency_sum_us = 0.0;

    for _ in 0..QUERY_SAMPLE {
        let query = rng.vector();

        let t0 = Instant::now();
        let approx = index.search(&query, K).expect("approx search");
        latency_sum_us += t0.elapsed().as_secs_f64() * 1_000_000.0;

        let exact = index.exact_search(&query, K).expect("exact search");
        let exact_keys: HashSet<u64> = exact.keys.into_iter().collect();
        let hits = approx.keys.iter().filter(|k| exact_keys.contains(k)).count();
        recall_sum += hits as f64 / K as f64;
    }

    (recall_sum / QUERY_SAMPLE as f64, latency_sum_us / QUERY_SAMPLE as f64)
}

fn run_regime(name: &str, compact_every: Option<usize>) {
    println!("\n=== regime: {name} ===");
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>10} {:>14} {:>14}",
        "cycle", "live_size", "graph_nodes", "graph_edges", "recall@10", "mean_us", "compact_ms"
    );

    let index = Index::new(&options()).expect("create index");
    index.reserve(INITIAL_VECTORS * 4).expect("reserve");

    let mut rng = Rng::seed(42);
    let mut keys: Vec<u64> = Vec::with_capacity(INITIAL_VECTORS * 2);
    let mut next_key: u64 = 0;

    for _ in 0..INITIAL_VECTORS {
        let key = next_key;
        next_key += 1;
        let v = rng.vector();
        index.add(key, &v).expect("initial add");
        keys.push(key);
    }

    let (recall, latency) = measure(&index, &mut rng);
    let stats = index.stats();
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>10.3} {:>14.1} {:>14}",
        0,
        index.size(),
        stats.nodes,
        stats.edges,
        recall,
        latency,
        "-"
    );

    for cycle in 1..=CYCLES {
        let removals = ((keys.len() as f64) * SUPERSEDE_FRACTION).round() as usize;
        for _ in 0..removals.min(keys.len()) {
            let idx = rng.next_index(keys.len());
            let old_key = keys.swap_remove(idx);
            index.remove(old_key).expect("remove superseded");

            let new_key = next_key;
            next_key += 1;
            let v = rng.vector();
            index.add(new_key, &v).expect("add superseding fact");
            keys.push(new_key);
        }

        let mut compact_ms = None;
        if let Some(every) = compact_every {
            if cycle % every == 0 {
                let t0 = Instant::now();
                index.compact().expect("compact");
                compact_ms = Some(t0.elapsed().as_secs_f64() * 1000.0);
            }
        }

        if cycle % MEASURE_EVERY == 0 {
            let (recall, latency) = measure(&index, &mut rng);
            let stats = index.stats();
            println!(
                "{:>6} {:>10} {:>12} {:>12} {:>10.3} {:>14.1} {:>14}",
                cycle,
                index.size(),
                stats.nodes,
                stats.edges,
                recall,
                latency,
                compact_ms.map(|ms| format!("{ms:.1}")).unwrap_or_else(|| "-".into())
            );
        }
    }

    // Rebuild-from-scratch baseline: what a fresh index over the same final
    // live set costs and performs like, as the ceiling this regime is
    // chasing (or not) after CYCLES rounds of churn.
    let t0 = Instant::now();
    let fresh = Index::new(&options()).expect("create fresh index");
    fresh.reserve(keys.len()).expect("reserve fresh");
    for &key in &keys {
        // Content doesn't matter for this comparison, only graph shape.
        let v = rng.vector();
        fresh.add(key, &v).expect("fresh add");
    }
    let rebuild_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (fresh_recall, fresh_latency) = measure(&fresh, &mut rng);
    println!(
        "rebuild-from-scratch: {} vectors in {:.1}ms, recall@10={:.3}, mean_us={:.1}",
        fresh.size(),
        rebuild_ms,
        fresh_recall,
        fresh_latency
    );
}

fn main() {
    println!(
        "P0-0 spike: {INITIAL_VECTORS} initial vectors, {CYCLES} supersession cycles, \
         {SUPERSEDE_FRACTION:.2} churn/cycle, dims={DIMENSIONS}"
    );

    run_regime("never compact", None);
    run_regime(&format!("compact every {COMPACT_EVERY} cycles"), Some(COMPACT_EVERY));

    println!(
        "\nCompare the two tables above: does 'never compact' recall@10 or \
         mean_us drift measurably worse than the rebuild-from-scratch baseline \
         by the final row? Does periodic compact() hold it near baseline, and \
         at what compact_ms cost relative to rebuild time? Write the answer \
         into the P0-0 task's PR description per docs/IMPLEMENTATION_PLAN.md."
    );
}
