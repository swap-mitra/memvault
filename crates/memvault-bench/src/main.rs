//! The latency benchmark protocol from product doc §7 (plan task P2-5).
//!
//! Three claims, each printed with the protocol that produced it: retrieval
//! latency (index-only and end-to-end, p50/p95/p99), chain verification
//! throughput, and full index rebuild time. Section 7's own rule is that a
//! number published without its protocol is marketing, so the protocol block
//! is not optional output and every figure carries a unit.
//!
//! Nothing here is compared against a threshold. This harness reports; it
//! does not pass or fail.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;
use memvault_core::{
    default_fingerprint, placeholder_embedding, recover, search, write_fact, Indexes, KeywordIndex,
    Keyring, Ledger, NamespaceId, Query, RecoveryConfig, SourceRef, VectorIndex, WriteInput,
    PLACEHOLDER_EMBEDDING_NAME,
};

const NAMESPACE: &str = "bench";

#[derive(Parser)]
#[command(name = "memvault-bench", about = "Latency, verification, and rebuild benchmarks under the product doc §7 protocol")]
struct Args {
    /// Number of facts written into the corpus before measuring.
    #[arg(long, default_value_t = 10_000)]
    corpus_size: u64,

    /// Number of queries timed per latency figure.
    #[arg(long, default_value_t = 1_000)]
    queries: u64,

    #[arg(long, default_value_t = 10)]
    k: usize,

    #[arg(long = "max-tokens", default_value_t = 2048)]
    max_tokens: u32,

    /// Free-text machine description. §7 requires stating the hardware, and
    /// only the operator knows what it actually is; the auto-detected
    /// default names the platform but not the CPU.
    #[arg(long)]
    hardware: Option<String>,

    /// Where to build the corpus. Defaults to a fresh temp directory that is
    /// removed afterwards.
    #[arg(long = "data-dir")]
    data_dir: Option<PathBuf>,
}

/// Nearest-rank percentile over an already-sorted slice. Textbook definition
/// rather than an interpolating one, so p99 of 100 samples is a sample that
/// was actually observed and not an average of two.
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    assert!(!sorted.is_empty(), "percentile of an empty sample");
    let rank = (p / 100.0 * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank - 1]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn print_latency(label: &str, protocol: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    println!("\n{label}");
    println!("  protocol             {protocol}");
    println!("  samples              {} queries", samples.len());
    for p in [50.0, 95.0, 99.0] {
        println!("  p{:<19.0} {:.3} ms", p, ms(percentile(&samples, p)));
    }
}

fn detected_hardware() -> String {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    format!(
        "{}/{}, {cores} logical cores (CPU model not detected -- pass --hardware)",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

/// Enough lexical variety that BM25 has something to discriminate on, and
/// enough repetition that queries hit. Deterministic, so two runs on the
/// same machine are comparable.
fn corpus_text(i: u64) -> String {
    const SUBJECTS: [&str; 8] = ["deploy script", "staging database", "api gateway", "billing job", "search index", "auth service", "cache layer", "metrics pipeline"];
    const PREDICATES: [&str; 6] = ["lives in", "was migrated to", "is owned by", "depends on", "was rewritten in", "is monitored by"];
    const OBJECTS: [&str; 8] = ["ops/deploy.sh", "postgres 16", "the platform team", "the shared queue", "rust", "the on-call rotation", "redis", "a nightly cron"];
    format!(
        "fact {i}: the {} {} {}",
        SUBJECTS[(i % 8) as usize],
        PREDICATES[(i % 6) as usize],
        OBJECTS[((i / 8) % 8) as usize]
    )
}

fn query_text(i: u64) -> String {
    const SUBJECTS: [&str; 8] = ["deploy script", "staging database", "api gateway", "billing job", "search index", "auth service", "cache layer", "metrics pipeline"];
    SUBJECTS[(i % 8) as usize].to_string()
}

fn open(data_dir: &std::path::Path) -> Result<(Ledger, Keyring, Indexes), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(data_dir)?;
    let ledger = Ledger::open(&data_dir.join("ledger.redb"))?;
    let keyring = Keyring::open(&data_dir.join("keys.redb"))?;
    let vector = VectorIndex::open_or_create(&data_dir.join("vectors.usearch"), &default_fingerprint())?;
    let keyword = KeywordIndex::open_or_create(&data_dir.join("keyword"))?;
    Ok((ledger, keyring, Indexes { vector, keyword }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let scratch = args.data_dir.clone().unwrap_or_else(|| {
        std::env::temp_dir().join(format!("memvault-bench-{}", std::process::id()))
    });
    let owns_scratch = args.data_dir.is_none();

    let fingerprint = default_fingerprint();
    let (ledger, mut keyring, mut indexes) = open(&scratch)?;

    // --- corpus ---------------------------------------------------------
    let ingest_start = Instant::now();
    for i in 0..args.corpus_size {
        let content = corpus_text(i);
        write_fact(
            &ledger,
            &mut indexes,
            &mut keyring,
            WriteInput {
                namespace: NamespaceId(NAMESPACE.into()),
                embedding: Some(placeholder_embedding(&content)),
                content: content.into_bytes(),
                embedding_model: fingerprint.clone(),
                valid_from: chrono::Utc::now(),
                valid_to: None,
                fact_id: None,
                keywords: vec![],
                pinned: false,
                source: SourceRef::default(),
            },
        )?;
    }
    let ingest = ingest_start.elapsed();

    println!("memvault benchmark report");
    println!("\nprotocol");
    println!("  corpus_size          {} records", args.corpus_size);
    println!("  embedding_dimensions {} dimensions", fingerprint.dimensions);
    println!("  embedding_model      {PLACEHOLDER_EMBEDDING_NAME}");
    println!("  k                    {} candidates", args.k);
    println!("  max_tokens           {} tokens", args.max_tokens);
    println!("  hardware             {}", args.hardware.clone().unwrap_or_else(detected_hardware));
    println!("  build_profile        {}", if cfg!(debug_assertions) { "debug (NOT a publishable figure -- build with --release)" } else { "release" });
    println!("  ingest               {:.3} s for {} records", ingest.as_secs_f64(), args.corpus_size);

    // --- retrieval latency ----------------------------------------------
    // Index-only: the embedding is computed outside the timed region, so
    // this measures fusion over a warm index and nothing else.
    let mut index_only = Vec::with_capacity(args.queries as usize);
    for i in 0..args.queries {
        let text = query_text(i);
        let embedding = placeholder_embedding(&text);
        let start = Instant::now();
        search(
            &ledger,
            &indexes,
            Query {
                text: Some(text),
                embedding: Some(embedding),
                embedding_model: None,
                namespace: NamespaceId(NAMESPACE.into()),
                as_of: None,
                k: args.k,
                max_tokens: args.max_tokens,
            },
        )?;
        index_only.push(start.elapsed());
    }
    print_latency(
        "retrieval latency, index-only",
        "ANN + BM25 + RRF + decay + budget over a warm index; embedding computed outside the timed region",
        index_only,
    );

    let mut end_to_end = Vec::with_capacity(args.queries as usize);
    for i in 0..args.queries {
        let text = query_text(i);
        let start = Instant::now();
        let embedding = placeholder_embedding(&text);
        search(
            &ledger,
            &indexes,
            Query {
                text: Some(text),
                embedding: Some(embedding),
                embedding_model: None,
                namespace: NamespaceId(NAMESPACE.into()),
                as_of: None,
                k: args.k,
                max_tokens: args.max_tokens,
            },
        )?;
        end_to_end.push(start.elapsed());
    }
    print_latency(
        "retrieval latency, end-to-end",
        "index-only plus embedding the query text with the model named above",
        end_to_end,
    );

    // --- verification throughput ----------------------------------------
    let head = ledger.head()?;
    let verify_start = Instant::now();
    ledger.verify()?;
    let verify = verify_start.elapsed();
    println!("\nchain verification");
    // Higher than corpus_size: every search above appended a Retrieval
    // record, and verification walks the whole chain.
    println!("  protocol             full BLAKE3 chain walk from genesis to head, single-threaded; the chain holds retrieval records too, so this exceeds corpus_size");
    println!("  records              {head} records");
    println!("  elapsed              {:.3} s", verify.as_secs_f64());
    println!("  throughput           {:.1} records/s", head as f64 / verify.as_secs_f64());

    // --- index rebuild --------------------------------------------------
    // Reset one index at a time: replay skips records already at the other
    // index's watermark, so each figure is that index alone.
    println!("\nindex rebuild, full, from the ledger");
    println!("  protocol             reset the index, then replay every ledger record into it");
    println!("  corpus_size          {} records", args.corpus_size);

    indexes.vector.reset(&fingerprint)?;
    let start = Instant::now();
    recover(&ledger, &mut indexes, &keyring, &fingerprint, RecoveryConfig { verify_chain: false })?;
    let vector_rebuild = start.elapsed();
    println!("  vector (HNSW)        {:.3} s", vector_rebuild.as_secs_f64());

    indexes.keyword.reset()?;
    let start = Instant::now();
    recover(&ledger, &mut indexes, &keyring, &fingerprint, RecoveryConfig { verify_chain: false })?;
    let keyword_rebuild = start.elapsed();
    println!("  keyword (tantivy)    {:.3} s", keyword_rebuild.as_secs_f64());
    println!("  total                {:.3} s", (vector_rebuild + keyword_rebuild).as_secs_f64());

    drop(indexes);
    drop(keyring);
    drop(ledger);
    if owns_scratch {
        std::fs::remove_dir_all(&scratch).ok();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_percentile_picks_an_observed_sample() {
        let sorted: Vec<Duration> = (1..=100).map(|n| Duration::from_millis(n)).collect();
        assert_eq!(percentile(&sorted, 50.0), Duration::from_millis(50));
        assert_eq!(percentile(&sorted, 95.0), Duration::from_millis(95));
        assert_eq!(percentile(&sorted, 99.0), Duration::from_millis(99));
    }

    #[test]
    fn percentile_of_a_single_sample_is_that_sample() {
        let sorted = vec![Duration::from_millis(7)];
        for p in [50.0, 95.0, 99.0] {
            assert_eq!(percentile(&sorted, p), Duration::from_millis(7));
        }
    }
}
