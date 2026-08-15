//! `memvault write/search/explain` -- the first demo checkpoint (see
//! docs/IMPLEMENTATION_PLAN.md §0.1, task P0-D1): a runnable, watchable
//! proof that hybrid retrieval, decay, pinning, and budget cuts all work
//! end to end, calling memvault-core directly with no server in between.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::Utc;
use clap::{Parser, Subcommand};
use uuid::Uuid;

use memvault_core::{
    explain, search, write_fact, Explanation, Indexes, KeywordIndex, Keyring, Ledger,
    ModelFingerprint, NamespaceId, Outcome, Query, SourceRef, VectorIndex, WriteInput,
};

const EMBEDDING_DIMENSIONS: u32 = 32;

#[derive(Parser)]
#[command(name = "memvault")]
struct Cli {
    /// Directory holding the ledger, keyring, and indexes.
    #[arg(long, default_value = "./memvault-data")]
    data_dir: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Assert a fact. Supplying --fact-id supersedes that fact's current version.
    Write {
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        content: String,
        #[arg(long)]
        pin: bool,
        #[arg(long = "fact-id")]
        fact_id: Option<Uuid>,
    },
    /// Hybrid search with full provenance for every candidate considered.
    Search {
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 10)]
        k: usize,
        #[arg(long = "max-tokens", default_value_t = 2048)]
        max_tokens: u32,
    },
    /// Reconstruct a past retrieval from the ledger by its id.
    Explain { retrieval_id: Uuid },
}

fn fingerprint() -> ModelFingerprint {
    ModelFingerprint {
        name: "demo-hash-embedding".into(),
        dimensions: EMBEDDING_DIMENSIONS,
        revision_hash: [0u8; 32],
    }
}

/// ponytail: a real embedding model is out of scope by design (product doc
/// P5: no model calls on the hot path, no embedding model in the default
/// build). This hashes overlapping trigrams into a fixed-width vector so
/// the demo has *something* to run ANN search over -- not semantically
/// meaningful, only enough to show fusion/decay/budget mechanics working
/// end to end. Upgrade path: accept a real embedding from a caller (a file
/// or stdin) once one wants to supply it; the engine already treats
/// embeddings as caller-supplied.
fn fake_embedding(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; EMBEDDING_DIMENSIONS as usize];
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

struct Stores {
    ledger: Ledger,
    keyring: Keyring,
    indexes: Indexes,
}

fn open_stores(data_dir: &Path) -> Result<Stores, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(data_dir)?;
    let ledger = Ledger::open(&data_dir.join("ledger.redb"))?;
    let keyring = Keyring::open(&data_dir.join("keys.redb"))?;
    let vector = VectorIndex::open_or_create(&data_dir.join("vectors.usearch"), &fingerprint())?;
    let keyword = KeywordIndex::open_or_create(&data_dir.join("keyword"))?;
    Ok(Stores { ledger, keyring, indexes: Indexes { vector, keyword } })
}

/// True only for a real terminal with color not explicitly disabled
/// (the NO_COLOR convention, https://no-color.org). Piping the CLI's
/// output -- as demo/run_demo_1.sh does, with `awk`/`grep` against exact
/// column text -- must never see escape codes, so this gates every color
/// call site rather than being applied unconditionally.
fn color_enabled() -> bool {
    std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none()
}

/// Wraps already-width-padded text in an ANSI SGR code, so alignment is
/// computed on the plain text first and the escape bytes never throw off
/// column widths.
fn colorize(padded_text: &str, sgr_code: &str, enabled: bool) -> String {
    if enabled {
        format!("\x1b[{sgr_code}m{padded_text}\x1b[0m")
    } else {
        padded_text.to_string()
    }
}

fn outcome_sgr_code(outcome: Outcome) -> &'static str {
    use Outcome::*;
    match outcome {
        Injected => "32",       // green: made it into the response
        CutByBudget | CutByK => "33", // yellow: relevant, but trimmed
        FilteredByTime => "2",  // dim: no longer valid at query time
    }
}

fn print_explanations(explanations: &[Explanation]) {
    let color = color_enabled();
    let header = format!(
        "{:<36} {:>8} {:>10} {:>8} {:>10} {:>9} {:>9} {:>9} {:>13} {:>6}",
        "fact_id", "ann_rank", "ann_dist", "bm25_rk", "bm25_score", "rrf", "decay_wt", "final", "outcome", "tokens"
    );
    println!("{}", colorize(&header, "1", color)); // bold

    for e in explanations {
        let outcome_padded = format!("{:>13}", format!("{:?}", e.outcome));
        let outcome_field = colorize(&outcome_padded, outcome_sgr_code(e.outcome), color);
        println!(
            "{:<36} {:>8} {:>10} {:>8} {:>10} {:>9.4} {:>9.4} {:>9.4} {outcome_field} {:>6}",
            e.fact_id,
            e.ann_rank.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
            e.ann_distance.map(|d| format!("{d:.4}")).unwrap_or_else(|| "-".into()),
            e.bm25_rank.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
            e.bm25_score.map(|s| format!("{s:.4}")).unwrap_or_else(|| "-".into()),
            e.rrf_score,
            e.decay_weight,
            e.final_score,
            e.token_cost,
        );
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Write { namespace, content, pin, fact_id } => {
            let mut stores = open_stores(&cli.data_dir)?;
            let embedding = fake_embedding(&content);
            let written_id = write_fact(
                &stores.ledger,
                &mut stores.indexes,
                &mut stores.keyring,
                WriteInput {
                    namespace: NamespaceId(namespace),
                    content: content.into_bytes(),
                    embedding: Some(embedding),
                    embedding_model: fingerprint(),
                    valid_from: Utc::now(),
                    valid_to: None,
                    fact_id,
                    keywords: vec![],
                    pinned: pin,
                    source: SourceRef::default(),
                },
            )?;
            println!("fact_id: {written_id}");
            if fact_id.is_some() {
                println!("(superseded the prior version of this fact)");
            }
        }
        Command::Search { namespace, query, k, max_tokens } => {
            let stores = open_stores(&cli.data_dir)?;
            let embedding = fake_embedding(&query);
            let (explanations, retrieval_id) = search(
                &stores.ledger,
                &stores.indexes,
                Query {
                    text: Some(query),
                    embedding: Some(embedding),
                    embedding_model: None,
                    namespace: NamespaceId(namespace),
                    as_of: None,
                    k,
                    max_tokens,
                },
            )?;
            println!("retrieval_id: {retrieval_id}");
            print_explanations(&explanations);
        }
        Command::Explain { retrieval_id } => {
            let stores = open_stores(&cli.data_dir)?;
            let explanations = explain(&stores.ledger, retrieval_id)?;
            print_explanations(&explanations);
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colorize_wraps_in_ansi_when_enabled() {
        assert_eq!(colorize("Injected", "32", true), "\x1b[32mInjected\x1b[0m");
    }

    #[test]
    fn colorize_is_a_no_op_when_disabled() {
        assert_eq!(colorize("Injected", "32", false), "Injected");
    }

    #[test]
    fn each_outcome_has_a_distinct_sgr_code_except_the_two_cut_variants() {
        assert_eq!(outcome_sgr_code(Outcome::Injected), "32");
        assert_eq!(outcome_sgr_code(Outcome::CutByBudget), outcome_sgr_code(Outcome::CutByK));
        assert_ne!(outcome_sgr_code(Outcome::Injected), outcome_sgr_code(Outcome::FilteredByTime));
    }
}
