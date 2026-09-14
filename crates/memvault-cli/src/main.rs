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
    erase, explain, explanation_row, get_fact, injected_contents, memory_as_of, open_stores, outcome_cell,
    placeholder_embedding, recover, search, supersede_fact, write_fact_with_limit, AsOfQuery, Explanation,
    NamespaceId, Outcome, Payload, Query, RecoveryConfig, SourceRef, Stores, WriteInput, EXPLANATION_HEADER,
};

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
    /// Assert a fact. Supplying --fact-id supersedes that fact's current
    /// version; the fact must belong to --namespace.
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
    /// Read one fact's current version by id.
    Get { fact_id: Uuid },
    /// Print the MCP client config for this data directory, with absolute
    /// paths filled in, ready to paste into .mcp.json or
    /// claude_desktop_config.json.
    McpConfig {
        /// OpenAI-compatible embeddings base URL the server should call,
        /// e.g. http://localhost:11434/v1 for Ollama.
        #[arg(long = "embed-url", requires = "embed_model")]
        embed_url: Option<String>,
        /// Embedding model name at that URL, e.g. nomic-embed-text.
        #[arg(long = "embed-model", requires = "embed_url")]
        embed_model: Option<String>,
        /// Embedding width for a new directory when the client supplies its
        /// own vectors instead.
        #[arg(long = "embedding-dim")]
        embedding_dim: Option<u32>,
    },
    /// Point-in-time query: what was true, or what was believed true, as
    /// of a given moment (RFC 3339, e.g. 2026-01-01T00:00:00Z). Omitting
    /// either bound means "now" on that axis.
    AsOf {
        #[arg(long)]
        namespace: String,
        #[arg(long = "valid-time")]
        valid_time: Option<chrono::DateTime<Utc>>,
        #[arg(long = "transaction-time")]
        transaction_time: Option<chrono::DateTime<Utc>>,
    },
    /// Close a fact's open interval without asserting a replacement.
    Supersede {
        fact_id: Uuid,
        #[arg(long = "valid-to")]
        valid_to: Option<chrono::DateTime<Utc>>,
        #[arg(long)]
        reason: Option<String>,
    },
    /// Cryptographically erase a fact: destroy its key so its content is
    /// permanently unreadable, everywhere, forever. The ledger record
    /// stays -- only its content becomes unrecoverable.
    Forget {
        fact_id: Uuid,
        #[arg(long)]
        reason: String,
    },
    /// Verify both hash chains: facts, and retrievals. Exits non-zero and
    /// reports the first diverging seq if either is broken.
    Verify {
        /// Trust everything in the facts chain before this seq and verify
        /// only from here forward. Defaults to the genesis.
        #[arg(long, default_value_t = 0)]
        from: u64,
    },
    /// Rebuild the vector/keyword indexes from the ledger.
    Replay,
    /// Drop old retrieval records from the front of the retrievals chain.
    /// The newest always stays, and the chain still verifies from the
    /// first record kept. Pruned searches can no longer be explained.
    Prune {
        /// Prune retrievals older than this many days. Defaults to
        /// `[retrievals] keep_days` from memvault.toml.
        #[arg(long = "keep-days", conflicts_with = "before")]
        keep_days: Option<u32>,
        /// Prune retrievals recorded before this instant (RFC 3339).
        #[arg(long)]
        before: Option<chrono::DateTime<Utc>>,
    },
    /// Print the effective memvault.toml for this data directory, defaults
    /// filled in.
    Config,
    /// Debug: print a raw ledger record by seq, attempting to decrypt an
    /// Assert's content so an erased fact visibly shows as undecryptable
    /// rather than simply omitted.
    DumpRecord { seq: u64 },
}

/// The data directory decides its embedding width: the CLI never asks for
/// one, so a new directory gets the default and an existing one keeps what
/// it has. The stand-in embeddings are sized to whatever that turns out to
/// be.
fn open(data_dir: &Path) -> Result<Stores, Box<dyn std::error::Error>> {
    open_stores(data_dir, None)
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

fn print_fact(f: &memvault_core::AsOfFact) {
    let content = String::from_utf8_lossy(&f.content);
    let valid_to = f.valid_to.map(|t| t.to_rfc3339()).unwrap_or_else(|| "open".into());
    println!("{} [{} .. {}] {}", f.fact_id, f.valid_from.to_rfc3339(), valid_to, content);
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The `mcpServers` block a client wants, by hand rather than through a
/// JSON crate: four keys, and the shape is fixed by the MCP clients.
fn mcp_config_json(server: &Path, data_dir: &Path, env: &[(&str, String)]) -> String {
    let mut out = String::new();
    out.push_str("{\n  \"mcpServers\": {\n    \"memvault\": {\n");
    out.push_str(&format!("      \"command\": {},\n", json_string(&server.to_string_lossy())));
    out.push_str(&format!("      \"args\": [{}]", json_string(&data_dir.to_string_lossy())));
    if !env.is_empty() {
        out.push_str(",\n      \"env\": {\n");
        let entries: Vec<String> = env.iter().map(|(k, v)| format!("        {}: {}", json_string(k), json_string(v))).collect();
        out.push_str(&entries.join(",\n"));
        out.push_str("\n      }");
    }
    out.push_str("\n    }\n  }\n}");
    out
}

fn print_explanations(explanations: &[Explanation]) {
    let color = color_enabled();
    println!("{}", colorize(EXPLANATION_HEADER, "1", color)); // bold

    for e in explanations {
        // Colour the padded cell, never the row: the escape bytes would
        // otherwise count toward the column width.
        let outcome = colorize(&outcome_cell(e), outcome_sgr_code(e.outcome), color);
        println!("{}", explanation_row(e, &outcome));
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Write { namespace, content, pin, fact_id } => {
            let mut stores = open(&cli.data_dir)?;
            let fingerprint = stores.indexes.vector.fingerprint().clone();
            let embedding = placeholder_embedding(&content, fingerprint.dimensions);
            let written_id = write_fact_with_limit(
                &stores.ledger,
                &mut stores.indexes,
                &mut stores.keyring,
                WriteInput {
                    namespace: NamespaceId(namespace),
                    content: content.into_bytes(),
                    embedding: Some(embedding),
                    embedding_model: fingerprint,
                    valid_from: Utc::now(),
                    valid_to: None,
                    fact_id,
                    keywords: vec![],
                    pinned: pin,
                    source: SourceRef::default(),
                },
                stores.config.limits.max_content_bytes,
            )?;
            println!("fact_id: {written_id}");
            if fact_id.is_some() {
                println!("(superseded the prior version of this fact)");
            }
        }
        Command::Search { namespace, query, k, max_tokens } => {
            let stores = open(&cli.data_dir)?;
            let embedding = placeholder_embedding(&query, stores.indexes.vector.fingerprint().dimensions);
            let (explanations, retrieval_id) = search(
                &stores.ledger,
                &stores.indexes,
                &stores.keyring,
                Query {
                    text: Some(query),
                    embedding: Some(embedding),
                    embedding_model: None,
                    decay: stores.config.decay_for(&NamespaceId(namespace.clone())),
                    namespace: NamespaceId(namespace),
                    as_of: None,
                    k,
                    max_tokens,
                },
            )?;
            println!("retrieval_id: {retrieval_id}");
            print_explanations(&explanations);
            // Indented so the table rows stay the only lines that start
            // with a fact_id (demo/run_demo_1.sh counts them that way).
            let injected = injected_contents(&stores.ledger, &stores.keyring, &explanations)?;
            if !injected.is_empty() {
                println!("injected:");
                for f in &injected {
                    println!("  {}  {}", f.fact_id, String::from_utf8_lossy(&f.content));
                }
            }
        }
        Command::Explain { retrieval_id } => {
            let stores = open(&cli.data_dir)?;
            let explanations = explain(&stores.ledger, retrieval_id)?;
            print_explanations(&explanations);
        }
        Command::AsOf { namespace, valid_time, transaction_time } => {
            let stores = open(&cli.data_dir)?;
            let facts = memory_as_of(&stores.ledger, &stores.keyring, &NamespaceId(namespace), AsOfQuery { valid_time, transaction_time })?;
            for f in &facts {
                print_fact(f);
            }
        }
        Command::Get { fact_id } => {
            let stores = open(&cli.data_dir)?;
            let fact = get_fact(&stores.ledger, &stores.keyring, fact_id)?.ok_or_else(|| format!("no open version of fact {fact_id}"))?;
            print_fact(&fact);
        }
        Command::McpConfig { embed_url, embed_model, embedding_dim } => {
            // The server binary ships next to this one, in a release
            // archive and in target/ alike.
            let server = std::env::current_exe()?.with_file_name(format!("memvault-server{}", std::env::consts::EXE_SUFFIX));
            if !server.exists() {
                eprintln!("warning: {} does not exist yet; build or download memvault-server next to this binary", server.display());
            }
            let data_dir = std::path::absolute(&cli.data_dir)?;
            let mut env = Vec::new();
            if let Some(url) = embed_url {
                env.push(("MEMVAULT_EMBED_URL", url));
            }
            if let Some(model) = embed_model {
                env.push(("MEMVAULT_EMBED_MODEL", model));
            }
            if let Some(dim) = embedding_dim {
                env.push(("MEMVAULT_EMBEDDING_DIM", dim.to_string()));
            }
            println!("{}", mcp_config_json(&server, &data_dir, &env));
        }
        Command::Supersede { fact_id, valid_to, reason } => {
            let mut stores = open(&cli.data_dir)?;
            let valid_to = valid_to.unwrap_or_else(Utc::now);
            supersede_fact(&stores.ledger, &mut stores.indexes, fact_id, valid_to, reason)?;
            println!("superseded fact_id: {fact_id}");
        }
        Command::Forget { fact_id, reason } => {
            let mut stores = open(&cli.data_dir)?;
            erase(&stores.ledger, &mut stores.keyring, &mut stores.indexes, fact_id, reason)?;
            println!("forgot fact_id: {fact_id}");
        }
        Command::Verify { from } => {
            let stores = open(&cli.data_dir)?;
            stores.ledger.verify_from(from)?;
            println!("chain verified from seq {from}");
            let first = stores.ledger.retrievals_first_seq()?.unwrap_or(0);
            stores.ledger.verify_retrievals()?;
            println!("retrievals chain verified from seq {first}");
        }
        Command::Prune { keep_days, before } => {
            let stores = open(&cli.data_dir)?;
            let cutoff = match (before, keep_days.or(stores.config.retrievals.keep_days)) {
                (Some(before), _) => before,
                (None, Some(days)) => Utc::now() - chrono::Duration::days(i64::from(days)),
                (None, None) => return Err("nothing to prune by: pass --keep-days or --before, or set [retrievals] keep_days in memvault.toml".into()),
            };
            let outcome = stores.ledger.prune_retrievals_before(cutoff)?;
            println!(
                "pruned {} retrieval records recorded before {}; retrievals chain now starts at seq {}",
                outcome.pruned,
                cutoff.to_rfc3339(),
                outcome.first_seq.unwrap_or(0)
            );
        }
        Command::Config => {
            let config = memvault_core::Config::load(&cli.data_dir)?;
            println!("# {}/{}: every key optional; these are the effective values.", cli.data_dir.display(), memvault_core::CONFIG_FILE);
            println!("# [retrievals] keep_days = N prunes retrievals older than N days at server start and on `memvault prune`.");
            println!("# [namespaces.<name>] half_life_days / floor override [decay] for one namespace.");
            print!("{}", config.to_toml());
        }
        Command::Replay => {
            let mut stores = open(&cli.data_dir)?;
            let fingerprint = stores.indexes.vector.fingerprint().clone();
            let report = recover(&stores.ledger, &mut stores.indexes, &stores.keyring, &fingerprint, RecoveryConfig { verify_chain: true })?;
            println!("{report:?}");
        }
        Command::DumpRecord { seq } => {
            let stores = open(&cli.data_dir)?;
            let record = stores.ledger.read(seq)?.ok_or_else(|| format!("no record at seq {seq}"))?;
            println!("seq {} kind {:?} recorded_at {}", record.header.seq, record.header.kind, record.header.recorded_at.to_rfc3339());
            match record.payload {
                Payload::Assert(a) => {
                    let content_hash_hex: String = a.content_hash.iter().map(|b| format!("{b:02x}")).collect();
                    println!("fact_id {} content_hash {} ciphertext_len {}", a.fact_id, content_hash_hex, a.content.ciphertext.len());
                    match stores.keyring.decrypt(a.fact_id, &a.content) {
                        Ok(plaintext) => println!("content: {}", String::from_utf8_lossy(&plaintext)),
                        Err(e) => println!("content: undecryptable ({e})"),
                    }
                }
                other => println!("payload: {other:?}"),
            }
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
    fn json_string_escapes_windows_paths_and_quotes() {
        assert_eq!(json_string(r"C:\Users\me\mem"), r#""C:\\Users\\me\\mem""#);
        assert_eq!(json_string("a\"b"), r#""a\"b""#);
    }

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
