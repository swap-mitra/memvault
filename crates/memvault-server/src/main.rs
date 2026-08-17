//! MCP server exposing `memory_write` and `memory_search` over stdio.
//! Stdout is the protocol channel; only stderr is used for logging.
//!
//! Setting `MEMVAULT_GRPC_ADDR` serves the same operations over gRPC
//! instead, if the binary was built with `--features grpc`.

#[cfg(feature = "grpc")]
mod grpc;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use memvault_core::{
    default_fingerprint, erase, explain, memory_as_of, recover, supersede_fact, write_fact,
    AsOfQuery, Explanation, Indexes, KeywordIndex, Keyring, Ledger, NamespaceId, Query,
    RecoveryConfig, SourceRef, VectorIndex, WriteInput,
};

fn default_k() -> usize {
    10
}

fn default_max_tokens() -> u32 {
    2048
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WriteParams {
    namespace: String,
    content: String,
    /// Caller-supplied embedding; must match the server's configured
    /// dimensionality if present. Omit for a text-only fact.
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default)]
    pinned: bool,
    /// Supersedes this fact's currently-open version instead of asserting
    /// a new one, if set.
    #[serde(default)]
    fact_id: Option<String>,
    #[serde(default)]
    keywords: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct AsOfParams {
    namespace: String,
    /// RFC 3339 timestamp. Omit for "now" on this axis.
    #[serde(default)]
    valid_time: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    transaction_time: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SupersedeParams {
    fact_id: String,
    /// When the interval closes. Omit for now.
    #[serde(default)]
    valid_to: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ExplainParams {
    retrieval_id: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct ForgetParams {
    fact_id: String,
    reason: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SearchParams {
    namespace: String,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    embedding: Option<Vec<f32>>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
}

fn format_explanations(retrieval_id: Uuid, explanations: &[Explanation]) -> String {
    let mut out = format!("retrieval_id: {retrieval_id}\n");
    out.push_str(
        "fact_id                              ann_rank   ann_dist  bm25_rk bm25_score       rrf  decay_wt     final       outcome tokens\n",
    );
    for e in explanations {
        out.push_str(&format!(
            "{:<36} {:>8} {:>10} {:>8} {:>10} {:>9.4} {:>9.4} {:>9.4} {:>13} {:>6}\n",
            e.fact_id,
            e.ann_rank.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
            e.ann_distance.map(|d| format!("{d:.4}")).unwrap_or_else(|| "-".into()),
            e.bm25_rank.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
            e.bm25_score.map(|s| format!("{s:.4}")).unwrap_or_else(|| "-".into()),
            e.rrf_score,
            e.decay_weight,
            e.final_score,
            format!("{:?}", e.outcome),
            e.token_cost,
        ));
    }
    out
}

struct Stores {
    ledger: Ledger,
    keyring: Mutex<Keyring>,
    /// ponytail: one lock over both indexes together, not the doc's
    /// eventual per-index RwLock scheme (§6.7) -- correct and simple for
    /// Phase 0's single-connection stdio server; upgrade path is splitting
    /// this once concurrent multi-client throughput is actually a
    /// bottleneck worth measuring.
    indexes: Mutex<Indexes>,
}

impl Stores {
    fn open(data_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(data_dir)?;
        let ledger = Ledger::open(&data_dir.join("ledger.redb"))?;
        let keyring = Keyring::open(&data_dir.join("keys.redb"))?;
        let vector = VectorIndex::open_or_create(&data_dir.join("vectors.usearch"), &default_fingerprint())?;
        let keyword = KeywordIndex::open_or_create(&data_dir.join("keyword"))?;
        let mut indexes = Indexes { vector, keyword };

        let report = recover(&ledger, &mut indexes, &keyring, &default_fingerprint(), RecoveryConfig { verify_chain: true })?;
        eprintln!("memvault-server: recovery report: {report:?}");

        Ok(Stores { ledger, keyring: Mutex::new(keyring), indexes: Mutex::new(indexes) })
    }
}

#[derive(Clone)]
struct MemVaultServer {
    stores: Arc<Stores>,
    tool_router: ToolRouter<Self>,
}

impl MemVaultServer {
    fn open(data_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(MemVaultServer { stores: Arc::new(Stores::open(data_dir)?), tool_router: Self::tool_router() })
    }
}

#[tool_router(router = tool_router)]
impl MemVaultServer {
    #[tool(name = "memory_write", description = "Assert a fact, with optional embedding and pin")]
    async fn memory_write(&self, Parameters(params): Parameters<WriteParams>) -> Result<String, String> {
        let fact_id = params
            .fact_id
            .map(|s| Uuid::parse_str(&s))
            .transpose()
            .map_err(|e| format!("invalid fact_id: {e}"))?;

        let mut keyring = self.stores.keyring.lock().unwrap();
        let mut indexes = self.stores.indexes.lock().unwrap();
        let written = write_fact(
            &self.stores.ledger,
            &mut indexes,
            &mut keyring,
            WriteInput {
                namespace: NamespaceId(params.namespace),
                content: params.content.into_bytes(),
                embedding: params.embedding,
                embedding_model: default_fingerprint(),
                valid_from: chrono::Utc::now(),
                valid_to: None,
                fact_id,
                keywords: params.keywords,
                pinned: params.pinned,
                source: SourceRef::default(),
            },
        )
        .map_err(|e| e.to_string())?;

        Ok(format!("fact_id: {written}"))
    }

    #[tool(
        name = "memory_as_of",
        description = "Point-in-time query: what was true, or what was believed true, at a given valid_time/transaction_time. Omit either for 'now' on that axis."
    )]
    async fn memory_as_of(&self, Parameters(params): Parameters<AsOfParams>) -> Result<String, String> {
        // Reads the ledger/keyring directly, not the indexes -- see bitemporal.rs.
        let keyring = self.stores.keyring.lock().unwrap();
        let facts = memory_as_of(
            &self.stores.ledger,
            &keyring,
            &NamespaceId(params.namespace),
            AsOfQuery { valid_time: params.valid_time, transaction_time: params.transaction_time },
        )
        .map_err(|e| e.to_string())?;

        let mut out = String::new();
        for f in &facts {
            let content = String::from_utf8_lossy(&f.content);
            let valid_to = f.valid_to.map(|t| t.to_rfc3339()).unwrap_or_else(|| "open".into());
            out.push_str(&format!("{} [{} .. {}] {}\n", f.fact_id, f.valid_from.to_rfc3339(), valid_to, content));
        }
        Ok(out)
    }

    #[tool(name = "memory_supersede", description = "Close a fact's open interval without asserting a replacement")]
    async fn memory_supersede(&self, Parameters(params): Parameters<SupersedeParams>) -> Result<String, String> {
        let fact_id = Uuid::parse_str(&params.fact_id).map_err(|e| format!("invalid fact_id: {e}"))?;
        let valid_to = params.valid_to.unwrap_or_else(chrono::Utc::now);

        let mut indexes = self.stores.indexes.lock().unwrap();
        supersede_fact(&self.stores.ledger, &mut indexes, fact_id, valid_to, params.reason).map_err(|e| e.to_string())?;

        Ok(format!("superseded fact_id: {fact_id}"))
    }

    #[tool(name = "memory_forget", description = "Cryptographically erase a fact: destroy its key so its content is permanently unreadable, everywhere. The ledger record stays; only its content becomes unrecoverable.")]
    async fn memory_forget(&self, Parameters(params): Parameters<ForgetParams>) -> Result<String, String> {
        let fact_id = Uuid::parse_str(&params.fact_id).map_err(|e| format!("invalid fact_id: {e}"))?;

        let mut keyring = self.stores.keyring.lock().unwrap();
        let mut indexes = self.stores.indexes.lock().unwrap();
        erase(&self.stores.ledger, &mut keyring, &mut indexes, fact_id, params.reason).map_err(|e| e.to_string())?;

        Ok(format!("forgot fact_id: {fact_id}"))
    }

    #[tool(name = "memory_search", description = "Hybrid search with full provenance for every candidate considered")]
    async fn memory_search(&self, Parameters(params): Parameters<SearchParams>) -> Result<String, String> {
        let indexes = self.stores.indexes.lock().unwrap();
        let (explanations, retrieval_id) = explain::search(
            &self.stores.ledger,
            &indexes,
            Query {
                text: params.query,
                embedding: params.embedding,
                embedding_model: None,
                namespace: NamespaceId(params.namespace),
                as_of: None,
                k: params.k,
                max_tokens: params.max_tokens,
            },
        )
        .map_err(|e| e.to_string())?;

        Ok(format_explanations(retrieval_id, &explanations))
    }

    #[tool(name = "memory_explain", description = "Reconstruct a past retrieval exactly from the ledger, including candidates that didn't make it")]
    async fn memory_explain(&self, Parameters(params): Parameters<ExplainParams>) -> Result<String, String> {
        let retrieval_id = Uuid::parse_str(&params.retrieval_id).map_err(|e| format!("invalid retrieval_id: {e}"))?;
        let explanations = explain::explain(&self.stores.ledger, retrieval_id).map_err(|e| e.to_string())?;
        Ok(format_explanations(retrieval_id, &explanations))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemVaultServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("MemVault: local-first memory engine for AI agents. Every write and retrieval is recorded in an append-only, hash-chained ledger.")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("./memvault-data"));

    let grpc_addr = std::env::var("MEMVAULT_GRPC_ADDR").ok();
    #[cfg(not(feature = "grpc"))]
    if grpc_addr.is_some() {
        return Err("MEMVAULT_GRPC_ADDR is set but this binary was built without --features grpc".into());
    }
    #[cfg(feature = "grpc")]
    if let Some(addr) = grpc_addr {
        return grpc::serve(Arc::new(Stores::open(&data_dir)?), addr.parse()?).await;
    }

    let server = MemVaultServer::open(&data_dir)?;
    let running = server.serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}
