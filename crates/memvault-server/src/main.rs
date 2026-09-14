//! MCP server exposing the six memory operations over stdio. Stdout is
//! the protocol channel; only stderr is used for logging.
//!
//! Every tool answers with structured JSON (`structuredContent`, mirrored
//! as a text block for clients that predate it), so an agent reads fields
//! rather than parsing a table.
//!
//! `MEMVAULT_EMBEDDING_DIM` sets the embedding width when a data directory
//! is first created; after that the directory's own index decides, and a
//! conflicting value is refused at startup. Setting `MEMVAULT_GRPC_ADDR`
//! serves the same operations over gRPC instead, if the binary was built
//! with `--features grpc`.

#[cfg(feature = "grpc")]
mod grpc;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{
        router::tool::ToolRouter,
        wrapper::{Json, Parameters},
    },
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use memvault_core::{
    erase, explain, memory_as_of, open_stores, recover, supersede_fact, write_fact, AsOfFact,
    AsOfQuery, Explanation, Indexes, InjectedFact, Keyring, Ledger, ModelFingerprint, NamespaceId,
    Query, RecoveryConfig, SourceRef, WriteInput,
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
    /// RFC 3339. When the fact became true. Omit for now.
    #[serde(default)]
    valid_from: Option<chrono::DateTime<chrono::Utc>>,
    /// RFC 3339. When the fact stopped being true, if already known.
    #[serde(default)]
    valid_to: Option<chrono::DateTime<chrono::Utc>>,
    /// Opaque provenance, stored and returned untouched (a URL, a
    /// message id, whatever the caller wants to find this by later).
    #[serde(default)]
    source: Option<String>,
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

/// One retrieval candidate and every score it earned; mirrors
/// `memvault_core::Explanation` field for field.
#[derive(Debug, Serialize, JsonSchema)]
struct ExplanationRow {
    fact_id: String,
    ledger_seq: u64,
    ann_rank: Option<u32>,
    ann_distance: Option<f32>,
    bm25_rank: Option<u32>,
    bm25_score: Option<f32>,
    rrf_score: f32,
    decay_weight: f32,
    final_score: f32,
    /// Injected | CutByBudget | CutByK | FilteredByTime
    outcome: String,
    token_cost: u32,
}

impl From<&Explanation> for ExplanationRow {
    fn from(e: &Explanation) -> Self {
        ExplanationRow {
            fact_id: e.fact_id.to_string(),
            ledger_seq: e.ledger_seq,
            ann_rank: e.ann_rank,
            ann_distance: e.ann_distance,
            bm25_rank: e.bm25_rank,
            bm25_score: e.bm25_score,
            rrf_score: e.rrf_score,
            decay_weight: e.decay_weight,
            final_score: e.final_score,
            outcome: format!("{:?}", e.outcome),
            token_cost: e.token_cost,
        }
    }
}

/// The text of a fact that made it into the answer.
#[derive(Debug, Serialize, JsonSchema)]
struct InjectedRow {
    fact_id: String,
    content: String,
}

impl From<&InjectedFact> for InjectedRow {
    fn from(f: &InjectedFact) -> Self {
        InjectedRow { fact_id: f.fact_id.to_string(), content: String::from_utf8_lossy(&f.content).into_owned() }
    }
}

#[derive(Debug, Serialize, JsonSchema)]
struct FactIdResult {
    fact_id: String,
}

#[derive(Debug, Serialize, JsonSchema)]
struct SearchResult {
    retrieval_id: String,
    /// What to put in context, best first. Only `Injected` candidates.
    injected: Vec<InjectedRow>,
    /// Every candidate considered, the cut ones included.
    candidates: Vec<ExplanationRow>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct ExplainResult {
    retrieval_id: String,
    candidates: Vec<ExplanationRow>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct FactRow {
    fact_id: String,
    valid_from: chrono::DateTime<chrono::Utc>,
    /// Absent while the fact's interval is still open.
    valid_to: Option<chrono::DateTime<chrono::Utc>>,
    content: String,
}

impl From<&AsOfFact> for FactRow {
    fn from(f: &AsOfFact) -> Self {
        FactRow {
            fact_id: f.fact_id.to_string(),
            valid_from: f.valid_from,
            valid_to: f.valid_to,
            content: String::from_utf8_lossy(&f.content).into_owned(),
        }
    }
}

#[derive(Debug, Serialize, JsonSchema)]
struct AsOfResult {
    facts: Vec<FactRow>,
}

/// The width a *new* data directory's vector index gets, from
/// `MEMVAULT_EMBEDDING_DIM`. `None` when unset: the default for a new
/// directory, and "whatever is stored" for an existing one.
fn configured_fingerprint() -> Result<Option<ModelFingerprint>, Box<dyn std::error::Error>> {
    match std::env::var("MEMVAULT_EMBEDDING_DIM") {
        Ok(raw) => {
            let dim: u32 = raw
                .trim()
                .parse()
                .map_err(|e| format!("MEMVAULT_EMBEDDING_DIM={raw:?} is not a positive integer: {e}"))?;
            if dim == 0 {
                return Err("MEMVAULT_EMBEDDING_DIM must be at least 1".into());
            }
            Ok(Some(ModelFingerprint::caller_supplied(dim)))
        }
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("MEMVAULT_EMBEDDING_DIM: {e}").into()),
    }
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
    /// The data directory's embedding fingerprint, read from the vector
    /// index once at open: what every write is validated against.
    fingerprint: ModelFingerprint,
}

impl Stores {
    fn open(data_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let requested = configured_fingerprint()?;
        let (ledger, keyring, mut indexes) = open_stores(data_dir, requested.as_ref())?;
        let fingerprint = indexes.vector.fingerprint().clone();

        let report = recover(&ledger, &mut indexes, &keyring, &fingerprint, RecoveryConfig { verify_chain: true })?;
        eprintln!("memvault-server: recovery report: {report:?} ({} dimensions)", fingerprint.dimensions);

        Ok(Stores { ledger, keyring: Mutex::new(keyring), indexes: Mutex::new(indexes), fingerprint })
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
    #[tool(name = "memory_write", description = "Assert a fact, with optional embedding, pin, and validity interval")]
    async fn memory_write(&self, Parameters(params): Parameters<WriteParams>) -> Result<Json<FactIdResult>, String> {
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
                embedding_model: self.stores.fingerprint.clone(),
                valid_from: params.valid_from.unwrap_or_else(chrono::Utc::now),
                valid_to: params.valid_to,
                fact_id,
                keywords: params.keywords,
                pinned: params.pinned,
                source: params.source.map(|s| SourceRef(s.into_bytes())).unwrap_or_default(),
            },
        )
        .map_err(|e| e.to_string())?;

        Ok(Json(FactIdResult { fact_id: written.to_string() }))
    }

    #[tool(
        name = "memory_as_of",
        description = "Point-in-time query: what was true, or what was believed true, at a given valid_time/transaction_time. Omit either for 'now' on that axis."
    )]
    async fn memory_as_of(&self, Parameters(params): Parameters<AsOfParams>) -> Result<Json<AsOfResult>, String> {
        // Reads the ledger/keyring directly, not the indexes -- see bitemporal.rs.
        let keyring = self.stores.keyring.lock().unwrap();
        let facts = memory_as_of(
            &self.stores.ledger,
            &keyring,
            &NamespaceId(params.namespace),
            AsOfQuery { valid_time: params.valid_time, transaction_time: params.transaction_time },
        )
        .map_err(|e| e.to_string())?;

        Ok(Json(AsOfResult { facts: facts.iter().map(FactRow::from).collect() }))
    }

    #[tool(name = "memory_supersede", description = "Close a fact's open interval without asserting a replacement")]
    async fn memory_supersede(&self, Parameters(params): Parameters<SupersedeParams>) -> Result<Json<FactIdResult>, String> {
        let fact_id = Uuid::parse_str(&params.fact_id).map_err(|e| format!("invalid fact_id: {e}"))?;
        let valid_to = params.valid_to.unwrap_or_else(chrono::Utc::now);

        let mut indexes = self.stores.indexes.lock().unwrap();
        supersede_fact(&self.stores.ledger, &mut indexes, fact_id, valid_to, params.reason).map_err(|e| e.to_string())?;

        Ok(Json(FactIdResult { fact_id: fact_id.to_string() }))
    }

    #[tool(name = "memory_forget", description = "Cryptographically erase a fact: destroy its key so its content is permanently unreadable, everywhere. The ledger record stays; only its content becomes unrecoverable.")]
    async fn memory_forget(&self, Parameters(params): Parameters<ForgetParams>) -> Result<Json<FactIdResult>, String> {
        let fact_id = Uuid::parse_str(&params.fact_id).map_err(|e| format!("invalid fact_id: {e}"))?;

        let mut keyring = self.stores.keyring.lock().unwrap();
        let mut indexes = self.stores.indexes.lock().unwrap();
        erase(&self.stores.ledger, &mut keyring, &mut indexes, fact_id, params.reason).map_err(|e| e.to_string())?;

        Ok(Json(FactIdResult { fact_id: fact_id.to_string() }))
    }

    #[tool(
        name = "memory_search",
        description = "Hybrid search. Returns the injected facts' content, best first, plus full provenance for every candidate considered"
    )]
    async fn memory_search(&self, Parameters(params): Parameters<SearchParams>) -> Result<Json<SearchResult>, String> {
        // Both locks, in the same order as memory_forget, so no erase can
        // land between scoring a fact and reading its content.
        let keyring = self.stores.keyring.lock().unwrap();
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
        let injected = explain::injected_contents(&self.stores.ledger, &keyring, &explanations).map_err(|e| e.to_string())?;

        Ok(Json(SearchResult {
            retrieval_id: retrieval_id.to_string(),
            injected: injected.iter().map(InjectedRow::from).collect(),
            candidates: explanations.iter().map(ExplanationRow::from).collect(),
        }))
    }

    #[tool(name = "memory_explain", description = "Reconstruct a past retrieval exactly from the ledger, including candidates that didn't make it")]
    async fn memory_explain(&self, Parameters(params): Parameters<ExplainParams>) -> Result<Json<ExplainResult>, String> {
        let retrieval_id = Uuid::parse_str(&params.retrieval_id).map_err(|e| format!("invalid retrieval_id: {e}"))?;
        let explanations = explain::explain(&self.stores.ledger, retrieval_id).map_err(|e| e.to_string())?;
        Ok(Json(ExplainResult {
            retrieval_id: retrieval_id.to_string(),
            candidates: explanations.iter().map(ExplanationRow::from).collect(),
        }))
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
