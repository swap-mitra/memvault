//! MCP server exposing the six memory operations over stdio. Stdout is
//! the protocol channel; only stderr is used for logging.
//!
//! Every tool answers with structured JSON (`structuredContent`, mirrored
//! as a text block for clients that predate it), so an agent reads fields
//! rather than parsing a table.
//!
//! `MEMVAULT_EMBEDDING_DIM` sets the embedding width when a data directory
//! is first created; after that the directory's own index decides, and a
//! conflicting value is refused at startup. `MEMVAULT_EMBED_URL` and
//! `MEMVAULT_EMBED_MODEL` point the server at an embedding provider so it
//! can vectorise writes and queries itself (see embed.rs). Setting
//! `MEMVAULT_GRPC_ADDR` serves the same operations over gRPC instead, if
//! the binary was built with `--features grpc`.

mod embed;
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
    erase, explain, get_fact, lock, memory_as_of, open_stores, recover, supersede_fact,
    write_fact_with_limit, AsOfFact, AsOfQuery, Config, Explanation, Indexes, InjectedFact, Keyring,
    Ledger, ModelFingerprint, NamespaceId, Query, RecoveryConfig, SourceRef, WriteInput,
};

use crate::embed::Embedder;

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
struct GetParams {
    fact_id: String,
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
    /// Set when `MEMVAULT_EMBED_URL`/`MEMVAULT_EMBED_MODEL` are configured.
    embedder: Option<Embedder>,
    /// The directory's `memvault.toml`, defaults filled in.
    config: Config,
}

impl Stores {
    async fn open(data_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let embedder = Embedder::from_env()?;
        let configured_dim = configured_fingerprint()?;

        // A provider decides the fingerprint for a new directory: its
        // model's name and whatever width it actually returns, learned by
        // embedding a probe once at startup rather than trusted from config.
        let requested = match &embedder {
            Some(embedder) => {
                let probe = embedder.embed("memvault embedding width probe").await.map_err(|e| format!("embedding provider: {e}"))?;
                let dimensions = probe.len() as u32;
                if let Some(configured) = &configured_dim {
                    if configured.dimensions != dimensions {
                        return Err(format!(
                            "MEMVAULT_EMBEDDING_DIM={} but model {:?} returns {dimensions}-dimensional embeddings",
                            configured.dimensions,
                            embedder.model()
                        )
                        .into());
                    }
                }
                Some(ModelFingerprint { name: embedder.model().to_string(), dimensions, revision_hash: [0u8; 32] })
            }
            None => configured_dim,
        };

        let memvault_core::Stores { ledger, keyring, mut indexes, config } = open_stores(data_dir, requested.as_ref())?;
        let fingerprint = indexes.vector.fingerprint().clone();

        // open_stores checked the width. A directory that already names a
        // *different* model is refused too: same width, different vector
        // space, silently wrong neighbours. "caller-supplied" is the one
        // name that makes no claim, so adopting a provider over it is
        // allowed.
        if let Some(embedder) = &embedder {
            if fingerprint.name != embedder.model() && fingerprint.name != "caller-supplied" {
                return Err(format!(
                    "{} holds embeddings from model {:?}, but MEMVAULT_EMBED_MODEL is {:?}",
                    data_dir.display(),
                    fingerprint.name,
                    embedder.model()
                )
                .into());
            }
        }

        let report = recover(&ledger, &mut indexes, &keyring, &fingerprint, RecoveryConfig { verify_chain: true })?;
        eprintln!(
            "memvault-server: recovery report: {report:?} ({} dimensions, embeddings {})",
            fingerprint.dimensions,
            match &embedder {
                Some(e) => format!("by {:?}", e.model()),
                None => "caller-supplied".to_string(),
            }
        );

        // Retention runs at startup, where it is cheap and predictable,
        // rather than on some search in the middle of a session.
        if let Some(days) = config.retrievals.keep_days {
            let outcome = ledger.prune_retrievals_before(chrono::Utc::now() - chrono::Duration::days(i64::from(days)))?;
            if outcome.pruned > 0 {
                eprintln!(
                    "memvault-server: pruned {} retrieval records older than {days} days; retrievals chain now starts at seq {}",
                    outcome.pruned,
                    outcome.first_seq.unwrap_or(0)
                );
            }
        }

        Ok(Stores { ledger, keyring: Mutex::new(keyring), indexes: Mutex::new(indexes), fingerprint, embedder, config })
    }

    /// The provider's vector for `text`, or `None` when no provider is
    /// configured and the caller has to bring its own.
    async fn embed(&self, text: &str) -> Result<Option<Vec<f32>>, String> {
        match &self.embedder {
            Some(embedder) => embedder.embed(text).await.map(Some).map_err(|e| format!("embedding provider: {e}")),
            None => Ok(None),
        }
    }
}

#[derive(Clone)]
struct MemVaultServer {
    stores: Arc<Stores>,
    tool_router: ToolRouter<Self>,
}

impl MemVaultServer {
    async fn open(data_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(MemVaultServer { stores: Arc::new(Stores::open(data_dir).await?), tool_router: Self::tool_router() })
    }
}

#[tool_router(router = tool_router)]
impl MemVaultServer {
    #[tool(
        name = "memory_write",
        description = "Remember one durable fact, stated so it stands alone. Pass fact_id to supersede an existing fact with this new version. Set pinned for facts that must never decay. valid_from/valid_to (RFC 3339) bound when the fact is true."
    )]
    async fn memory_write(&self, Parameters(params): Parameters<WriteParams>) -> Result<Json<FactIdResult>, String> {
        let fact_id = params
            .fact_id
            .map(|s| Uuid::parse_str(&s))
            .transpose()
            .map_err(|e| format!("invalid fact_id: {e}"))?;
        let embedding = match params.embedding {
            Some(embedding) => Some(embedding),
            None => self.stores.embed(&params.content).await?,
        };

        let mut keyring = lock(&self.stores.keyring);
        let mut indexes = lock(&self.stores.indexes);
        let written = write_fact_with_limit(
            &self.stores.ledger,
            &mut indexes,
            &mut keyring,
            WriteInput {
                namespace: NamespaceId(params.namespace),
                content: params.content.into_bytes(),
                embedding,
                embedding_model: self.stores.fingerprint.clone(),
                valid_from: params.valid_from.unwrap_or_else(chrono::Utc::now),
                valid_to: params.valid_to,
                fact_id,
                keywords: params.keywords,
                pinned: params.pinned,
                source: params.source.map(|s| SourceRef(s.into_bytes())).unwrap_or_default(),
            },
            self.stores.config.limits.max_content_bytes,
        )
        .map_err(|e| e.to_string())?;

        Ok(Json(FactIdResult { fact_id: written.to_string() }))
    }

    #[tool(name = "memory_get", description = "Read one fact's current version by fact_id. Errors if no version is open (superseded, forgotten, or never written).")]
    async fn memory_get(&self, Parameters(params): Parameters<GetParams>) -> Result<Json<FactRow>, String> {
        let fact_id = Uuid::parse_str(&params.fact_id).map_err(|e| format!("invalid fact_id: {e}"))?;
        let keyring = lock(&self.stores.keyring);
        let fact = get_fact(&self.stores.ledger, &keyring, fact_id).map_err(|e| e.to_string())?;
        fact.as_ref().map(FactRow::from).map(Json).ok_or_else(|| format!("no open version of fact {fact_id}"))
    }

    #[tool(
        name = "memory_as_of",
        description = "Every fact in a namespace that was true at valid_time, as believed at transaction_time (RFC 3339; omit either for now). Whole namespace, no ranking: use memory_search to find things, this to audit a moment."
    )]
    async fn memory_as_of(&self, Parameters(params): Parameters<AsOfParams>) -> Result<Json<AsOfResult>, String> {
        // Reads the ledger/keyring directly, not the indexes -- see bitemporal.rs.
        let keyring = lock(&self.stores.keyring);
        let facts = memory_as_of(
            &self.stores.ledger,
            &keyring,
            &NamespaceId(params.namespace),
            AsOfQuery { valid_time: params.valid_time, transaction_time: params.transaction_time },
        )
        .map_err(|e| e.to_string())?;

        Ok(Json(AsOfResult { facts: facts.iter().map(FactRow::from).collect() }))
    }

    #[tool(name = "memory_supersede", description = "Mark a fact as no longer true from valid_to (default now) without asserting a replacement. To replace it instead, memory_write with its fact_id.")]
    async fn memory_supersede(&self, Parameters(params): Parameters<SupersedeParams>) -> Result<Json<FactIdResult>, String> {
        let fact_id = Uuid::parse_str(&params.fact_id).map_err(|e| format!("invalid fact_id: {e}"))?;
        let valid_to = params.valid_to.unwrap_or_else(chrono::Utc::now);

        let mut indexes = lock(&self.stores.indexes);
        supersede_fact(&self.stores.ledger, &mut indexes, fact_id, valid_to, params.reason).map_err(|e| e.to_string())?;

        Ok(Json(FactIdResult { fact_id: fact_id.to_string() }))
    }

    #[tool(name = "memory_forget", description = "Cryptographically erase a fact when asked to forget it: its key is destroyed and the content is unrecoverable everywhere, forever. The ledger keeps a record that something was forgotten, with the reason.")]
    async fn memory_forget(&self, Parameters(params): Parameters<ForgetParams>) -> Result<Json<FactIdResult>, String> {
        let fact_id = Uuid::parse_str(&params.fact_id).map_err(|e| format!("invalid fact_id: {e}"))?;

        let mut keyring = lock(&self.stores.keyring);
        let mut indexes = lock(&self.stores.indexes);
        erase(&self.stores.ledger, &mut keyring, &mut indexes, fact_id, params.reason).map_err(|e| e.to_string())?;

        Ok(Json(FactIdResult { fact_id: fact_id.to_string() }))
    }

    #[tool(
        name = "memory_search",
        description = "Recall: hybrid keyword + vector search over one namespace. Put `injected` (content, best first, already packed to max_tokens) in context; `candidates` is the audit trail of everything considered, cut ones included, with a retrieval_id for memory_explain."
    )]
    async fn memory_search(&self, Parameters(params): Parameters<SearchParams>) -> Result<Json<SearchResult>, String> {
        let (embedding, embedding_model) = match (params.embedding, params.query.as_deref()) {
            (Some(embedding), _) => (Some(embedding), None),
            (None, Some(query)) => match self.stores.embed(query).await? {
                Some(embedding) => (Some(embedding), Some(self.stores.fingerprint.clone())),
                None => (None, None),
            },
            (None, None) => (None, None),
        };

        // Both locks, in the same order as memory_forget, so no erase can
        // land between scoring a fact and reading its content.
        let keyring = lock(&self.stores.keyring);
        let indexes = lock(&self.stores.indexes);
        let (explanations, retrieval_id) = explain::search(
            &self.stores.ledger,
            &indexes,
            &keyring,
            Query {
                text: params.query,
                embedding,
                embedding_model,
                decay: self.stores.config.decay_for(&NamespaceId(params.namespace.clone())),
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

    #[tool(name = "memory_explain", description = "Why did a past memory_search return what it did? Replays its full candidate table from the ledger by retrieval_id, cut candidates included.")]
    async fn memory_explain(&self, Parameters(params): Parameters<ExplainParams>) -> Result<Json<ExplainResult>, String> {
        let retrieval_id = Uuid::parse_str(&params.retrieval_id).map_err(|e| format!("invalid retrieval_id: {e}"))?;
        let explanations = explain::explain(&self.stores.ledger, retrieval_id).map_err(|e| e.to_string())?;
        Ok(Json(ExplainResult {
            retrieval_id: retrieval_id.to_string(),
            candidates: explanations.iter().map(ExplanationRow::from).collect(),
        }))
    }
}

/// What the agent reads once, at connect time. This shapes how memory gets
/// used more than any code below it, so it says when to write, how to
/// recall, and what to do when a fact changes.
const INSTRUCTIONS: &str = "\
MemVault is your long-term memory: a local, append-only, hash-chained store of facts that outlives this conversation.

When to write: after learning something durable about the user, the project, or a decision -- a preference, a convention, a path, a date, an owner, a reason. One fact per memory_write, phrased so it makes sense with no conversation around it (\"the deploy script lives in ops/deploy.sh\", not \"it's in the ops folder\"). Don't store transcripts, or anything you can re-derive by looking at the workspace. Pin facts that must never fade.

Namespaces: one per subject that must never mix, such as one per project or per user. A search only ever sees the namespace it was given.

Recall: before answering from memory, memory_search the namespace. Put the `injected` contents in context; they are already packed to the token budget, best first. `candidates` is the audit trail of everything considered, including what was cut and why. Keep the retrieval_id if you may need to justify the answer; memory_explain replays it later.

When a fact changes: memory_write the new version with the old fact_id, which supersedes it and keeps the history. When something stopped being true with no replacement: memory_supersede. When the user asks you to forget something: memory_forget, and it is unrecoverable. memory_get reads one fact by id; memory_as_of lists what was true at a moment.";

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemVaultServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(INSTRUCTIONS)
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
        return grpc::serve(Arc::new(Stores::open(&data_dir).await?), addr.parse()?).await;
    }

    let server = MemVaultServer::open(&data_dir).await?;
    let running = server.serve(rmcp::transport::stdio()).await?;
    running.waiting().await?;
    Ok(())
}
