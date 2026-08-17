//! Optional gRPC transport (product doc §6.8, plan task P2-2). Feature-gated
//! and off by default: it exists for multi-process deployments, and the
//! primary stdio/MCP case must not pay for it.
//!
//! Same six operations as the MCP tools, over the same `Stores`. The only
//! difference is the wire shape: MCP returns text an agent reads, this
//! returns structured messages a program consumes.

use std::sync::Arc;

use tonic::{Request, Response, Status};

use memvault_core::{
    default_fingerprint, erase, explain as core_explain, memory_as_of, search as core_search,
    supersede_fact, write_fact, AsOfQuery, Explanation, NamespaceId, Query, SourceRef, WriteInput,
};

use crate::Stores;

pub mod pb {
    tonic::include_proto!("memvault.v1");
}

use pb::memory_server::{Memory, MemoryServer};

/// An empty `repeated float` cannot be distinguished from an absent one on
/// the wire, so an empty embedding means "text-only fact/query".
fn embedding(v: Vec<f32>) -> Option<Vec<f32>> {
    (!v.is_empty()).then_some(v)
}

fn parse_uuid(what: &str, s: &str) -> Result<uuid::Uuid, Status> {
    uuid::Uuid::parse_str(s).map_err(|e| Status::invalid_argument(format!("invalid {what}: {e}")))
}

fn parse_time(what: &str, s: Option<&String>) -> Result<Option<chrono::DateTime<chrono::Utc>>, Status> {
    s.map(|s| {
        chrono::DateTime::parse_from_rfc3339(s)
            .map(|t| t.with_timezone(&chrono::Utc))
            .map_err(|e| Status::invalid_argument(format!("invalid {what}: {e}")))
    })
    .transpose()
}

/// Engine rejections are the caller's fault at the boundary (bad dimension,
/// incoherent interval, unknown id) or ours below it; both are reported as
/// a message rather than a panic, per product doc §6.10.
fn engine_err(e: impl std::fmt::Display) -> Status {
    Status::internal(e.to_string())
}

fn to_pb(e: &Explanation) -> pb::Explanation {
    pb::Explanation {
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

fn search_response(retrieval_id: uuid::Uuid, explanations: &[Explanation]) -> pb::SearchResponse {
    pb::SearchResponse {
        retrieval_id: retrieval_id.to_string(),
        explanations: explanations.iter().map(to_pb).collect(),
    }
}

pub struct MemoryService {
    stores: Arc<Stores>,
}

#[tonic::async_trait]
impl Memory for MemoryService {
    async fn write(&self, request: Request<pb::WriteRequest>) -> Result<Response<pb::WriteResponse>, Status> {
        let req = request.into_inner();
        let fact_id = req.fact_id.as_deref().map(|s| parse_uuid("fact_id", s)).transpose()?;

        let mut keyring = self.stores.keyring.lock().unwrap();
        let mut indexes = self.stores.indexes.lock().unwrap();
        let written = write_fact(
            &self.stores.ledger,
            &mut indexes,
            &mut keyring,
            WriteInput {
                namespace: NamespaceId(req.namespace),
                content: req.content.into_bytes(),
                embedding: embedding(req.embedding),
                embedding_model: default_fingerprint(),
                valid_from: chrono::Utc::now(),
                valid_to: None,
                fact_id,
                keywords: req.keywords,
                pinned: req.pinned,
                source: SourceRef::default(),
            },
        )
        .map_err(engine_err)?;

        Ok(Response::new(pb::WriteResponse { fact_id: written.to_string() }))
    }

    async fn search(&self, request: Request<pb::SearchRequest>) -> Result<Response<pb::SearchResponse>, Status> {
        let req = request.into_inner();
        let indexes = self.stores.indexes.lock().unwrap();
        let (explanations, retrieval_id) = core_search(
            &self.stores.ledger,
            &indexes,
            Query {
                text: req.query,
                embedding: embedding(req.embedding),
                embedding_model: None,
                namespace: NamespaceId(req.namespace),
                as_of: None,
                k: req.k as usize,
                max_tokens: req.max_tokens,
            },
        )
        .map_err(engine_err)?;

        Ok(Response::new(search_response(retrieval_id, &explanations)))
    }

    async fn as_of(&self, request: Request<pb::AsOfRequest>) -> Result<Response<pb::AsOfResponse>, Status> {
        let req = request.into_inner();
        let valid_time = parse_time("valid_time", req.valid_time.as_ref())?;
        let transaction_time = parse_time("transaction_time", req.transaction_time.as_ref())?;

        let keyring = self.stores.keyring.lock().unwrap();
        let facts = memory_as_of(
            &self.stores.ledger,
            &keyring,
            &NamespaceId(req.namespace),
            AsOfQuery { valid_time, transaction_time },
        )
        .map_err(engine_err)?;

        Ok(Response::new(pb::AsOfResponse {
            facts: facts
                .iter()
                .map(|f| pb::Fact {
                    fact_id: f.fact_id.to_string(),
                    ledger_seq: f.ledger_seq,
                    valid_from: f.valid_from.to_rfc3339(),
                    valid_to: f.valid_to.map(|t| t.to_rfc3339()),
                    content: String::from_utf8_lossy(&f.content).into_owned(),
                    pinned: f.pinned,
                })
                .collect(),
        }))
    }

    async fn supersede(&self, request: Request<pb::SupersedeRequest>) -> Result<Response<pb::SupersedeResponse>, Status> {
        let req = request.into_inner();
        let fact_id = parse_uuid("fact_id", &req.fact_id)?;
        let valid_to = parse_time("valid_to", req.valid_to.as_ref())?.unwrap_or_else(chrono::Utc::now);

        let mut indexes = self.stores.indexes.lock().unwrap();
        supersede_fact(&self.stores.ledger, &mut indexes, fact_id, valid_to, req.reason).map_err(engine_err)?;

        Ok(Response::new(pb::SupersedeResponse {}))
    }

    async fn forget(&self, request: Request<pb::ForgetRequest>) -> Result<Response<pb::ForgetResponse>, Status> {
        let req = request.into_inner();
        let fact_id = parse_uuid("fact_id", &req.fact_id)?;

        let mut keyring = self.stores.keyring.lock().unwrap();
        let mut indexes = self.stores.indexes.lock().unwrap();
        erase(&self.stores.ledger, &mut keyring, &mut indexes, fact_id, req.reason).map_err(engine_err)?;

        Ok(Response::new(pb::ForgetResponse {}))
    }

    async fn explain(&self, request: Request<pb::ExplainRequest>) -> Result<Response<pb::SearchResponse>, Status> {
        let retrieval_id = parse_uuid("retrieval_id", &request.into_inner().retrieval_id)?;
        let explanations = core_explain(&self.stores.ledger, retrieval_id).map_err(engine_err)?;
        Ok(Response::new(search_response(retrieval_id, &explanations)))
    }
}

pub async fn serve(stores: Arc<Stores>, addr: std::net::SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("memvault-server: gRPC listening on {addr}");
    tonic::transport::Server::builder()
        .add_service(MemoryServer::new(MemoryService { stores }))
        .serve(addr)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pb::memory_client::MemoryClient;

    /// P2-2's smoke test: one real RPC over a real socket, proving the
    /// feature-gated transport is wired to the same engine the MCP tools
    /// use. Writes a fact and searches it back.
    #[tokio::test]
    async fn grpc_write_then_search_roundtrips() {
        let dir = std::env::temp_dir().join(format!("memvault-grpc-smoke-{}", uuid::Uuid::new_v4()));
        let stores = Arc::new(Stores::open(&dir).expect("open stores"));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(MemoryServer::new(MemoryService { stores }))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .expect("serve");
        });

        let mut client = MemoryClient::connect(format!("http://{addr}")).await.expect("connect");

        let fact_id = client
            .write(pb::WriteRequest {
                namespace: "grpc-smoke".into(),
                content: "the deploy script lives in ops/deploy.sh".into(),
                ..Default::default()
            })
            .await
            .expect("write")
            .into_inner()
            .fact_id;

        let response = client
            .search(pb::SearchRequest {
                namespace: "grpc-smoke".into(),
                query: Some("deploy script".into()),
                k: 10,
                max_tokens: 2048,
                ..Default::default()
            })
            .await
            .expect("search")
            .into_inner();

        let hit = response
            .explanations
            .iter()
            .find(|e| e.fact_id == fact_id)
            .expect("written fact missing from search results");
        assert_eq!(hit.outcome, "Injected");
        assert!(hit.bm25_rank.is_some(), "no BM25 rank: the keyword axis never ran");

        std::fs::remove_dir_all(&dir).ok();
    }
}
