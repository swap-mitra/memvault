//! `import memvault` -- the pyo3 surface (product doc §6.8, plan task P2-1).
//!
//! Synchronous by construction: there is no event loop here and none is
//! required of the caller. The engine is pure CPU and disk work, so every
//! call releases the GIL around it and other Python threads keep running.
//!
//! Timestamps cross the boundary as RFC 3339 strings, the same encoding the
//! MCP tools use, so a caller can move between the two surfaces without
//! rewriting its date handling.

use std::path::PathBuf;
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use uuid::Uuid;

use memvault_core::{
    erase, explain as core_explain, get_fact, injected_contents, lock, memory_as_of, open_stores,
    recover, search as core_search, supersede_fact, write_fact_with_limit, AsOfFact, AsOfQuery, Config,
    Explanation as CoreExplanation,
    Indexes, InjectedFact, Keyring, Ledger, ModelFingerprint, NamespaceId, Query, RecoveryConfig,
    SourceRef, WriteInput,
};

create_exception!(
    memvault,
    MemVaultError,
    PyException,
    "Raised when the engine rejects an operation."
);

/// One retrieval candidate and every score it earned, mirroring
/// `memvault_core::Explanation` field for field so the FFI and MCP paths
/// report identical provenance.
#[pyclass(name = "Explanation", get_all, frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyExplanation {
    fact_id: String,
    ledger_seq: u64,
    ann_rank: Option<u32>,
    ann_distance: Option<f32>,
    bm25_rank: Option<u32>,
    bm25_score: Option<f32>,
    rrf_score: f32,
    decay_weight: f32,
    final_score: f32,
    outcome: String,
    token_cost: u32,
}

impl From<&CoreExplanation> for PyExplanation {
    fn from(e: &CoreExplanation) -> Self {
        PyExplanation {
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

#[pymethods]
impl PyExplanation {
    fn __repr__(&self) -> String {
        format!(
            "Explanation(fact_id={}, outcome={}, final_score={:.4})",
            self.fact_id, self.outcome, self.final_score
        )
    }
}

/// The text of a fact that made it into a search's answer.
#[pyclass(name = "Injected", get_all, frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyInjected {
    fact_id: String,
    content: String,
}

impl From<&InjectedFact> for PyInjected {
    fn from(f: &InjectedFact) -> Self {
        PyInjected { fact_id: f.fact_id.to_string(), content: String::from_utf8_lossy(&f.content).into_owned() }
    }
}

#[pymethods]
impl PyInjected {
    fn __repr__(&self) -> String {
        format!("Injected(fact_id={}, content={:?})", self.fact_id, self.content)
    }
}

/// One search: what to put in context, and why.
#[pyclass(name = "SearchResult", get_all, frozen, skip_from_py_object)]
struct PySearchResult {
    retrieval_id: String,
    /// `Injected` candidates' content, best first.
    injected: Vec<PyInjected>,
    /// Every candidate considered, the cut ones included.
    candidates: Vec<PyExplanation>,
}

#[pymethods]
impl PySearchResult {
    fn __repr__(&self) -> String {
        format!(
            "SearchResult(retrieval_id={}, injected={}, candidates={})",
            self.retrieval_id,
            self.injected.len(),
            self.candidates.len()
        )
    }
}

/// A fact as it stood at some point on both time axes.
#[pyclass(name = "Fact", get_all, frozen, skip_from_py_object)]
#[derive(Clone)]
struct PyFact {
    fact_id: String,
    valid_from: String,
    valid_to: Option<String>,
    content: String,
}

impl From<&AsOfFact> for PyFact {
    fn from(f: &AsOfFact) -> Self {
        PyFact {
            fact_id: f.fact_id.to_string(),
            valid_from: f.valid_from.to_rfc3339(),
            valid_to: f.valid_to.map(|t| t.to_rfc3339()),
            content: String::from_utf8_lossy(&f.content).into_owned(),
        }
    }
}

#[pymethods]
impl PyFact {
    fn __repr__(&self) -> String {
        format!("Fact(fact_id={}, content={:?})", self.fact_id, self.content)
    }
}

struct Stores {
    ledger: Ledger,
    keyring: Keyring,
    indexes: Indexes,
    /// The data directory's embedding fingerprint, read from the vector
    /// index once at open: what every write is validated against.
    fingerprint: ModelFingerprint,
    /// The directory's `memvault.toml`, defaults filled in.
    config: Config,
}

/// An open MemVault data directory. Cheap to keep alive for the life of the
/// process; opening it twice over the same directory is not supported (the
/// ledger takes an exclusive file lock).
#[pyclass(name = "MemVault", frozen)]
struct PyMemVault {
    /// ponytail: one lock over the whole engine rather than §6.7's
    /// per-component scheme. Writes serialise on the ledger's single writer
    /// anyway, and the FFI's realistic caller is one Python process; split
    /// this when concurrent in-process readers are a measured bottleneck.
    stores: Mutex<Stores>,
}

fn engine_err(e: impl std::fmt::Display) -> PyErr {
    MemVaultError::new_err(e.to_string())
}

fn parse_uuid(what: &str, s: &str) -> PyResult<Uuid> {
    Uuid::parse_str(s).map_err(|e| PyValueError::new_err(format!("invalid {what}: {e}")))
}

fn parse_time(what: &str, s: Option<&str>) -> PyResult<Option<DateTime<Utc>>> {
    s.map(|s| {
        DateTime::parse_from_rfc3339(s)
            .map(|t| t.with_timezone(&Utc))
            .map_err(|e| PyValueError::new_err(format!("invalid {what}: {e}")))
    })
    .transpose()
}

#[pymethods]
impl PyMemVault {
    /// Open (creating if absent) the ledger, keyring, and indexes under
    /// `data_dir`, running recovery first so a crashed previous process
    /// leaves no divergence behind.
    ///
    /// `embedding_dim` is the width your embedding model produces. It is
    /// fixed when the directory is first created; pass it again later or
    /// leave it out, but a different value is refused.
    #[new]
    #[pyo3(signature = (data_dir, *, embedding_dim=None))]
    fn new(py: Python<'_>, data_dir: PathBuf, embedding_dim: Option<u32>) -> PyResult<Self> {
        if embedding_dim == Some(0) {
            return Err(PyValueError::new_err("embedding_dim must be at least 1"));
        }
        py.detach(|| {
            let requested = embedding_dim.map(ModelFingerprint::caller_supplied);
            let memvault_core::Stores { ledger, keyring, mut indexes, config } = open_stores(&data_dir, requested.as_ref()).map_err(engine_err)?;
            let fingerprint = indexes.vector.fingerprint().clone();

            recover(&ledger, &mut indexes, &keyring, &fingerprint, RecoveryConfig { verify_chain: true })
                .map_err(engine_err)?;

            Ok(PyMemVault { stores: Mutex::new(Stores { ledger, keyring, indexes, fingerprint, config }) })
        })
    }

    /// The embedding width this data directory was created with.
    #[getter]
    fn embedding_dim(&self) -> u32 {
        lock(&self.stores).fingerprint.dimensions
    }

    /// Assert a fact. Passing `fact_id` supersedes that fact's currently-open
    /// version instead of asserting an unrelated one. Returns the fact id.
    #[pyo3(signature = (namespace, content, *, embedding=None, pinned=false, fact_id=None, keywords=Vec::new(), valid_from=None, valid_to=None, source=None))]
    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        py: Python<'_>,
        namespace: String,
        content: String,
        embedding: Option<Vec<f32>>,
        pinned: bool,
        fact_id: Option<&str>,
        keywords: Vec<String>,
        valid_from: Option<&str>,
        valid_to: Option<&str>,
        source: Option<String>,
    ) -> PyResult<String> {
        let fact_id = fact_id.map(|s| parse_uuid("fact_id", s)).transpose()?;
        let valid_from = parse_time("valid_from", valid_from)?.unwrap_or_else(Utc::now);
        let valid_to = parse_time("valid_to", valid_to)?;

        py.detach(|| {
            let mut stores = lock(&self.stores);
            let Stores { ledger, keyring, indexes, fingerprint, config } = &mut *stores;
            write_fact_with_limit(
                ledger,
                indexes,
                keyring,
                WriteInput {
                    namespace: NamespaceId(namespace),
                    content: content.into_bytes(),
                    embedding,
                    embedding_model: fingerprint.clone(),
                    valid_from,
                    valid_to,
                    fact_id,
                    keywords,
                    pinned,
                    source: source.map(|s| SourceRef(s.into_bytes())).unwrap_or_default(),
                },
                config.limits.max_content_bytes,
            )
            .map(|id| id.to_string())
            .map_err(engine_err)
        })
    }

    /// Hybrid search. Returns a `SearchResult`: the injected facts' content
    /// (best first) plus an `Explanation` for every candidate considered,
    /// cut ones included.
    #[pyo3(signature = (namespace, query=None, *, embedding=None, k=10, max_tokens=2048))]
    fn search(
        &self,
        py: Python<'_>,
        namespace: String,
        query: Option<String>,
        embedding: Option<Vec<f32>>,
        k: usize,
        max_tokens: u32,
    ) -> PyResult<PySearchResult> {
        py.detach(|| {
            let stores = lock(&self.stores);
            let (explanations, retrieval_id) = core_search(
                &stores.ledger,
                &stores.indexes,
                &stores.keyring,
                Query {
                    text: query,
                    embedding,
                    embedding_model: None,
                    decay: stores.config.decay_for(&NamespaceId(namespace.clone())),
                    namespace: NamespaceId(namespace),
                    as_of: None,
                    k,
                    max_tokens,
                },
            )
            .map_err(engine_err)?;
            let injected = injected_contents(&stores.ledger, &stores.keyring, &explanations).map_err(engine_err)?;
            Ok(PySearchResult {
                retrieval_id: retrieval_id.to_string(),
                injected: injected.iter().map(PyInjected::from).collect(),
                candidates: explanations.iter().map(PyExplanation::from).collect(),
            })
        })
    }

    /// Reconstruct a past retrieval from the ledger, exactly as it was
    /// scored at the time.
    fn explain(&self, py: Python<'_>, retrieval_id: &str) -> PyResult<Vec<PyExplanation>> {
        let retrieval_id = parse_uuid("retrieval_id", retrieval_id)?;
        py.detach(|| {
            let stores = lock(&self.stores);
            core_explain(&stores.ledger, retrieval_id)
                .map(|es| es.iter().map(PyExplanation::from).collect())
                .map_err(engine_err)
        })
    }

    /// Point-in-time query on either axis: what was true (`valid_time`), or
    /// what the engine believed (`transaction_time`). Omit either for "now".
    #[pyo3(signature = (namespace, *, valid_time=None, transaction_time=None))]
    fn as_of(
        &self,
        py: Python<'_>,
        namespace: String,
        valid_time: Option<&str>,
        transaction_time: Option<&str>,
    ) -> PyResult<Vec<PyFact>> {
        let valid_time = parse_time("valid_time", valid_time)?;
        let transaction_time = parse_time("transaction_time", transaction_time)?;

        py.detach(|| {
            let stores = lock(&self.stores);
            let facts = memory_as_of(
                &stores.ledger,
                &stores.keyring,
                &NamespaceId(namespace),
                AsOfQuery { valid_time, transaction_time },
            )
            .map_err(engine_err)?;
            Ok(facts.iter().map(PyFact::from).collect())
        })
    }

    /// One fact's current version by id, or `None` when no version is open
    /// (never written, superseded without a replacement, or forgotten).
    fn get(&self, py: Python<'_>, fact_id: &str) -> PyResult<Option<PyFact>> {
        let fact_id = parse_uuid("fact_id", fact_id)?;
        py.detach(|| {
            let stores = lock(&self.stores);
            get_fact(&stores.ledger, &stores.keyring, fact_id)
                .map(|fact| fact.as_ref().map(PyFact::from))
                .map_err(engine_err)
        })
    }

    /// Close a fact's open valid interval without asserting a replacement.
    #[pyo3(signature = (fact_id, *, valid_to=None, reason=None))]
    fn supersede(
        &self,
        py: Python<'_>,
        fact_id: &str,
        valid_to: Option<&str>,
        reason: Option<String>,
    ) -> PyResult<()> {
        let fact_id = parse_uuid("fact_id", fact_id)?;
        let valid_to = parse_time("valid_to", valid_to)?.unwrap_or_else(Utc::now);

        py.detach(|| {
            let mut stores = lock(&self.stores);
            let Stores { ledger, indexes, .. } = &mut *stores;
            supersede_fact(ledger, indexes, fact_id, valid_to, reason).map_err(engine_err)
        })
    }

    /// Cryptographically erase a fact: its key is destroyed, so its content
    /// becomes permanently unreadable. The ledger record stays and the chain
    /// keeps verifying.
    fn forget(&self, py: Python<'_>, fact_id: &str, reason: String) -> PyResult<()> {
        let fact_id = parse_uuid("fact_id", fact_id)?;
        py.detach(|| {
            let mut stores = lock(&self.stores);
            let Stores { ledger, keyring, indexes, .. } = &mut *stores;
            erase(ledger, keyring, indexes, fact_id, reason).map_err(engine_err)
        })
    }

    /// Verify both hash chains: facts from `from` forward, retrievals from
    /// the oldest record still kept. Raises on the first divergent seq.
    #[pyo3(signature = (*, from=0))]
    fn verify(&self, py: Python<'_>, from: u64) -> PyResult<()> {
        py.detach(|| {
            let stores = lock(&self.stores);
            stores.ledger.verify_from(from).map_err(engine_err)?;
            stores.ledger.verify_retrievals().map_err(engine_err)
        })
    }

    /// Drop retrieval records older than `keep_days` (default: the
    /// directory's `[retrievals] keep_days`) or recorded before `before`
    /// (RFC 3339). The newest always stays. Returns how many were pruned.
    #[pyo3(signature = (*, keep_days=None, before=None))]
    fn prune_retrievals(&self, py: Python<'_>, keep_days: Option<u32>, before: Option<&str>) -> PyResult<u64> {
        let before = parse_time("before", before)?;
        py.detach(|| {
            let stores = lock(&self.stores);
            let cutoff = match (before, keep_days.or(stores.config.retrievals.keep_days)) {
                (Some(before), _) => before,
                (None, Some(days)) => Utc::now() - chrono::Duration::days(i64::from(days)),
                (None, None) => return Err(PyValueError::new_err("pass keep_days or before, or set [retrievals] keep_days in memvault.toml")),
            };
            stores.ledger.prune_retrievals_before(cutoff).map(|o| o.pruned).map_err(engine_err)
        })
    }
}

#[pymodule]
fn memvault(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemVault>()?;
    m.add_class::<PyExplanation>()?;
    m.add_class::<PyInjected>()?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyFact>()?;
    m.add("MemVaultError", m.py().get_type::<MemVaultError>())?;
    Ok(())
}
