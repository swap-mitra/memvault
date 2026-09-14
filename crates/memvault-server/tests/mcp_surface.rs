//! The MCP surface's contract with an agent: results are structured
//! JSON, a search hands back the injected facts' content, a write can
//! carry its validity interval, and a data directory's embedding width is
//! set once and then defended.

mod common;

use std::process::{Command, Stdio};

use common::ServerProcess;
use serde_json::json;

#[test]
fn search_returns_injected_content_and_structured_provenance() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let mut server = ServerProcess::spawn(tmp.path());

    let written = server.call("memory_write", json!({"namespace": "project", "content": "the deploy script lives in ops/deploy.sh"}));
    let fact_id = written["fact_id"].as_str().expect("write returns a fact_id").to_string();
    server.call("memory_write", json!({"namespace": "project", "content": "staging runs postgres 16"}));

    let result = server.call("memory_search", json!({"namespace": "project", "query": "deploy script"}));
    assert!(result["retrieval_id"].is_string(), "{result}");

    let injected = result["injected"].as_array().expect("injected is a list");
    assert_eq!(injected[0]["fact_id"], fact_id, "best match first: {result}");
    assert_eq!(injected[0]["content"], "the deploy script lives in ops/deploy.sh");

    let candidates = result["candidates"].as_array().expect("candidates is a list");
    let row = candidates.iter().find(|c| c["fact_id"] == fact_id.as_str()).expect("the written fact was considered");
    assert_eq!(row["outcome"], "Injected");
    assert!(row["bm25_rank"].is_number(), "keyword axis ran: {row}");

    // Explain reconstructs the same candidate list, and never content.
    let explained = server.call("memory_explain", json!({"retrieval_id": result["retrieval_id"]}));
    assert_eq!(explained["candidates"], result["candidates"]);
    assert!(explained.get("injected").is_none());

    server.kill();
}

#[test]
fn write_accepts_a_validity_interval_and_source() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let mut server = ServerProcess::spawn(tmp.path());

    // Already closed: true last year, not any more.
    server.call(
        "memory_write",
        json!({
            "namespace": "hist",
            "content": "the on-call rotation was in PagerDuty schedule P3",
            "valid_from": "2025-01-01T00:00:00Z",
            "valid_to": "2025-12-31T00:00:00Z",
            "source": "runbook v1"
        }),
    );

    let now = server.call("memory_as_of", json!({"namespace": "hist"}));
    assert_eq!(now["facts"].as_array().map(Vec::len), Some(0), "a closed fact is not true now: {now}");

    let then = server.call("memory_as_of", json!({"namespace": "hist", "valid_time": "2025-06-01T00:00:00Z"}));
    let facts = then["facts"].as_array().expect("facts is a list");
    assert_eq!(facts.len(), 1, "{then}");
    assert_eq!(facts[0]["content"], "the on-call rotation was in PagerDuty schedule P3");
    assert_eq!(facts[0]["valid_from"], "2025-01-01T00:00:00Z");
    assert_eq!(facts[0]["valid_to"], "2025-12-31T00:00:00Z");

    // The engine's interval check surfaces as a tool error, not a panic.
    let bad = server.call_raw(
        "memory_write",
        json!({"namespace": "hist", "content": "x", "valid_from": "2025-02-01T00:00:00Z", "valid_to": "2025-01-01T00:00:00Z"}),
    );
    assert_eq!(bad["result"]["isError"], true, "{bad}");

    server.kill();
}

#[test]
fn embedding_width_is_set_at_creation_and_defended_after() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let data_dir = tmp.path();

    // Created at 4 dimensions: a 4-wide embedding is accepted.
    let mut server = ServerProcess::spawn_with_env(data_dir, &[("MEMVAULT_EMBEDDING_DIM", "4")]);
    server.call("memory_write", json!({"namespace": "v", "content": "four wide", "embedding": [1.0, 0.0, 0.0, 0.0]}));
    let wrong = server.call_raw("memory_write", json!({"namespace": "v", "content": "wrong width", "embedding": [1.0, 0.0]}));
    assert_eq!(wrong["result"]["isError"], true, "{wrong}");
    server.kill();

    // Reopened with nothing configured: the directory remembers its width.
    let mut server = ServerProcess::spawn(data_dir);
    server.call("memory_write", json!({"namespace": "v", "content": "still four wide", "embedding": [0.0, 1.0, 0.0, 0.0]}));
    let result = server.call("memory_search", json!({"namespace": "v", "embedding": [0.0, 1.0, 0.0, 0.0]}));
    assert_eq!(result["injected"][0]["content"], "still four wide", "{result}");
    server.kill();

    // Reopened with a conflicting width: refused at startup, before
    // anything could be written at the wrong shape.
    let output = Command::new(env!("CARGO_BIN_EXE_memvault-server"))
        .arg(data_dir)
        .env("MEMVAULT_EMBEDDING_DIM", "8")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn memvault-server");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "server started against a 4-wide directory with MEMVAULT_EMBEDDING_DIM=8:\n{stderr}");
    assert!(stderr.contains("already holds 4-dimensional"), "{stderr}");
}
