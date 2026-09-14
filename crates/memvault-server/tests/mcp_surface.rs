//! The MCP surface's contract with an agent: results are structured
//! JSON, a search hands back the injected facts' content, a write can
//! carry its validity interval, and a data directory's embedding width is
//! set once and then defended.

mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
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

#[test]
fn get_reads_one_fact_until_it_is_forgotten() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let mut server = ServerProcess::spawn(tmp.path());

    let written = server.call("memory_write", json!({"namespace": "g", "content": "the api key lives in vault at secret/api"}));
    let fact_id = written["fact_id"].as_str().expect("fact_id").to_string();

    let fact = server.call("memory_get", json!({"fact_id": fact_id}));
    assert_eq!(fact["content"], "the api key lives in vault at secret/api");
    assert!(fact["valid_to"].is_null(), "{fact}");

    server.call("memory_forget", json!({"fact_id": fact_id, "reason": "test"}));
    let gone = server.call_raw("memory_get", json!({"fact_id": fact_id}));
    assert_eq!(gone["result"]["isError"], true, "{gone}");

    server.kill();
}

/// An OpenAI-compatible `/embeddings` endpoint small enough to live in a
/// test: any input mentioning "deploy" embeds to one axis, everything else
/// to another, so nearest-neighbour results are predictable. Serves until
/// the test process exits.
fn fake_embedding_provider() -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let models_seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = models_seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let mut reader = BufReader::new(stream);
            let mut content_length = 0usize;
            let mut line = String::new();
            loop {
                line.clear();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; content_length];
            reader.read_exact(&mut body).expect("body");
            let request: serde_json::Value = serde_json::from_slice(&body).expect("json body");
            seen.lock().unwrap().push(request["model"].as_str().unwrap_or("").to_string());
            let embedding = if request["input"].as_str().unwrap_or("").contains("deploy") { [1.0, 0.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0, 0.0] };
            let payload = json!({"data": [{"embedding": embedding}]}).to_string();
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", payload.len(), payload);
            let mut stream = reader.into_inner();
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (url, models_seen)
}

#[test]
fn a_configured_provider_embeds_writes_and_queries_and_names_the_directory() {
    let (url, models_seen) = fake_embedding_provider();
    let tmp = tempfile::tempdir().expect("temp dir");
    let data_dir = tmp.path();
    let env = [("MEMVAULT_EMBED_URL", url.as_str()), ("MEMVAULT_EMBED_MODEL", "fake-embed")];

    let mut server = ServerProcess::spawn_with_env(data_dir, &env);
    let deploy = server.call("memory_write", json!({"namespace": "p", "content": "the deploy script lives in ops/deploy.sh"}));
    server.call("memory_write", json!({"namespace": "p", "content": "staging runs postgres 16"}));

    // No embedding supplied by the caller: the server fetched one, so the
    // vector axis ran and ranked the deploy fact first.
    let result = server.call("memory_search", json!({"namespace": "p", "query": "where is the deploy script"}));
    let candidates = result["candidates"].as_array().expect("candidates");
    let deploy_row = candidates.iter().find(|c| c["fact_id"] == deploy["fact_id"]).expect("deploy fact considered");
    assert_eq!(deploy_row["ann_rank"], 0, "{result}");
    assert!(candidates.iter().all(|c| c["ann_rank"].is_number()), "every candidate got a vector rank: {result}");
    assert_eq!(result["injected"][0]["fact_id"], deploy["fact_id"]);
    server.kill();

    assert!(models_seen.lock().unwrap().iter().all(|m| m == "fake-embed"), "{models_seen:?}");

    // The directory now carries the model's name; a different model is
    // refused at startup, same width or not.
    let output = Command::new(env!("CARGO_BIN_EXE_memvault-server"))
        .arg(data_dir)
        .env("MEMVAULT_EMBED_URL", &url)
        .env("MEMVAULT_EMBED_MODEL", "some-other-model")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn memvault-server");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "started against a fake-embed directory with another model:\n{stderr}");
    assert!(stderr.contains("holds embeddings from model") && stderr.contains("fake-embed"), "{stderr}");
}
