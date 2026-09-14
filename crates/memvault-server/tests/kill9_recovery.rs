//! Phase 0's exit test (docs/IMPLEMENTATION_PLAN.md, task P0-13): spawn
//! the server, burst-write facts, hard-kill it mid-burst, restart it
//! against the same data directory, and confirm recovery brings it back
//! to a correct, searchable state -- not just that the process comes back
//! up.
//!
//! `Child::kill()` sends SIGKILL on Unix and calls TerminateProcess on
//! Windows; neither gives the child a chance at graceful shutdown, which
//! is the point.

mod common;

use common::ServerProcess;
use serde_json::json;

/// A confirmed write is guaranteed durable: write_fact only returns, and
/// so the tool call only responds, after the ledger commit and the index
/// updates both complete.
fn write_fact_confirmed(server: &mut ServerProcess, content: &str) -> String {
    let result = server.call("memory_write", json!({"namespace": "default", "content": content}));
    result["fact_id"].as_str().unwrap_or_else(|| panic!("unexpected write result shape: {result}")).to_string()
}

#[test]
fn test_exit_kill9_mid_write_recovers() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let data_dir = tmp.path();

    let mut server = ServerProcess::spawn(data_dir);

    // A handful of confirmed, guaranteed-durable writes.
    const CONFIRMED: usize = 5;
    let confirmed_fact_ids: Vec<String> =
        (0..CONFIRMED).map(|i| write_fact_confirmed(&mut server, &format!("kill9probe confirmed fact number {i}"))).collect();

    // A rapid burst fired without waiting for responses, then an
    // immediate hard kill -- some of these may or may not have landed
    // (anywhere from "not yet received" to "ledger committed but indexes
    // not yet updated" when the process dies).
    const BURST: usize = 15;
    for i in CONFIRMED..CONFIRMED + BURST {
        server.call_unconfirmed("memory_write", json!({"namespace": "default", "content": format!("kill9probe burst fact number {i}")}));
    }
    server.kill();

    // Restart against the same data directory. Stores::open() runs
    // recover() with chain verification before the server accepts any
    // tool call -- if the kill corrupted the chain, this would fail here.
    let mut restarted = ServerProcess::spawn(data_dir);

    let result = restarted.call("memory_search", json!({"namespace": "default", "query": "kill9probe", "k": 50}));
    let candidates = result["candidates"].as_array().unwrap_or_else(|| panic!("unexpected search result shape: {result}"));
    let found: Vec<&str> = candidates.iter().filter_map(|c| c["fact_id"].as_str()).collect();

    // Every one of the confirmed writes' fact_ids must appear; not
    // asserting an exact count for the unconfirmed burst, since exactly
    // how many of those landed before the kill is inherently timing-
    // dependent -- the property under test is "no corruption and
    // confirmed writes survive recovery", not a specific count.
    for fact_id in &confirmed_fact_ids {
        assert!(found.contains(&fact_id.as_str()), "confirmed fact_id {fact_id} missing after recovery:\n{result}");
    }
    assert!(found.len() >= confirmed_fact_ids.len(), "search after recovery found fewer results than confirmed writes:\n{result}");

    restarted.kill();
}
