//! Phase 0's exit test (docs/IMPLEMENTATION_PLAN.md, task P0-13): spawn
//! the server, burst-write facts, hard-kill it mid-burst, restart it
//! against the same data directory, and confirm recovery brings it back
//! to a correct, searchable state -- not just that the process comes back
//! up.
//!
//! `Child::kill()` sends SIGKILL on Unix and calls TerminateProcess on
//! Windows; neither gives the child a chance at graceful shutdown, which
//! is the point.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_data_dir(tag: &str) -> std::path::PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("memvault-server-kill9-test-{tag}-{}-{n}", std::process::id()))
}

struct ServerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl ServerProcess {
    fn spawn(data_dir: &std::path::Path) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_memvault-server"))
            .arg(data_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn memvault-server");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut server = ServerProcess { child, stdin, stdout };
        server.initialize();
        server
    }

    fn send(&mut self, msg: &serde_json::Value) {
        let line = serde_json::to_string(msg).unwrap();
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn recv(&mut self) -> serde_json::Value {
        let mut line = String::new();
        self.stdout.read_line(&mut line).expect("failed to read a response line from the server");
        assert!(!line.is_empty(), "server closed stdout without responding");
        serde_json::from_str(&line).unwrap_or_else(|e| panic!("non-JSON response line {line:?}: {e}"))
    }

    fn initialize(&mut self) {
        self.send(&serde_json::json!({
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "kill9-test", "version": "0.1.0"}
            }
        }));
        let response = self.recv();
        assert!(response.get("error").is_none(), "initialize failed: {response}");
        self.send(&serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    }

    /// Sends a memory_write call and waits for its response -- a
    /// confirmed write is guaranteed durable (write_fact only returns,
    /// and so the tool call only responds, after the ledger commit and
    /// the index updates both complete). Returns the written fact_id.
    fn write_fact_confirmed(&mut self, id: i64, content: &str) -> String {
        self.send(&serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": {"name": "memory_write", "arguments": {"namespace": "default", "content": content}}
        }));
        let response = self.recv();
        assert!(response.get("error").is_none(), "memory_write failed: {response}");
        let text = response["result"]["content"][0]["text"].as_str().unwrap_or_else(|| panic!("unexpected write response shape: {response}"));
        text.strip_prefix("fact_id: ").unwrap_or_else(|| panic!("unexpected write response text: {text}")).trim().to_string()
    }

    /// Fires a memory_write request without waiting for its response, so
    /// it may still be in flight (anywhere from "not yet received" to
    /// "ledger committed but indexes not yet updated") when the process
    /// is killed immediately afterward.
    fn write_fact_unconfirmed(&mut self, id: i64, content: &str) {
        self.send(&serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": {"name": "memory_write", "arguments": {"namespace": "default", "content": content}}
        }));
    }

    fn search(&mut self, id: i64, query: &str) -> serde_json::Value {
        self.send(&serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": {"name": "memory_search", "arguments": {"namespace": "default", "query": query, "k": 50}}
        }));
        self.recv()
    }

    fn kill(mut self) {
        self.child.kill().expect("failed to kill server process");
        let _ = self.child.wait();
    }
}

#[test]
fn test_exit_kill9_mid_write_recovers() {
    let data_dir = temp_data_dir("main");

    let mut server = ServerProcess::spawn(&data_dir);

    // A handful of confirmed, guaranteed-durable writes.
    const CONFIRMED: i64 = 5;
    let confirmed_fact_ids: Vec<String> =
        (0..CONFIRMED).map(|i| server.write_fact_confirmed(i, &format!("kill9probe confirmed fact number {i}"))).collect();

    // A rapid burst fired without waiting for responses, then an
    // immediate hard kill -- some of these may or may not have landed.
    const BURST: i64 = 15;
    for i in CONFIRMED..CONFIRMED + BURST {
        server.write_fact_unconfirmed(i, &format!("kill9probe burst fact number {i}"));
    }
    server.kill();

    // Restart against the same data directory. Stores::open() runs
    // recover() with chain verification before the server accepts any
    // tool call -- if the kill corrupted the chain, this would fail here.
    let mut restarted = ServerProcess::spawn(&data_dir);

    let response = restarted.search(100, "kill9probe");
    let text = response["result"]["content"][0]["text"].as_str().unwrap_or_else(|| panic!("unexpected search response shape: {response}"));
    assert!(!response["result"]["isError"].as_bool().unwrap_or(false), "search reported an error after recovery: {text}");

    let is_uuid = |token: &str| token.len() == 36 && token.chars().filter(|&c| c == '-').count() == 4;
    let found_rows = text.lines().filter(|l| l.split_whitespace().next().is_some_and(is_uuid)).count();
    // Every one of the confirmed writes' fact_ids must appear; not
    // asserting an exact count for the unconfirmed burst, since exactly
    // how many of those landed before the kill is inherently timing-
    // dependent -- the property under test is "no corruption and
    // confirmed writes survive recovery", not a specific count.
    for fact_id in &confirmed_fact_ids {
        assert!(text.contains(fact_id.as_str()), "confirmed fact_id {fact_id} missing after recovery:\n{text}");
    }
    assert!(found_rows >= confirmed_fact_ids.len(), "search after recovery found fewer results than confirmed writes:\n{text}");

    restarted.kill();
    let _ = std::fs::remove_dir_all(&data_dir);
}
