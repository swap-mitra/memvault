//! A stdio MCP client just big enough to drive `memvault-server` from a
//! test: spawn against a data directory, call tools, read the structured
//! result back.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

pub struct ServerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    next_id: i64,
}

impl ServerProcess {
    pub fn spawn(data_dir: &std::path::Path) -> Self {
        Self::spawn_with_env(data_dir, &[])
    }

    pub fn spawn_with_env(data_dir: &std::path::Path, env: &[(&str, &str)]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_memvault-server"));
        command.arg(data_dir).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        for (k, v) in env {
            command.env(k, v);
        }
        let mut child = command.spawn().expect("failed to spawn memvault-server");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let mut server = ServerProcess { child, stdin, stdout, next_id: 1 };
        server.initialize();
        server
    }

    pub fn send(&mut self, msg: &serde_json::Value) {
        let line = serde_json::to_string(msg).unwrap();
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
    }

    pub fn recv(&mut self) -> serde_json::Value {
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
                "clientInfo": {"name": "memvault-server-test", "version": "0.1.0"}
            }
        }));
        let response = self.recv();
        assert!(response.get("error").is_none(), "initialize failed: {response}");
        self.send(&serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
    }

    /// Fires a tool call without waiting for its response.
    pub fn call_unconfirmed(&mut self, tool: &str, arguments: serde_json::Value) {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments}
        }));
    }

    /// Calls a tool and returns the whole JSON-RPC response.
    pub fn call_raw(&mut self, tool: &str, arguments: serde_json::Value) -> serde_json::Value {
        self.call_unconfirmed(tool, arguments);
        self.recv()
    }

    /// Calls a tool that is expected to succeed and returns its
    /// `structuredContent`.
    pub fn call(&mut self, tool: &str, arguments: serde_json::Value) -> serde_json::Value {
        let response = self.call_raw(tool, arguments);
        assert!(response.get("error").is_none(), "{tool} failed: {response}");
        assert!(!response["result"]["isError"].as_bool().unwrap_or(false), "{tool} reported an error: {response}");
        let structured = response["result"]["structuredContent"].clone();
        assert!(structured.is_object(), "{tool} returned no structuredContent: {response}");
        structured
    }

    pub fn kill(mut self) {
        self.child.kill().expect("failed to kill server process");
        let _ = self.child.wait();
    }
}
