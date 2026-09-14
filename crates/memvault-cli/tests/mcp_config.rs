//! `memvault mcp-config` prints a client config an agent can paste in
//! unchanged: absolute paths, valid JSON, and the provider env when asked.

use std::process::Command;

#[test]
fn mcp_config_is_valid_json_with_absolute_paths() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let output = Command::new(env!("CARGO_BIN_EXE_memvault"))
        .arg("--data-dir")
        .arg(tmp.path().join("relative-looking"))
        .args(["mcp-config", "--embed-url", "http://localhost:11434/v1", "--embed-model", "nomic-embed-text"])
        .output()
        .expect("run memvault");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));

    let config: serde_json::Value = serde_json::from_slice(&output.stdout).expect("stdout is JSON");
    let server = &config["mcpServers"]["memvault"];

    let command = std::path::Path::new(server["command"].as_str().expect("command is a string"));
    assert!(command.is_absolute(), "{command:?}");
    assert!(command.file_stem().is_some_and(|s| s == "memvault-server"), "{command:?}");

    let data_dir = std::path::Path::new(server["args"][0].as_str().expect("args[0] is a string"));
    assert!(data_dir.is_absolute(), "{data_dir:?}");
    assert!(data_dir.ends_with("relative-looking"), "{data_dir:?}");

    assert_eq!(server["env"]["MEMVAULT_EMBED_URL"], "http://localhost:11434/v1");
    assert_eq!(server["env"]["MEMVAULT_EMBED_MODEL"], "nomic-embed-text");
}

#[test]
fn mcp_config_omits_env_when_nothing_is_configured() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let output = Command::new(env!("CARGO_BIN_EXE_memvault"))
        .arg("--data-dir")
        .arg(tmp.path())
        .arg("mcp-config")
        .output()
        .expect("run memvault");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));

    let config: serde_json::Value = serde_json::from_slice(&output.stdout).expect("stdout is JSON");
    assert!(config["mcpServers"]["memvault"].get("env").is_none(), "{config}");
}
