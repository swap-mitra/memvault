//! `memvault prune` drops old retrievals from the front of their chain,
//! keeps the newest as the anchor, and leaves `verify` passing and the
//! surviving searches explainable.

use std::process::{Command, Output};

fn run(data_dir: &std::path::Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_memvault"))
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .output()
        .expect("failed to run the memvault binary")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn retrieval_id(search_output: &str) -> String {
    search_output
        .lines()
        .find_map(|l| l.strip_prefix("retrieval_id: "))
        .unwrap_or_else(|| panic!("no retrieval_id in:\n{search_output}"))
        .trim()
        .to_string()
}

#[test]
fn prune_keeps_the_newest_retrieval_and_the_chain_verifying() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();

    for content in ["the deploy script lives in ops/deploy.sh", "staging runs postgres 16"] {
        let out = run(dir, &["write", "--namespace", "p", "--content", content]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }
    let ids: Vec<String> = (0..3)
        .map(|_| {
            let out = run(dir, &["search", "--namespace", "p", "--query", "deploy script"]);
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            retrieval_id(&stdout(&out))
        })
        .collect();

    // A cutoff in the past prunes nothing.
    let out = run(dir, &["prune", "--before", "2000-01-01T00:00:00Z"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout(&out).contains("pruned 0 retrieval records"), "{}", stdout(&out));

    // A cutoff in the future covers everything, and the newest still stays.
    let out = run(dir, &["prune", "--before", "2100-01-01T00:00:00Z"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);
    assert!(text.contains("pruned 2 retrieval records"), "{text}");
    assert!(text.contains("starts at seq 2"), "{text}");

    let out = run(dir, &["verify"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = stdout(&out);
    assert!(text.contains("chain verified from seq 0"), "{text}");
    assert!(text.contains("retrievals chain verified from seq 2"), "{text}");

    assert!(!run(dir, &["explain", &ids[0]]).status.success(), "a pruned retrieval is gone");
    assert!(run(dir, &["explain", &ids[2]]).status.success(), "the kept retrieval still explains");

    // Nothing configured and nothing asked for is an error, not a silent no-op.
    let out = run(dir, &["prune"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("keep_days"), "{}", String::from_utf8_lossy(&out.stderr));
}
