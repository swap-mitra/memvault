//! The plan's acceptance test for `memvault verify`: a corrupted ledger
//! makes the CLI exit non-zero and report the first diverging seq.

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use memvault_core::record::{canonical_bytes, decode_record, Payload};
use redb::{ReadableTable, TableDefinition};

const RECORDS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("records");

fn temp_data_dir(tag: &str) -> std::path::PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("memvault-cli-verify-test-{tag}-{}-{n}", std::process::id()))
}

fn run(data_dir: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_memvault"))
        .arg("--data-dir")
        .arg(data_dir)
        .args(args)
        .output()
        .expect("failed to run the memvault binary")
}

/// Flips a byte inside seq `seq`'s stored Assert ciphertext, bypassing the
/// ledger's own API entirely (which has no mutation path by design) to
/// simulate on-disk corruption discovered at read time -- same technique
/// chain_tests.rs uses at the in-memory level.
fn corrupt_ciphertext_at(ledger_path: &std::path::Path, seq: u64) {
    let db = redb::Database::open(ledger_path).unwrap();
    let write_txn = db.begin_write().unwrap();
    {
        let mut table = write_txn.open_table(RECORDS_TABLE).unwrap();
        let bytes = table.get(seq).unwrap().unwrap().value().to_vec();
        let mut record = decode_record(&bytes).unwrap();
        match &mut record.payload {
            Payload::Assert(a) => a.content.ciphertext[0] ^= 0xFF,
            other => panic!("expected an Assert at seq {seq}, got {other:?}"),
        }
        table.insert(seq, canonical_bytes(&record).as_slice()).unwrap();
    }
    write_txn.commit().unwrap();
}

#[test]
fn test_cli_verify_reports_divergent_seq() {
    let data_dir = temp_data_dir("divergent");

    let first = run(&data_dir, &["write", "--namespace", "default", "--content", "alpha bravo charlie"]);
    assert!(first.status.success(), "first write failed: {}", String::from_utf8_lossy(&first.stderr));
    let second = run(&data_dir, &["write", "--namespace", "default", "--content", "delta echo foxtrot"]);
    assert!(second.status.success(), "second write failed: {}", String::from_utf8_lossy(&second.stderr));

    // A clean ledger verifies fine before corruption.
    let clean = run(&data_dir, &["verify"]);
    assert!(clean.status.success(), "clean ledger failed to verify: {}", String::from_utf8_lossy(&clean.stderr));

    corrupt_ciphertext_at(&data_dir.join("ledger.redb"), 0);

    let corrupted = run(&data_dir, &["verify"]);
    assert!(!corrupted.status.success(), "verify should fail non-zero on a corrupted chain");
    let stderr = String::from_utf8_lossy(&corrupted.stderr);
    assert!(stderr.contains("seq 0"), "expected the divergent seq (0) in stderr, got: {stderr}");

    let _ = std::fs::remove_dir_all(&data_dir);
}

#[test]
fn test_cli_verify_from_skips_a_trusted_prefix() {
    let data_dir = temp_data_dir("from-skip");

    for content in ["alpha", "bravo", "charlie"] {
        let out = run(&data_dir, &["write", "--namespace", "default", "--content", content]);
        assert!(out.status.success());
    }

    // Corrupt seq 0. A full verify (from 0) still catches it...
    corrupt_ciphertext_at(&data_dir.join("ledger.redb"), 0);
    let full = run(&data_dir, &["verify"]);
    assert!(!full.status.success());

    // ...but trusting everything up through seq 1 and verifying only from
    // seq 2 onward never looks at the corrupted record, so it passes.
    let partial = run(&data_dir, &["verify", "--from", "2"]);
    assert!(partial.status.success(), "verify --from 2 should skip the corruption at seq 0: {}", String::from_utf8_lossy(&partial.stderr));

    let _ = std::fs::remove_dir_all(&data_dir);
}
