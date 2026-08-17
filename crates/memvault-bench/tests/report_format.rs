//! P2-5's acceptance check. Product doc §7's rule is that a number published
//! without its protocol is marketing, so what is tested here is the report's
//! shape -- every claim present, every figure carrying a unit, every section
//! carrying its protocol line -- and never the numbers themselves, which are
//! whatever the machine gives.

use std::process::Command;

fn run_tiny_benchmark() -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_memvault-bench"))
        .args(["--corpus-size", "50", "--queries", "20", "--hardware", "test rig"])
        .output()
        .expect("benchmark binary should run");

    assert!(
        output.status.success(),
        "benchmark exited {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("report should be utf-8")
}

#[test]
fn report_states_every_claim_with_its_protocol_and_units() {
    let report = run_tiny_benchmark();

    // The three claims of product doc §7, plus the protocol block that
    // qualifies them.
    for section in [
        "protocol",
        "retrieval latency, index-only",
        "retrieval latency, end-to-end",
        "chain verification",
        "index rebuild, full, from the ledger",
    ] {
        assert!(report.contains(section), "report is missing the {section:?} section:\n{report}");
    }

    // Measurement conditions §7 requires to be stated alongside the numbers.
    for field in ["corpus_size", "embedding_dimensions", "embedding_model", "hardware", "build_profile"] {
        assert!(report.contains(field), "report is missing the {field} protocol field:\n{report}");
    }
    assert!(report.contains("test rig"), "--hardware was not echoed into the report:\n{report}");

    // Tails matter more than the median here, so all three are mandatory.
    assert_eq!(report.matches("p50").count(), 2, "both latency figures need a p50:\n{report}");
    assert_eq!(report.matches("p95").count(), 2, "both latency figures need a p95:\n{report}");
    assert_eq!(report.matches("p99").count(), 2, "both latency figures need a p99:\n{report}");

    for unit in ["ms", "records/s", " s", "records", "dimensions", "tokens", "queries"] {
        assert!(report.contains(unit), "no figure is labeled in {unit:?}:\n{report}");
    }

    // Every section that reports a number explains how it was measured.
    assert_eq!(
        report.matches("protocol ").count(),
        4,
        "each of the four measured sections needs its own protocol line:\n{report}"
    );
}

#[test]
fn a_debug_build_says_so_rather_than_publishing_its_numbers() {
    let report = run_tiny_benchmark();
    // Integration tests build the binary in the same profile as the test, so
    // a plain `cargo test` exercises the debug branch and `--release` the
    // other. Either way the profile must be stated.
    if cfg!(debug_assertions) {
        assert!(report.contains("NOT a publishable figure"), "a debug run must disclaim its numbers:\n{report}");
    } else {
        assert!(report.contains("build_profile        release"), "a release run must say so:\n{report}");
    }
}
