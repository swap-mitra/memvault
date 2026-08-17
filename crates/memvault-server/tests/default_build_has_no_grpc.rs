//! P2-2's first acceptance criterion: the default build must not pay for the
//! optional gRPC transport. Feature unification across a workspace is easy to
//! break by accident (one crate turning the feature on drags tonic into
//! everyone's tree), so this asserts on the real resolved graph rather than
//! trusting the Cargo.toml to still say `optional = true`.

use std::process::Command;

#[test]
fn default_dependency_tree_contains_no_grpc_stack() {
    let output = Command::new(env!("CARGO"))
        .args(["tree", "--package", "memvault-server", "--edges", "normal,build", "--prefix", "none"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("cargo tree should run");

    assert!(
        output.status.success(),
        "cargo tree failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let tree = String::from_utf8_lossy(&output.stdout);
    for crate_name in ["tonic ", "prost ", "tonic-prost ", "protoc-bin-vendored "] {
        assert!(
            !tree.contains(crate_name),
            "{crate_name}is in the default dependency tree; the grpc feature is leaking:\n{tree}"
        );
    }
}
