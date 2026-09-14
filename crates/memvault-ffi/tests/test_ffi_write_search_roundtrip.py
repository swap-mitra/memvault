"""P2-1's acceptance test: the built wheel gives a plain Python script the
same engine and the same Explanation fields as the MCP path, with no event
loop anywhere.

Run against an installed wheel:

    maturin develop -m crates/memvault-ffi/Cargo.toml
    python crates/memvault-ffi/tests/test_ffi_write_search_roundtrip.py

pytest picks the same functions up if you'd rather run it that way.
"""

import tempfile

import memvault

# Same field list as memvault_core::Explanation (product doc §6.4). If the
# engine grows a field and the FFI doesn't forward it, this test fails.
EXPLANATION_FIELDS = [
    "fact_id",
    "ledger_seq",
    "ann_rank",
    "ann_distance",
    "bm25_rank",
    "bm25_score",
    "rrf_score",
    "decay_weight",
    "final_score",
    "outcome",
    "token_cost",
]

NS = "ffi-test"


def test_write_search_roundtrip():
    with tempfile.TemporaryDirectory() as data_dir:
        mv = memvault.MemVault(data_dir)
        fact_id = mv.write(NS, "the deploy script lives in ops/deploy.sh")
        mv.write(NS, "the staging database is postgres 16")

        result = mv.search(NS, "deploy script")
        assert result.candidates, "search returned no candidates at all"

        # The agent gets the text, best first; only Injected facts have any.
        assert result.injected[0].fact_id == fact_id
        assert result.injected[0].content == "the deploy script lives in ops/deploy.sh"
        injected_ids = {i.fact_id for i in result.injected}
        assert injected_ids == {e.fact_id for e in result.candidates if e.outcome == "Injected"}

        found = [e for e in result.candidates if e.fact_id == fact_id]
        assert found, f"{fact_id} missing from {[e.fact_id for e in result.candidates]}"
        assert found[0].outcome == "Injected"
        assert found[0].bm25_rank is not None, "no BM25 rank -> keyword axis didn't run"

        for field in EXPLANATION_FIELDS:
            assert hasattr(found[0], field), f"Explanation is missing {field}"

        # explain() is a ledger read, so it must reproduce the search exactly.
        replayed = mv.explain(result.retrieval_id)
        assert [(e.fact_id, e.outcome, e.final_score) for e in replayed] == [
            (e.fact_id, e.outcome, e.final_score) for e in result.candidates
        ]


def test_pinned_fact_bypasses_decay():
    with tempfile.TemporaryDirectory() as data_dir:
        mv = memvault.MemVault(data_dir)
        pinned_id = mv.write(NS, "never forget: prod credentials rotate monthly", pinned=True)

        pinned = next(e for e in mv.search(NS, "prod credentials rotate").candidates if e.fact_id == pinned_id)
        assert pinned.decay_weight == 1.0


def test_forget_removes_from_search_and_leaves_chain_verifying():
    with tempfile.TemporaryDirectory() as data_dir:
        mv = memvault.MemVault(data_dir)
        fact_id = mv.write(NS, "the api key is stored in vault at secret/api")

        before = mv.search(NS, "api key vault").candidates
        assert fact_id in [e.fact_id for e in before]

        assert mv.get(fact_id).content == "the api key is stored in vault at secret/api"
        mv.forget(fact_id, reason="test erasure")

        after = mv.search(NS, "api key vault").candidates
        assert fact_id not in [e.fact_id for e in after]
        assert mv.get(fact_id) is None, "a forgotten fact has no readable version"
        mv.verify()  # raises if erasure broke the chain


def test_invalid_input_raises_rather_than_panics():
    with tempfile.TemporaryDirectory() as data_dir:
        mv = memvault.MemVault(data_dir)

        try:
            mv.write(NS, "x", fact_id="not-a-uuid")
        except ValueError:
            pass
        else:
            raise AssertionError("a malformed fact_id should raise ValueError")

        try:
            mv.write(NS, "x", embedding=[0.1, 0.2])  # wrong dimensionality
        except memvault.MemVaultError:
            pass
        else:
            raise AssertionError("a wrong-dimension embedding should raise MemVaultError")


def test_embedding_dim_is_fixed_at_creation():
    with tempfile.TemporaryDirectory() as data_dir:
        mv = memvault.MemVault(data_dir, embedding_dim=4)
        assert mv.embedding_dim == 4
        fact_id = mv.write(NS, "four wide", embedding=[1.0, 0.0, 0.0, 0.0])
        del mv  # releases the ledger's exclusive lock

        # Reopened without saying: the directory remembers.
        mv = memvault.MemVault(data_dir)
        assert mv.embedding_dim == 4
        result = mv.search(NS, embedding=[1.0, 0.0, 0.0, 0.0])
        assert result.injected[0].fact_id == fact_id
        del mv

        try:
            memvault.MemVault(data_dir, embedding_dim=8)
        except memvault.MemVaultError:
            pass
        else:
            raise AssertionError("a conflicting embedding_dim should be refused")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all ffi tests passed")
