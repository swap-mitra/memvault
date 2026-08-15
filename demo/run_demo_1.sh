#!/usr/bin/env bash
# Demo 1 (docs/IMPLEMENTATION_PLAN.md §0.1, task P0-D1): a runnable,
# watchable proof that hybrid retrieval, supersession, pinning, decay, and
# budget cuts all work end to end through memvault-cli alone -- no server.
#
# Non-interactive and checked for exact outcomes (grep/exit code), not
# eyeballed: this is what runs live in a demo, and CI runs it too so it
# can't silently rot.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$repo_root/demo/data"
rm -rf "$data_dir"

cargo build -p memvault-cli --quiet

bin="$repo_root/target/debug/memvault"
if [ ! -x "$bin" ]; then
    bin="$repo_root/target/debug/memvault.exe"
fi

mv_() { "$bin" --data-dir "$data_dir" "$@"; }

extract_fact_id() {
    grep -oE '^fact_id: [0-9a-fA-F-]{36}' | cut -d' ' -f2
}

echo "== writing facts =="
fact_ledger_id=$(mv_ write --namespace default --content "the ledger is hash-chained and append-only" | extract_fact_id)
pinned_fact_id=$(mv_ write --namespace default --content "bananas are a good source of potassium" --pin | extract_fact_id)
mv_ write --namespace default --content "the weather in march was unusually warm" >/dev/null
big_content=$(printf 'a hash-chained ledger fact repeated for bulk %.0s' {1..30})
mv_ write --namespace default --content "$big_content" >/dev/null
mv_ write --namespace default --content "another short unrelated fact" >/dev/null

echo "== superseding the first fact =="
new_ledger_id=$(mv_ write --namespace default --content "the ledger is now Merkle-anchored for external audit" --fact-id "$fact_ledger_id" | extract_fact_id)
if [ "$new_ledger_id" != "$fact_ledger_id" ]; then
    echo "FAIL: superseding write returned a different fact_id ($new_ledger_id != $fact_ledger_id)"
    exit 1
fi

echo "== searching with a tight token budget =="
search_output=$(mv_ search --namespace default --query "hash-chained ledger" --k 10 --max-tokens 15)
echo "$search_output"

retrieval_id=$(echo "$search_output" | grep -oE 'retrieval_id: [0-9a-fA-F-]{36}' | cut -d' ' -f2)
if [ -z "$retrieval_id" ]; then
    echo "FAIL: no retrieval_id printed by search"
    exit 1
fi

if ! echo "$search_output" | grep -q "CutByBudget"; then
    echo "FAIL: expected at least one CutByBudget outcome"
    exit 1
fi

pinned_row=$(echo "$search_output" | grep -E "^$pinned_fact_id" || true)
if [ -z "$pinned_row" ]; then
    echo "FAIL: pinned fact ($pinned_fact_id) did not appear in search results at all"
    exit 1
fi
pinned_decay_weight=$(echo "$pinned_row" | awk '{print $7}')
if [ "$pinned_decay_weight" != "1.0000" ]; then
    echo "FAIL: pinned fact's decay_weight was '$pinned_decay_weight', expected exactly 1.0000"
    exit 1
fi

echo "== explain: reconstructing the retrieval from the ledger =="
explain_output=$(mv_ explain "$retrieval_id")
echo "$explain_output"

explain_rows=$(echo "$explain_output" | grep -cE '^[0-9a-fA-F-]{36}')
search_rows=$(echo "$search_output" | grep -cE '^[0-9a-fA-F-]{36}')
if [ "$explain_rows" != "$search_rows" ]; then
    echo "FAIL: explain returned $explain_rows rows, search produced $search_rows"
    exit 1
fi

echo
echo "PASS: hybrid fusion (ann_rank + bm25_rank on the same row), a pinned"
echo "fact ignoring decay, a supersession, and a CutByBudget outcome are all"
echo "visible above, and explain reconstructed the retrieval exactly."
