#!/usr/bin/env bash
# Demo 2 (docs/IMPLEMENTATION_PLAN.md §0.1, task P1-D2): the product's
# flagship claim as one continuous, watchable story -- the agent remembers
# across a crash, and forgetting is provable rather than trusted.
#
# Non-interactive and checked for exact outcomes (grep/exit code), not
# eyeballed: this is what runs live in a demo, and CI runs it too so it
# can't silently rot.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="$repo_root/demo/data2"
rm -rf "$data_dir"

cargo build -p memvault-cli -p memvault-server --quiet

cli_bin="$repo_root/target/debug/memvault"
[ -x "$cli_bin" ] || cli_bin="$repo_root/target/debug/memvault.exe"
server_bin="$repo_root/target/debug/memvault-server"
[ -x "$server_bin" ] || server_bin="$repo_root/target/debug/memvault-server.exe"

mv_() { "$cli_bin" --data-dir "$data_dir" "$@"; }
extract_fact_id() { grep -oE '^fact_id: [0-9a-fA-F-]{36}' | cut -d' ' -f2; }
extract_any_uuid() { grep -oE '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}' | head -1; }

echo "== 1. writing facts via the CLI =="
fact_a=$(mv_ write --namespace default --content "the deploy pipeline runs every saturday at noon" | extract_fact_id)
mv_ write --namespace default --content "the staging database is called shadow" >/dev/null
big_content=$(printf 'padding content repeated to force a budget cut %.0s' {1..30})
mv_ write --namespace default --content "$big_content" >/dev/null
mv_ write --namespace default --content "an unrelated short fact" >/dev/null

echo "== 2. searching, with full provenance =="
search_output=$(mv_ search --namespace default --query "deploy pipeline" --k 5 --max-tokens 15)
echo "$search_output"
retrieval_id=$(echo "$search_output" | grep -oE 'retrieval_id: [0-9a-fA-F-]{36}' | cut -d' ' -f2)
[ -n "$retrieval_id" ] || { echo "FAIL: no retrieval_id printed by search"; exit 1; }
echo "$search_output" | grep -q "CutBy" || { echo "FAIL: expected a rejected candidate in the search output"; exit 1; }

echo "== 3. starting the MCP server, then SIGKILL mid-burst =="
coproc SERVER { "$server_bin" "$data_dir" 2>/dev/null; }
send() { printf '%s\n' "$1" >&"${SERVER[1]}"; }
recv() { local line; read -r -u "${SERVER[0]}" line; printf '%s' "$line"; }

send '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"demo2","version":"0.1.0"}}}'
recv >/dev/null
send '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# A handful of confirmed (response awaited, so guaranteed durable) writes.
# recv's output is captured to a plain variable first, never piped
# directly (`recv | cmd`) -- piping straight from a coproc-fd read loses
# the fd on this bash build, a known coprocess/pipeline interaction.
confirmed_ids=()
for i in 1 2 3; do
    send "{\"jsonrpc\":\"2.0\",\"id\":$i,\"method\":\"tools/call\",\"params\":{\"name\":\"memory_write\",\"arguments\":{\"namespace\":\"default\",\"content\":\"crash-survivor fact $i\"}}}"
    write_resp=$(recv)
    confirmed_ids+=("$(echo "$write_resp" | extract_any_uuid)")
done
# ...then a burst fired without waiting for responses, followed by an
# immediate hard kill -- some of these may or may not have landed.
for i in 4 5 6 7 8 9 10; do
    send "{\"jsonrpc\":\"2.0\",\"id\":$i,\"method\":\"tools/call\",\"params\":{\"name\":\"memory_write\",\"arguments\":{\"namespace\":\"default\",\"content\":\"burst fact $i\"}}}"
done
server_pid="$SERVER_PID"
kill -9 "$server_pid" 2>/dev/null || true
# bash unsets NAME_PID once the coproc is reaped, so the wait above
# must be the last reference to $SERVER_PID itself.
wait "$server_pid" 2>/dev/null || true
echo "server killed mid-burst (pid $server_pid)"

echo "== 4. recovery: same path the server runs automatically at every startup =="
# The killed process's ledger file lock isn't always released the instant
# `wait` returns -- retry briefly rather than racing it with a fixed sleep.
replay_output=""
for attempt in $(seq 1 20); do
    if replay_output=$(mv_ replay 2>&1); then
        break
    fi
    if [ "$attempt" -eq 20 ]; then
        echo "FAIL: could not replay after the kill: $replay_output"
        exit 1
    fi
    sleep 0.25
done
echo "$replay_output"
mv_ verify

post_kill_search=$(mv_ search --namespace default --query "crash-survivor" --k 10 --max-tokens 4096)
echo "$post_kill_search"
for id in "${confirmed_ids[@]}"; do
    echo "$post_kill_search" | grep -q "$id" || { echo "FAIL: confirmed fact $id missing after recovery"; exit 1; }
done

echo "== 5. explain: reconstructing the retrieval from step 2 =="
explain_output=$(mv_ explain "$retrieval_id")
echo "$explain_output"
explain_rows=$(echo "$explain_output" | grep -cE '^[0-9a-fA-F-]{36}')
search_rows=$(echo "$search_output" | grep -cE '^[0-9a-fA-F-]{36}')
[ "$explain_rows" = "$search_rows" ] || { echo "FAIL: explain returned $explain_rows rows, search produced $search_rows"; exit 1; }
echo "$explain_output" | grep -q "CutBy" || { echo "FAIL: explain lost the rejected candidate from step 2"; exit 1; }

echo "== 6. forget: cryptographic erase =="
mv_ forget "$fact_a" --reason "demo cleanup"
mv_ verify

post_forget_search=$(mv_ search --namespace default --query "deploy pipeline" --k 10 --max-tokens 4096)
if echo "$post_forget_search" | grep -q "$fact_a"; then
    echo "FAIL: forgotten fact still returned by search"
    exit 1
fi

# Walk the raw ledger until we find fact_a's own Assert record: still
# present, but its content now reads as undecryptable.
seq=0
dump=""
while raw=$(mv_ dump-record "$seq" 2>/dev/null); do
    if echo "$raw" | grep -q "$fact_a"; then
        dump="$raw"
        break
    fi
    seq=$((seq + 1))
done
[ -n "$dump" ] || { echo "FAIL: could not find the forgotten fact's raw ledger record"; exit 1; }
echo "$dump"
echo "$dump" | grep -qi "undecryptable" || { echo "FAIL: raw record does not show as undecryptable"; exit 1; }

echo
echo "PASS: wrote and searched with a rejected candidate, survived a hard"
echo "kill mid-burst with recovery bringing every confirmed write back,"
echo "reconstructed the original retrieval exactly via explain, and"
echo "cryptographically forgot a fact -- its ledger record remains, still"
echo "chain-verified, but is now provably unreadable."
