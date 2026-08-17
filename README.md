# MemVault

A local-first memory engine for AI agents. One binary, no service to operate,
no network required.

Every fact it holds and every retrieval decision it makes is recorded in an
append-only, hash-chained ledger — so "why did the agent know that?" always
has an answer, and "forget this" is something you can prove rather than
trust.

MemVault does not call an LLM, extract facts from conversation, or train on
anything. It stores what you give it and explains what it hands back.

---

## Requirements

- **Rust 1.85 or newer** (the workspace uses edition 2024). `rustup update` if
  `cargo build` complains about the edition.
- A C++ toolchain, which the vector index (`usearch`) needs to build:
  Xcode command line tools on macOS, `build-essential` on Debian/Ubuntu,
  Visual Studio Build Tools on Windows.
- Python 3.9+ only if you want the Python bindings.

Nothing else. No database to provision, no service to keep alive.

```sh
git clone https://github.com/swap-mitra/memvault
cd memvault
cargo build --release
```

That produces two binaries in `target/release/`:

| Binary            | What it is                                        |
|-------------------|---------------------------------------------------|
| `memvault`        | CLI — write, search, explain, verify, forget      |
| `memvault-server` | MCP server over stdio, for agents to talk to      |

The examples below use `./target/release/memvault`. Put it on your `PATH` if
you'd rather not type that.

---

## Quickstart 1 — the CLI, in about a minute

No server, no config. Write three facts:

```sh
memvault --data-dir ./my-memory write --namespace project \
  --content "the deploy script lives in ops/deploy.sh"
memvault --data-dir ./my-memory write --namespace project \
  --content "staging runs postgres 16"
memvault --data-dir ./my-memory write --namespace project \
  --content "the on-call rotation is in PagerDuty schedule P7" --pin
```

Each prints the id it assigned:

```
fact_id: 34e2001b-1603-45c5-abbe-f5fbcf31d4a8
```

Now search, with a deliberately small token budget so you can see something
get cut:

```sh
memvault --data-dir ./my-memory search --namespace project \
  --query "deploy script" --max-tokens 20
```

```
retrieval_id: f96a229c-4d63-4db8-834d-1b98877611dd
fact_id                              ann_rank   ann_dist  bm25_rk bm25_score       rrf  decay_wt     final       outcome tokens
34e2001b-1603-45c5-abbe-f5fbcf31d4a8        0     0.3518        0     2.2232    0.0328    1.0000    0.0328      Injected     15
fdaa5149-440e-4a42-b2e6-d67653f2871d        1     0.6311        -          -    0.0161    1.0000    0.0161   CutByBudget     17
b5aa16a8-aa14-48cb-917d-1bc5d048f184        2     1.0000        -          -    0.0159    1.0000    0.0159   CutByBudget     11
```

That table is the point of the product, so it is worth reading across:

| Column | Meaning |
|---|---|
| `ann_rank` / `ann_dist` | Where the fact placed on the vector-similarity axis, and how far. `-` means it didn't surface there at all. |
| `bm25_rk` / `bm25_score` | Same, for the keyword axis. Rows with a rank on both axes were found two different ways. |
| `rrf` | The two ranks fused. Ranks only — never raw scores, since a cosine distance and a BM25 score aren't on the same scale. |
| `decay_wt` | How much age discounted it. `1.0000` means no discount — a pinned fact never decays. |
| `final` | `rrf × decay_wt`, what the ordering is actually by. |
| `outcome` | `Injected` made it into the answer. `CutByBudget` lost to the token budget, `CutByK` to the `--k` limit, `FilteredByTime` was no longer valid at query time. |
| `tokens` | What that fact would cost you in context. |

**Every candidate considered appears here, including the ones that were
cut.** That is what makes a bad retrieval debuggable instead of mysterious.

---

## Quickstart 2 — connect it to an MCP client

This is the setup an actual agent uses. Point your MCP client at
`memvault-server`, giving it a data directory as its one argument:

```json
{
  "mcpServers": {
    "memvault": {
      "command": "/absolute/path/to/memvault/target/release/memvault-server",
      "args": ["/absolute/path/to/my-memory"]
    }
  }
}
```

- **Claude Code** — save this as `.mcp.json` in your project root.
- **Claude Desktop** — merge it into `claude_desktop_config.json`.
- Any other MCP client — same shape; it is a plain stdio server.

Use absolute paths for both. The server inherits whatever working directory
the client happens to launch it from, which is rarely the one you expect.

Restart the client, and the agent has six tools:

| Tool | What it does |
|---|---|
| `memory_write` | Assert a fact. Pass an existing `fact_id` to supersede that fact instead. |
| `memory_search` | Hybrid retrieval with the full provenance table above. |
| `memory_as_of` | What was true — or what the engine believed — at a given moment. |
| `memory_supersede` | Close a fact's interval without asserting a replacement. |
| `memory_forget` | Cryptographic erase. |
| `memory_explain` | Reconstruct any past retrieval from its `retrieval_id`. |

On startup the server runs recovery and verifies the chain, logging the
result to stderr. Stdout is the protocol channel and carries nothing else.

### Checking it works without a client

If the agent isn't seeing the tools and you want to know whether the problem
is the server or the client, drive it directly. Save as `smoke.py`:

```python
import json, subprocess, sys

proc = subprocess.Popen(
    ["./target/release/memvault-server", "./my-memory"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
)

def call(msg, wants_response=True):
    proc.stdin.write(json.dumps(msg) + "\n"); proc.stdin.flush()
    return json.loads(proc.stdout.readline()) if wants_response else None

call({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {},
    "clientInfo": {"name": "smoke", "version": "0.1.0"}}})
call({"jsonrpc": "2.0", "method": "notifications/initialized"}, wants_response=False)

print(call({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
    "name": "memory_write",
    "arguments": {"namespace": "project",
                  "content": "the deploy script lives in ops/deploy.sh"}}
})["result"]["content"][0]["text"])

print(call({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
    "name": "memory_search",
    "arguments": {"namespace": "project", "query": "deploy script"}}
})["result"]["content"][0]["text"])

proc.terminate()
```

```
memvault-server: recovery report: RecoveryReport { replayed_from: None, rebuilt: [], verified: true }
fact_id: 77e049f3-bd99-4b7d-b211-1401ccf44a5a
retrieval_id: a3be11c3-ef33-4851-b50f-80a905ce5cb4
fact_id                              ann_rank   ann_dist  bm25_rk bm25_score       rrf  decay_wt     final       outcome tokens
77e049f3-bd99-4b7d-b211-1401ccf44a5a        -          -        0     0.6832    0.0164    1.0000    0.0164      Injected     15
```

Three things about that output surprise people:

**That first line is not an error.** The server reports what recovery found
on stderr every time it starts; `verified: true` means the chain checked out.
Only stdout carries protocol traffic.

**Read each response before sending the next request.** The server handles
requests concurrently, so a script that pipes every line in at once can have
its search answered before its write finishes — and get an empty result that
looks like a bug. Real MCP clients already wait per response; hand-rolled
smoke tests are where this bites.

**`ann_rank` is `-` here.** Embeddings are caller-supplied: MemVault runs no
model, so if your client doesn't pass an `embedding`, only the keyword axis
runs. That's a working configuration, not a broken one — it is just BM25
rather than hybrid retrieval. The CLI shows both axes because it generates a
stand-in vector; see [Known limits](#known-limits).

---

## Quickstart 3 — Python, in-process

For callers who want the engine in the same process, with no subprocess and
no event loop:

```sh
pip install maturin
maturin build -m crates/memvault-ffi/Cargo.toml --release
pip install --find-links target/wheels memvault
```

```python
import memvault

mv = memvault.MemVault("./my-memory")
fact_id = mv.write("project", "the deploy script lives in ops/deploy.sh")

retrieval_id, explanations = mv.search("project", "deploy script")
for e in explanations:
    print(e.fact_id, e.outcome, e.final_score, e.token_cost)

mv.verify()  # raises if the chain is broken
```

The API is synchronous, and every call releases the GIL while the engine
works. `Explanation` carries the same fields as the table above, so
provenance doesn't get thinner just because you came in this way.

---

## Proving the accountability claims

Two of MemVault's three commitments are things you can check yourself rather
than take on faith.

**Any past retrieval reconstructs exactly**, including what was cut, from its
id alone — it's a ledger read, not a re-run:

```sh
memvault --data-dir ./my-memory explain f96a229c-4d63-4db8-834d-1b98877611dd
```

**Forgetting is provable.** Erase a fact, and the chain still verifies:

```sh
memvault --data-dir ./my-memory verify
# chain verified from seq 0

memvault --data-dir ./my-memory forget 34e2001b-1603-45c5-abbe-f5fbcf31d4a8 \
  --reason "customer deletion request"
# forgot fact_id: 34e2001b-1603-45c5-abbe-f5fbcf31d4a8

memvault --data-dir ./my-memory verify
# chain verified from seq 0
```

The fact is gone from search, but its record is still in the ledger — the
history stays honest, and the content is unrecoverable because its key was
destroyed:

```sh
memvault --data-dir ./my-memory dump-record 0
```

```
seq 0 kind Assert recorded_at 2026-08-17T18:30:07.541767+00:00
fact_id 34e2001b-...  content_hash 76d44f5d...  ciphertext_len 56
content: undecryptable (key destroyed or never existed)
```

The `content_hash` survives, so anyone holding the original plaintext can
still prove what that record used to say. Nobody can recover it from here.

There are two narrated end-to-end demos in `demo/` — `run_demo_1.sh` covers
retrieval and provenance, `run_demo_2.sh` kills the server mid-write and
shows recovery, verification, and erasure.

---

## Optional: gRPC

For multi-process deployments where stdio isn't available. Off by default, so
the primary local-agent case doesn't pay for it:

```sh
cargo build -p memvault-server --features grpc
MEMVAULT_GRPC_ADDR=127.0.0.1:50051 ./target/debug/memvault-server ./my-memory
```

Same six operations, structured messages instead of text. The schema is
`crates/memvault-server/proto/memvault.proto`. A default build rejects
`MEMVAULT_GRPC_ADDR` rather than ignoring it, so a half-configured deployment
fails at startup instead of quietly serving the wrong thing.

---

## Known limits

Stated plainly, because each one will otherwise look like a bug:

- **Embeddings are caller-supplied.** MemVault runs no model. The MCP and
  Python surfaces accept an `embedding` and use keyword-only retrieval
  without one. The CLI and demos hash trigrams into a vector so the fusion
  machinery has something to run on — that stand-in is *not* semantically
  meaningful, and no number produced with it should be read as retrieval
  quality.
- **`namespace` does not filter search results yet.** It is recorded on
  every write and every retrieval, and `memory_as_of` respects it, but
  `memory_search` currently scores across all namespaces in a data
  directory. Until that's fixed, use a separate `--data-dir` per tenant if
  you need isolation.
- **Token counts are estimates** — ciphertext bytes / 4, not a tokenizer.
  Close enough for budgeting English prose, drifting on code.
- **Decay measures from a fact's own start**, not from last access, so
  retrieval does not yet reinforce a fact against decay.

---

## Benchmarks

`benchmarks/` holds the LongMemEval and LOCOMO harnesses and its own README.
`cargo run --release -p memvault-bench` reports retrieval latency, chain
verification throughput, and index rebuild time, each printed with the
protocol that produced it.

---

## Layout

```
crates/
  memvault-core/     the engine: ledger, crypto, indexes, read and write paths
  memvault-server/   MCP server over stdio; optional gRPC behind a feature
  memvault-cli/      the memvault binary
  memvault-ffi/      Python bindings (pyo3)
  memvault-bench/    latency, verification, and rebuild benchmarks
benchmarks/          LongMemEval and LOCOMO harnesses
demo/                narrated end-to-end demo scripts
```

## License

MIT OR Apache-2.0
