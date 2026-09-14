# Changelog

Notable changes, newest first. MemVault is pre-1.0: the on-disk format and
the Python API may change between minor versions, and this file is where
that is announced.

## 0.1.0 (2026-09-14)

First tagged release. Prebuilt `memvault` and `memvault-server` for Linux
x86_64, macOS (Intel and Apple silicon) and Windows x86_64, plus Python
wheels for the same four platforms, are attached to the GitHub release.

### The engine

- Hybrid retrieval: vector ANN (usearch) and BM25 (tantivy), fused by
  reciprocal rank, weighted by decay, packed to a token budget.
- Append-only, hash-chained ledger (redb). Every write and every search is
  a record; `verify` walks the chain, `replay` rebuilds both indexes from it.
- Provenance: every search returns a `retrieval_id` and a row for every
  candidate considered, including the ones cut, and `explain` replays any
  past retrieval from the ledger alone.
- Bitemporal facts: `valid_from`/`valid_to` on every fact, `as_of` on both
  the valid-time and the transaction-time axis.
- Cryptographic erase: `forget` destroys a fact's key; the record stays, the
  chain still verifies, the content is unrecoverable.
- Crash safety: a hard kill mid-write leaves the ledger correct and
  recovery reconciles the indexes at the next start.

### Surfaces

- MCP server over stdio with seven tools: `memory_write`, `memory_search`,
  `memory_get`, `memory_as_of`, `memory_supersede`, `memory_forget`,
  `memory_explain`. Every tool answers with structured JSON. The server's
  instructions tell the agent when to write, how to recall, and what to do
  when a fact changes.
- Optional embedding provider: `MEMVAULT_EMBED_URL` and
  `MEMVAULT_EMBED_MODEL` point the server at any OpenAI-compatible
  `/embeddings` endpoint, and it embeds writes and queries itself. Without
  it, retrieval over MCP is keyword-only.
- `memvault mcp-config` prints the client configuration with absolute paths.
- Python bindings (`import memvault`), synchronous, GIL released during
  engine work. `search()` returns a `SearchResult` with `injected`
  (content, best first) and `candidates` (provenance rows).
- gRPC transport behind `--features grpc`, same operations as protobuf.
- `--features tokenizer`: real cl100k_base token counts instead of the
  bytes/4 estimate.

### Known limits

See the README's Known limits table. The two that matter most in daily
use: every search appends a Retrieval record, so the ledger grows with
reads as well as writes and there is no compaction yet; and decay is
measured from a fact's own start, not its last access.
