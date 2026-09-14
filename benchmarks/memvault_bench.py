"""Shared plumbing for the LongMemEval and LOCOMO harnesses.

Both benchmarks have the same shape: a haystack of conversation turns, a
question, and a gold answer. MemVault's job is only the middle step -- given
the haystack and the question, decide what goes in the model's context. So
this module ingests turns, retrieves for a question, and reports what the
retrieval cost. Generation and grading stay with each benchmark's own
scripts, which is what "run unmodified" means (plan tasks P2-3/P2-4).

Requires the `memvault` wheel (see the README's Install section).

Retrieval quality depends on the embedding model, and MemVault has none of
its own, so without `--embed-url`/`--embed-model` the harness runs keyword
(BM25) retrieval only. Any OpenAI-compatible `/embeddings` endpoint works,
the same way the server's MEMVAULT_EMBED_URL does; the summary names the
model so a number never travels without it.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

import memvault

# Input price per million tokens, from the published Claude API rates. Only
# the input side is here on purpose: a memory layer decides what goes into
# the prompt and nothing about what comes out, so input cost is the figure
# it is accountable for. Anyone comparing total turn cost has to add their
# own output-token measurement, which is a property of the generator.
INPUT_PRICE_PER_MTOK = {
    "claude-opus-5": 5.00,
    "claude-sonnet-5": 3.00,
    "claude-haiku-4-5": 1.00,
}

NAMESPACE = "bench"


class Embedder:
    """An OpenAI-compatible `/embeddings` endpoint, called with the standard
    library so the harness adds no dependency. Ollama, OpenAI and Voyage all
    speak this shape."""

    def __init__(self, url, model, api_key=None):
        self.endpoint = url.rstrip("/") + "/embeddings"
        self.model = model
        self.api_key = api_key

    def embed(self, text):
        body = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        embedding = payload["data"][0]["embedding"]
        if not embedding:
            raise RuntimeError(f"{self.endpoint} returned an empty embedding for model {self.model!r}")
        return embedding


def embedder_from(args):
    """The provider the flags describe, or None for keyword-only retrieval."""
    if bool(args.embed_url) != bool(args.embed_model):
        sys.exit("--embed-url and --embed-model must be given together")
    if not args.embed_url:
        return None
    return Embedder(args.embed_url, args.embed_model, args.embed_api_key)


@dataclass
class RetrievalCost:
    """What one retrieval call cost, in the terms product doc §7 asks for."""

    injected: int = 0
    considered: int = 0
    cut_by_budget: int = 0
    cut_by_k: int = 0
    filtered_by_time: int = 0
    tokens: int = 0

    def cost_usd(self, model: str) -> float:
        return self.tokens / 1_000_000 * INPUT_PRICE_PER_MTOK[model]


@dataclass
class Turn:
    """One utterance from a haystack, with when it happened."""

    text: str
    when: datetime
    session_id: str = ""


@dataclass
class Store:
    """A MemVault instance over one question's haystack.

    ponytail: one data directory per question, not one namespace per
    question. Namespaces do isolate results, but they filter after fusion
    out of one shared candidate pool, so a single directory holding every
    question's haystack would let the corpus crowd each question's own
    evidence out of that pool and depress the score for a reason that has
    nothing to do with retrieval quality. The cost is a store open per
    question. Upgrade path: one namespace per question, once the indexes
    filter before fusion rather than after.
    """

    embedder: Embedder = None
    data_dir: str = field(default_factory=lambda: tempfile.mkdtemp(prefix="memvault-bench-"))
    _mv: memvault.MemVault = field(init=False)

    def __post_init__(self):
        if self.embedder is None:
            self._mv = memvault.MemVault(self.data_dir)
        else:
            # The directory's width is fixed at creation, so learn it from
            # the model rather than trusting a flag.
            width = len(self.embedder.embed("memvault embedding width probe"))
            self._mv = memvault.MemVault(self.data_dir, embedding_dim=width)

    def _embed(self, text):
        return self.embedder.embed(text) if self.embedder else None

    def ingest(self, turns):
        for turn in turns:
            self._mv.write(
                NAMESPACE,
                turn.text,
                embedding=self._embed(turn.text),
                valid_from=turn.when.isoformat(),
                source=turn.session_id or None,
            )

    def retrieve(self, question, k=10, max_tokens=2048):
        """Return (context_strings, RetrievalCost) for one question."""
        result = self._mv.search(NAMESPACE, question, embedding=self._embed(question), k=k, max_tokens=max_tokens)

        cost = RetrievalCost(considered=len(result.candidates))
        for e in result.candidates:
            if e.outcome == "Injected":
                cost.injected += 1
                cost.tokens += e.token_cost
            elif e.outcome == "CutByBudget":
                cost.cut_by_budget += 1
            elif e.outcome == "CutByK":
                cost.cut_by_k += 1
            elif e.outcome == "FilteredByTime":
                cost.filtered_by_time += 1
        return [f.content for f in result.injected], cost

    def close(self):
        del self._mv
        shutil.rmtree(self.data_dir, ignore_errors=True)


# Each benchmark stamps its dates differently and neither is ISO. Parsing is
# not optional: valid_from drives decay and every temporal-reasoning
# question, so a silently-wrong date quietly changes the score. `%a` and `%B`
# read English names, which is what both datasets ship and what the C locale
# Python starts in expects -- neither script calls setlocale.
_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",  # LongMemEval
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%I:%M %p on %d %B, %Y",  # LOCOMO
    "%I:%M %p on %d %B %Y",
)


def parse_timestamp(raw):
    """Parse either benchmark's timestamp format into an aware UTC datetime.

    Raises on anything unrecognized rather than defaulting to now(): a
    fabricated date would silently corrupt every temporal question in the run.
    """
    raw = raw.strip()
    for fmt in _FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unrecognized timestamp format: {raw!r}")


def summarize(costs, model="claude-opus-5", embedding_model=None):
    """Aggregate per-retrieval costs into the figures product doc §7 requires."""
    if not costs:
        return {}
    n = len(costs)
    total_tokens = sum(c.tokens for c in costs)
    return {
        "retrieval_calls": n,
        "tokens_per_retrieval_call_mean": total_tokens / n,
        "tokens_per_retrieval_call_max": max(c.tokens for c in costs),
        "injected_per_call_mean": sum(c.injected for c in costs) / n,
        "considered_per_call_mean": sum(c.considered for c in costs) / n,
        "cut_by_budget_total": sum(c.cut_by_budget for c in costs),
        "cut_by_k_total": sum(c.cut_by_k for c in costs),
        "filtered_by_time_total": sum(c.filtered_by_time for c in costs),
        "input_cost_per_turn_usd": total_tokens / n / 1_000_000 * INPUT_PRICE_PER_MTOK[model],
        "input_price_model": model,
        "input_price_per_mtok_usd": INPUT_PRICE_PER_MTOK[model],
        "embedding_model": embedding_model or "none: keyword-only (BM25) retrieval",
        "token_cost_basis": "memvault Explanation.token_cost: ciphertext bytes / 4 unless the wheel was built with --features tokenizer",
    }


def parse_args(description, dataset_help, default_out):
    """The argument set both harnesses take. They differ only in wording."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help=dataset_help)
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, help="Only run the first N samples (smoke test)")
    ap.add_argument("--model", default="claude-opus-5", help="Model whose input price prices the context")
    ap.add_argument("--embed-url", help="OpenAI-compatible embeddings base URL, e.g. http://localhost:11434/v1")
    ap.add_argument("--embed-model", help="Embedding model name at that URL, e.g. nomic-embed-text")
    ap.add_argument(
        "--embed-api-key",
        default=os.environ.get("MEMVAULT_EMBED_API_KEY"),
        help="Bearer token for the embeddings endpoint (default: $MEMVAULT_EMBED_API_KEY)",
    )
    return ap.parse_args()


def run(items, unit_fn, args, progress):
    """Retrieve for every question and write one JSONL row each.

    `unit_fn(item)` returns (turns, [(question, row_fields), ...]): the turns
    sharing one haystack, and the questions asked against it. One Store per
    item, for the reason in `Store`'s own docstring.

    `progress(n, total, questions)` renders the stderr progress line, which
    counts different things in each benchmark.
    """
    embedder = embedder_from(args)
    costs = []
    questions = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for n, item in enumerate(items, 1):
            store = Store(embedder=embedder)
            try:
                turns, asks = unit_fn(item)
                store.ingest(turns)
                for question, row in asks:
                    context, cost = store.retrieve(question, k=args.k, max_tokens=args.max_tokens)
                    costs.append(cost)
                    questions += 1
                    row = {**row, "question": question, "retrieved_context": context, "retrieval_cost": vars(cost)}
                    out.write(json.dumps(row) + "\n")
            finally:
                store.close()
            print(f"\r{progress(n, len(items), questions)}", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    return costs, questions


def report(costs, args, extra, scripts):
    """Print the cost summary the product doc requires, and where to go next."""
    summary = summarize(costs, model=args.model, embedding_model=args.embed_model)
    summary.update(extra)
    summary["k"] = args.k
    summary["max_tokens"] = args.max_tokens
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out} -- feed it to {scripts}", file=sys.stderr)


def _demo():
    """Self-check: the pieces with a branch or a parser in them."""
    assert parse_timestamp("2023/05/20 (Sat) 02:33") == datetime(2023, 5, 20, 2, 33, tzinfo=timezone.utc)
    assert parse_timestamp("2023/05/20") == datetime(2023, 5, 20, 0, 0, tzinfo=timezone.utc)
    assert parse_timestamp("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56, tzinfo=timezone.utc)
    assert parse_timestamp("12:30 am on 1 January 2024") == datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc)
    for bad in ("", "yesterday", "2023-05-20T00:00:00Z"):
        try:
            parse_timestamp(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} should not have parsed")

    store = Store()
    try:
        store.ingest([
            Turn("the deploy script lives in ops/deploy.sh", parse_timestamp("2023/05/20 (Sat) 02:33"), "s1"),
            Turn("the staging database is postgres 16", parse_timestamp("2023/05/21 (Sun) 09:00"), "s1"),
        ])
        context, cost = store.retrieve("deploy script")
        assert any("deploy.sh" in c for c in context), context
        assert cost.injected >= 1 and cost.tokens > 0
        assert cost.considered >= cost.injected
        assert cost.cost_usd("claude-opus-5") > 0
    finally:
        store.close()

    summary = summarize([cost])
    assert summary["retrieval_calls"] == 1
    assert summary["input_cost_per_turn_usd"] > 0
    assert summary["embedding_model"].startswith("none")
    print("ok  memvault_bench self-check")


if __name__ == "__main__":
    _demo()
