"""Shared plumbing for the LongMemEval and LOCOMO harnesses.

Both benchmarks have the same shape: a haystack of conversation turns, a
question, and a gold answer. MemVault's job is only the middle step -- given
the haystack and the question, decide what goes in the model's context. So
this module ingests turns, retrieves for a question, and reports what the
retrieval cost. Generation and grading stay with each benchmark's own
scripts, which is what "run unmodified" means (plan tasks P2-3/P2-4).

Requires the `memvault` wheel:

    maturin build -m crates/memvault-ffi/Cargo.toml --release
    pip install --find-links target/wheels memvault
"""

import re
import shutil
import tempfile
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

    data_dir: str = field(default_factory=lambda: tempfile.mkdtemp(prefix="memvault-bench-"))
    _mv: memvault.MemVault = field(init=False)
    _text_by_fact: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._mv = memvault.MemVault(self.data_dir)

    def ingest(self, turns):
        for turn in turns:
            fact_id = self._mv.write(
                NAMESPACE,
                turn.text,
                valid_from=turn.when.isoformat(),
                source=turn.session_id or None,
            )
            self._text_by_fact[fact_id] = turn.text

    def retrieve(self, question, k=10, max_tokens=2048):
        """Return (context_strings, RetrievalCost) for one question."""
        _, explanations = self._mv.search(NAMESPACE, question, k=k, max_tokens=max_tokens)

        cost = RetrievalCost(considered=len(explanations))
        context = []
        for e in explanations:
            if e.outcome == "Injected":
                cost.injected += 1
                cost.tokens += e.token_cost
                context.append(self._text_by_fact[e.fact_id])
            elif e.outcome == "CutByBudget":
                cost.cut_by_budget += 1
            elif e.outcome == "CutByK":
                cost.cut_by_k += 1
            elif e.outcome == "FilteredByTime":
                cost.filtered_by_time += 1
        return context, cost

    def close(self):
        del self._mv
        shutil.rmtree(self.data_dir, ignore_errors=True)


# Each benchmark stamps its dates differently and neither is ISO. Parsing is
# not optional: valid_from drives decay and every temporal-reasoning
# question, so a silently-wrong date quietly changes the score.
_LONGMEMEVAL_DATE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})(?:\s+\([A-Za-z]{3}\))?(?:\s+(\d{2}):(\d{2}))?$")
_LOCOMO_DATE = re.compile(
    r"^(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})$", re.IGNORECASE
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    )
}


def parse_timestamp(raw):
    """Parse either benchmark's timestamp format into an aware UTC datetime.

    Raises on anything unrecognized rather than defaulting to now(): a
    fabricated date would silently corrupt every temporal question in the run.
    """
    raw = raw.strip()

    m = _LONGMEMEVAL_DATE.match(raw)
    if m:
        year, month, day, hour, minute = m.groups()
        return datetime(int(year), int(month), int(day), int(hour or 0), int(minute or 0), tzinfo=timezone.utc)

    m = _LOCOMO_DATE.match(raw)
    if m:
        hour, minute, meridiem, day, month_name, year = m.groups()
        hour = int(hour) % 12 + (12 if meridiem.lower() == "pm" else 0)
        month = _MONTHS.get(month_name.lower())
        if month is None:
            raise ValueError(f"unknown month in timestamp: {raw!r}")
        return datetime(int(year), month, int(day), hour, int(minute), tzinfo=timezone.utc)

    raise ValueError(f"unrecognized timestamp format: {raw!r}")


def summarize(costs, model="claude-opus-5"):
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
        "token_cost_basis": "memvault Explanation.token_cost (ciphertext bytes / 4), not a real tokenizer",
    }


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
    print("ok  memvault_bench self-check")


if __name__ == "__main__":
    _demo()
