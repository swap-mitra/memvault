"""LongMemEval harness (plan task P2-3).

Runs LongMemEval's haystack through MemVault and writes the retrieved
context per question, in the shape LongMemEval's own generation and
evaluation scripts consume. Those scripts are not reimplemented here --
"run unmodified" (product doc §7) means the benchmark grades itself and
MemVault only supplies the memory layer.

    python benchmarks/longmemeval.py longmemeval_s.json --out retrievals.jsonl

The output carries, per question, the retrieved context plus the retrieval
cost figures §7 requires alongside any accuracy number. Feed it to
LongMemEval's generation script, then its evaluator, and publish the score
next to the cost summary this prints.
"""

import argparse
import json
import sys

from memvault_bench import Store, Turn, parse_timestamp, summarize

REQUIRED_FIELDS = ("question_id", "question", "haystack_sessions", "haystack_dates")


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path}: expected a non-empty JSON list of questions")

    missing = [k for k in REQUIRED_FIELDS if k not in data[0]]
    if missing:
        raise SystemExit(
            f"{path}: not a LongMemEval file -- first item is missing {missing}. "
            f"Expected the released longmemeval_s.json / longmemeval_m.json shape."
        )
    return data


def turns_for(item):
    """Flatten a question's haystack into dated turns.

    One fact per utterance rather than per session: the evidence for a
    question is usually a single turn, and session-sized facts would drag
    its whole session into the budget with it.
    """
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]
    session_ids = item.get("haystack_session_ids") or [f"session_{i}" for i in range(len(sessions))]
    if not (len(sessions) == len(dates) == len(session_ids)):
        raise SystemExit(
            f"{item['question_id']}: haystack_sessions/dates/session_ids differ in length "
            f"({len(sessions)}/{len(dates)}/{len(session_ids)})"
        )

    for session, raw_date, session_id in zip(sessions, dates, session_ids):
        when = parse_timestamp(raw_date)
        for turn in session:
            text = (turn.get("content") or "").strip()
            if not text:
                continue
            yield Turn(f"[{turn.get('role', 'user')}] {text}", when, session_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="Path to longmemeval_s.json (or _m / _oracle)")
    ap.add_argument("--out", default="longmemeval_retrievals.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, help="Only run the first N questions (smoke test)")
    ap.add_argument("--model", default="claude-opus-5", help="Model whose input price prices the context")
    args = ap.parse_args()

    items = load(args.dataset)
    if args.limit:
        items = items[: args.limit]

    costs = []
    with open(args.out, "w", encoding="utf-8") as out:
        for n, item in enumerate(items, 1):
            store = Store()
            try:
                store.ingest(turns_for(item))
                context, cost = store.retrieve(item["question"], k=args.k, max_tokens=args.max_tokens)
            finally:
                store.close()
            costs.append(cost)

            out.write(
                json.dumps(
                    {
                        "question_id": item["question_id"],
                        "question": item["question"],
                        "question_type": item.get("question_type"),
                        "answer": item.get("answer"),
                        "retrieved_context": context,
                        "retrieval_cost": vars(cost),
                    }
                )
                + "\n"
            )
            print(f"\r{n}/{len(items)} questions", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    summary = summarize(costs, model=args.model)
    summary["dataset"] = args.dataset
    summary["questions"] = len(items)
    summary["k"] = args.k
    summary["max_tokens"] = args.max_tokens
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out} -- feed it to LongMemEval's generation + evaluation scripts", file=sys.stderr)


if __name__ == "__main__":
    main()
