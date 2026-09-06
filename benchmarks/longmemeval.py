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

import json

from memvault_bench import Turn, parse_args, parse_timestamp, report, run

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
    args = parse_args(__doc__, "Path to longmemeval_s.json (or _m / _oracle)", "longmemeval_retrievals.jsonl")
    items = load(args.dataset)
    if args.limit:
        items = items[: args.limit]

    def unit(item):
        row = {
            "question_id": item["question_id"],
            "question_type": item.get("question_type"),
            "answer": item.get("answer"),
        }
        return turns_for(item), [(item["question"], row)]

    costs, questions = run(items, unit, args, lambda n, total, _q: f"{n}/{total} questions")
    report(costs, args, {"dataset": args.dataset, "questions": questions}, "LongMemEval's generation + evaluation scripts")


if __name__ == "__main__":
    main()
