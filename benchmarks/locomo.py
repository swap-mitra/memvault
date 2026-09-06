"""LOCOMO harness (plan task P2-4).

Same shape as the LongMemEval harness: run LOCOMO's conversations through
MemVault, write the retrieved context per question, and let LOCOMO's own
scripts generate and grade. LOCOMO differs only in how the dataset is laid
out -- one long multi-session conversation per sample, with its questions
attached, rather than a per-question haystack.

    python benchmarks/locomo.py locomo10.json --out retrievals.jsonl
"""

import json
import re

from memvault_bench import Turn, parse_args, parse_timestamp, report, run

_SESSION = re.compile(r"^session_(\d+)$")


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"{path}: expected a non-empty JSON list of samples")
    if "conversation" not in data[0] or "qa" not in data[0]:
        raise SystemExit(
            f"{path}: not a LOCOMO file -- first sample lacks 'conversation'/'qa'. "
            f"Expected the released locomo10.json shape."
        )
    return data


def turns_for(conversation):
    """Flatten `session_N` / `session_N_date_time` pairs into dated turns.

    Sessions are numbered keys rather than a list, so they are sorted
    numerically -- string sort would put session_10 before session_2 and
    scramble the conversation's chronology.
    """
    numbers = sorted(int(m.group(1)) for m in map(_SESSION.match, conversation) if m)
    if not numbers:
        raise SystemExit("conversation has no session_N keys")

    for n in numbers:
        raw_date = conversation.get(f"session_{n}_date_time")
        if not raw_date:
            raise SystemExit(f"session_{n} has no session_{n}_date_time")
        when = parse_timestamp(raw_date)
        for dialog in conversation[f"session_{n}"]:
            text = (dialog.get("text") or "").strip()
            if not text:
                continue
            speaker = dialog.get("speaker", "unknown")
            yield Turn(f"[{speaker}] {text}", when, f"session_{n}")


def main():
    args = parse_args(__doc__, "Path to locomo10.json", "locomo_retrievals.jsonl")
    samples = load(args.dataset)
    if args.limit:
        samples = samples[: args.limit]

    def unit(numbered):
        # One store per sample, not per question: every question in a sample
        # asks about the same conversation, so they share a haystack and
        # re-ingesting it per question would only be slower.
        n, sample = numbered
        asks = [
            (
                qa["question"],
                {
                    "sample_id": sample.get("sample_id", f"sample_{n}"),
                    "category": qa.get("category"),
                    "answer": qa.get("answer"),
                    "evidence": qa.get("evidence"),
                },
            )
            for qa in sample["qa"]
            if qa.get("question")
        ]
        return turns_for(sample["conversation"]), asks

    costs, questions = run(
        list(enumerate(samples, 1)), unit, args, lambda n, total, q: f"{n}/{total} samples, {q} questions"
    )
    report(
        costs,
        args,
        {"dataset": args.dataset, "samples": len(samples), "questions": questions},
        "LOCOMO's generation + evaluation scripts",
    )


if __name__ == "__main__":
    main()
