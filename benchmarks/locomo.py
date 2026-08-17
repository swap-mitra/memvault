"""LOCOMO harness (plan task P2-4).

Same shape as the LongMemEval harness: run LOCOMO's conversations through
MemVault, write the retrieved context per question, and let LOCOMO's own
scripts generate and grade. LOCOMO differs only in how the dataset is laid
out -- one long multi-session conversation per sample, with its questions
attached, rather than a per-question haystack.

    python benchmarks/locomo.py locomo10.json --out retrievals.jsonl
"""

import argparse
import json
import re
import sys

from memvault_bench import Store, Turn, parse_timestamp, summarize

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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="Path to locomo10.json")
    ap.add_argument("--out", default="locomo_retrievals.jsonl")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, help="Only run the first N samples (smoke test)")
    ap.add_argument("--model", default="claude-opus-5", help="Model whose input price prices the context")
    args = ap.parse_args()

    samples = load(args.dataset)
    if args.limit:
        samples = samples[: args.limit]

    costs = []
    questions = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for n, sample in enumerate(samples, 1):
            # One store per sample, not per question: every question in a
            # sample asks about the same conversation, so they share a
            # haystack and re-ingesting it per question would only be slower.
            store = Store()
            try:
                store.ingest(turns_for(sample["conversation"]))
                for qa in sample["qa"]:
                    question = qa.get("question")
                    if not question:
                        continue
                    context, cost = store.retrieve(question, k=args.k, max_tokens=args.max_tokens)
                    costs.append(cost)
                    questions += 1
                    out.write(
                        json.dumps(
                            {
                                "sample_id": sample.get("sample_id", f"sample_{n}"),
                                "question": question,
                                "category": qa.get("category"),
                                "answer": qa.get("answer"),
                                "evidence": qa.get("evidence"),
                                "retrieved_context": context,
                                "retrieval_cost": vars(cost),
                            }
                        )
                        + "\n"
                    )
            finally:
                store.close()
            print(f"\r{n}/{len(samples)} samples, {questions} questions", end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    summary = summarize(costs, model=args.model)
    summary["dataset"] = args.dataset
    summary["samples"] = len(samples)
    summary["questions"] = questions
    summary["k"] = args.k
    summary["max_tokens"] = args.max_tokens
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out} -- feed it to LOCOMO's generation + evaluation scripts", file=sys.stderr)


if __name__ == "__main__":
    main()
