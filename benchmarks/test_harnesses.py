"""Runnable check for both benchmark harnesses, without the real datasets.

The datasets are large external downloads, so the fixtures here are
miniatures in the same shape. What this pins down is the part that breaks
silently: dataset parsing, date handling, session ordering, and whether the
cost summary carries the figures product doc §7 requires.

    python benchmarks/test_harnesses.py
"""

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

LONGMEMEVAL = [
    {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "Where does the deploy script live?",
        "answer": "ops/deploy.sh",
        "haystack_dates": ["2023/05/20 (Sat) 02:33", "2023/06/01 (Thu) 11:15"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Where is the deploy script?"},
                {"role": "assistant", "content": "The deploy script lives in ops/deploy.sh"},
            ],
            [
                {"role": "user", "content": "What database does staging run?"},
                {"role": "assistant", "content": "Staging runs postgres 16"},
            ],
        ],
    }
]

LOCOMO = [
    {
        "sample_id": "c1",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Alice", "text": "I finally moved the deploy script to ops/deploy.sh"},
                {"speaker": "Bob", "text": "Good, the old path always confused me"},
            ],
            # Deliberately out of lexicographic order: session_10 must sort
            # after session_2, not between session_1 and session_2.
            "session_10_date_time": "9:00 am on 2 September, 2023",
            "session_10": [{"speaker": "Alice", "text": "Staging is on postgres 16 now"}],
            "session_2_date_time": "10:30 am on 12 June, 2023",
            "session_2": [{"speaker": "Bob", "text": "I rewrote the billing job in rust"}],
        },
        "qa": [
            {"question": "Where is the deploy script?", "answer": "ops/deploy.sh", "category": 1},
            {"question": "What did Bob rewrite in rust?", "answer": "the billing job", "category": 1},
        ],
    }
]


def run_harness(script, dataset, tmp):
    dataset_path = os.path.join(tmp, "dataset.json")
    out_path = os.path.join(tmp, "out.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, script), dataset_path, "--out", out_path],
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stderr}"

    rows = [json.loads(line) for line in open(out_path, encoding="utf-8")]
    return rows, json.loads(proc.stdout)


def check_summary(summary):
    # Product doc §7: accuracy alone is not sufficient reporting.
    for field in (
        "retrieval_calls",
        "tokens_per_retrieval_call_mean",
        "input_cost_per_turn_usd",
        "input_price_model",
        "token_cost_basis",
    ):
        assert field in summary, f"summary is missing {field}: {summary}"
    assert summary["tokens_per_retrieval_call_mean"] > 0
    assert summary["input_cost_per_turn_usd"] > 0


def test_longmemeval():
    with tempfile.TemporaryDirectory() as tmp:
        rows, summary = run_harness("longmemeval.py", LONGMEMEVAL, tmp)

    assert len(rows) == 1
    row = rows[0]
    assert row["question_id"] == "q1"
    assert row["retrieved_context"], "nothing was retrieved"
    assert any("ops/deploy.sh" in c for c in row["retrieved_context"]), row["retrieved_context"]
    assert row["retrieval_cost"]["injected"] >= 1
    check_summary(summary)


def test_locomo():
    with tempfile.TemporaryDirectory() as tmp:
        rows, summary = run_harness("locomo.py", LOCOMO, tmp)

    assert len(rows) == 2, rows
    by_question = {r["question"]: r for r in rows}
    deploy = by_question["Where is the deploy script?"]
    assert any("ops/deploy.sh" in c for c in deploy["retrieved_context"]), deploy["retrieved_context"]

    billing = by_question["What did Bob rewrite in rust?"]
    assert any("billing job" in c for c in billing["retrieved_context"]), billing["retrieved_context"]

    assert summary["questions"] == 2
    check_summary(summary)


def test_a_wrong_dataset_fails_loudly():
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = os.path.join(tmp, "dataset.json")
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump([{"not": "a benchmark file"}], f)

        for script in ("longmemeval.py", "locomo.py"):
            proc = subprocess.run(
                [sys.executable, os.path.join(HERE, script), dataset_path],
                capture_output=True,
                text=True,
                cwd=HERE,
            )
            assert proc.returncode != 0, f"{script} accepted a file that isn't its dataset"
            assert "Expected the released" in proc.stderr, proc.stderr


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all benchmark harness tests passed")
