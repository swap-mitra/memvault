# Benchmark harnesses

LongMemEval and LOCOMO, run against MemVault's Python surface.

## What these do, and what they deliberately don't

Both benchmarks are Python, and both ship their own generation and grading
scripts. These harnesses replace only the memory layer: given a haystack of
conversation turns and a question, they decide what goes into the model's
context, and write that out in the shape each benchmark's own scripts
consume. Generation and judging stay with the benchmark, which is what
"run unmodified" means — reimplementing the grader here would make the
score ours rather than theirs.

That split is also why nothing here calls an LLM and why the project takes
no dependency on one. MemVault never calls a model (product doc P5); a
harness that did would be measuring something the engine doesn't do.

Embeddings are the exception, and deliberately so: retrieval quality is a
property of the embedding model as much as of the engine, and MemVault has
no model of its own. Without one the harness measures keyword (BM25)
retrieval only, which is a real configuration but not the one a quality
number should be quoted for. Pass `--embed-url` and `--embed-model` to use
any OpenAI-compatible `/embeddings` endpoint, the same way the server's
`MEMVAULT_EMBED_URL` does; the standard library makes the calls, so this
adds no dependency, and the summary names the model so a number never
travels without it.

## Setup

Install the `memvault` wheel, as described under Install in the main README.

Then get the datasets from their own projects: LongMemEval publishes
`longmemeval_s.json` / `_m` / `_oracle`, LOCOMO publishes `locomo10.json`.
Neither is redistributed here.

## Running

```sh
python benchmarks/longmemeval.py longmemeval_s.json --out lme_retrievals.jsonl \
  --embed-url http://localhost:11434/v1 --embed-model nomic-embed-text
python benchmarks/locomo.py       locomo10.json      --out locomo_retrievals.jsonl \
  --embed-url http://localhost:11434/v1 --embed-model nomic-embed-text
```

`--embed-api-key` (or `MEMVAULT_EMBED_API_KEY` in the environment) is the
bearer token for providers that want one. Leave both embed flags off for a
keyword-only run.

Each writes one JSON object per question — the retrieved context plus that
retrieval's cost — and prints a summary to stdout. Feed the JSONL to the
benchmark's generation script, then its evaluator, and publish the score
next to the summary.

`--limit N` runs the first N items, which is the way to check the pipeline
before committing to a full run.

## Reading the summary

Product doc §7 requires accuracy to be reported alongside what it cost, so
the summary carries both halves of that trade:

- `tokens_per_retrieval_call_mean` / `_max` — context volume per question.
- `injected_per_call_mean` vs `considered_per_call_mean` — how much of what
  was found actually fit.
- `cut_by_budget_total`, `cut_by_k_total`, `filtered_by_time_total` — why
  the rest didn't.
- `input_cost_per_turn_usd` — the retrieved context priced at the input
  rate of `--model` (default `claude-opus-5`).

Two caveats travel with those numbers, and both are stated in the output
itself rather than left to a footnote:

**`token_cost_basis`.** Token counts come from MemVault's own estimate
(ciphertext bytes / 4) unless the wheel was built with `--features
tokenizer`, in which case they are cl100k_base counts. The estimate is fine
for English prose and drifts on code or other languages.

**`embedding_model`.** Which model produced the vectors, or the statement
that none did. A quality score without this line is not comparable to
anything.

## Published numbers

None yet. The harnesses run in CI against miniature fixtures with keyword
retrieval only, which proves the pipeline and nothing about quality. A
publishable figure needs the real datasets, a named embedding model, the
benchmark's own generator and grader, and the cost summary alongside; when
that run happens its protocol and result belong here, together.

**Input cost only.** A memory layer decides what goes into the prompt and
nothing about what comes out, so input cost is the figure it is accountable
for. Total cost per turn needs an output-token measurement from whoever runs
the generation step.

## Testing without the datasets

```sh
python benchmarks/test_harnesses.py
```

Miniature fixtures in each dataset's shape, covering what breaks quietly:
dataset parsing, both timestamp formats, LOCOMO's numeric session ordering,
and the presence of every summary field §7 asks for.
