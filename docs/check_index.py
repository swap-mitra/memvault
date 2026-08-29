"""Checks docs/index.html holds together. Run: python docs/check_index.py

The page's structural device is an append-only chain, so a typo that breaks it
undercuts the one claim the page exists to make. This is that check.
"""

import pathlib
import re
import sys

PAGE = pathlib.Path(__file__).with_name("index.html")

RECORD = re.compile(
    r'record__seq">SEQ (\d+)</span>\s*'
    r'<span class="record__hash">hash <b>([0-9a-f]{8})</b></span>\s*'
    r'<span class="record__hash">prev <b>([0-9a-f]{8})</b></span>'
)


def main() -> int:
    html = PAGE.read_text(encoding="utf-8")
    records = RECORD.findall(html)
    errors = []

    if len(records) < 2:
        errors.append(f"found {len(records)} records; the regex is out of date")

    prev_hash = "00000000"
    for i, (seq, digest, prev) in enumerate(records):
        if int(seq) != i:
            errors.append(f"record {i}: seq is {seq}")
        if prev != prev_hash:
            errors.append(f"record {i}: prev {prev} != previous hash {prev_hash}")
        prev_hash = digest

    # Every colour goes through a token, so the palette can be changed in one
    # place. :root is the only place a literal is allowed to appear.
    body = html.split("--data:", 1)[-1]
    strays = re.findall(r"#[0-9a-fA-F]{3,8}\b(?!;?\s*/\*)", body)
    if strays:
        errors.append(f"hard-coded colours below :root: {strays}")

    for line in errors:
        print(f"FAIL {line}")
    print("chain verified from seq 0" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
