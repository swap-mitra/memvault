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


ROW = re.compile(r"<tr class=\"(?:kept|cut)\">(.*?)</tr>", re.S)
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def hero_rows_match_readme(html: str) -> list:
    """The hero table quotes a real run from the README. If the README's
    numbers move, the page is quoting something that never happened."""
    readme = (PAGE.parent.parent / "README.md").read_text(encoding="utf-8")
    rid = html.split("retrieval_id: <b>", 1)[1][:8]
    tail = readme.split(f"retrieval_id: {rid}", 1)[-1]
    block = []
    for line in tail.splitlines():
        if line.startswith("```"):
            break
        if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-", line):
            block.append(line.split())
    page = [
        [re.sub(r"<[^>]+>", "", c).strip() for c in CELL.findall(row)]
        for row in ROW.findall(html)
    ]
    if page != block:
        return [f"hero table does not match README: {page} != {block}"]
    return []


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

    errors += hero_rows_match_readme(html)

    # An inline icon means the tab is drawn without a request. Swap it for a
    # file and every page load fetches one, or 404s on /favicon.ico.
    if 'rel="icon"' not in html or 'href="data:image/svg+xml,' not in html:
        errors.append("favicon is missing or is not an inline data: URI")

    for line in errors:
        print(f"FAIL {line}")
    print("chain verified from seq 0" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
