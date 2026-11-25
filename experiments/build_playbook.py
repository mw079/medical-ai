# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.60", "python-dotenv==1.1.0"]
# ///
"""Distill the 60 development lesson cards into one short, general playbook.

A retrieved card helps only about 1 new patient in 6, while the recurring missteps apply to
every case. One GPT-6 Luna call reads every card's missteps and lesson (no diagnoses, no
presentations) and writes at most 10 general rules. Code rejects a playbook that names any
development diagnosis. Output: evaluation/harnesses/memory/playbook.json, which the reported
results used; rebuilding it needs --force.

    uv run experiments/build_playbook.py --force
"""

import argparse
from datetime import date
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from models import api_key  # noqa: E402

CARDS = ROOT / "evaluation/harnesses/memory/dev_lessons.json"
OUT = ROOT / "evaluation/harnesses/memory/playbook.json"
MODEL = "gpt-6-luna"
CURATOR = """These are mistakes and lessons from a diagnostic agent's attempts on 60 different patients.
The agent asks questions, orders tests under a budget, then commits to one diagnosis.
Write a playbook of at most 10 short, general rules (each at most 200 characters) that would have
prevented the most frequent and costly mistakes. Rules must apply across diseases: do not name any
specific disease, syndrome, organism or gene. Order them by how often the mistake occurred.
Return {"rules": ["...", "..."]}."""


def names_a_diagnosis(rule, diagnoses):
    text = rule.casefold()
    return [d for d in diagnoses if len(d) > 6 and d.casefold() in text]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="overwrite the existing playbook (the harness results used the committed one)")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        parser.error(f"{OUT.relative_to(ROOT)} exists; pass --force to rebuild it")
    from openai import OpenAI

    cards = json.loads(CARDS.read_text())["cards"]
    lessons = [{"missteps": c["missteps"], "lesson": c["strategy"]} for c in cards]
    diagnoses = sorted({c["diagnosis"] for c in cards} | {
        part.strip() for c in cards for part in re.split(r"[,;()]| with | and ", c["diagnosis"]) if part.strip()})
    key = api_key()
    client = OpenAI(api_key=key, timeout=180)
    for _ in range(3):
        response = client.responses.create(
            model=MODEL, reasoning={"effort": "medium"}, max_output_tokens=4000, store=False,
            input=[{"role": "developer", "content": CURATOR},
                   {"role": "user", "content": json.dumps(lessons, ensure_ascii=False)}])
        try:
            rules = json.loads(response.output_text)["rules"]
        except (ValueError, KeyError, TypeError):
            continue
        if (isinstance(rules, list) and 1 <= len(rules) <= 10
                and all(isinstance(r, str) and 0 < len(r) <= 250 for r in rules)
                and not any(names_a_diagnosis(r, diagnoses) for r in rules)):
            break
    else:
        raise SystemExit("No valid playbook after three attempts")
    OUT.write_text(json.dumps({"built": date.today().isoformat(), "model": MODEL, "curator_prompt": CURATOR,
                               "source": "harnesses/memory/dev_lessons.json", "rules": rules},
                              indent=2, ensure_ascii=False) + "\n")
    print("\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))


if __name__ == "__main__":
    main()
