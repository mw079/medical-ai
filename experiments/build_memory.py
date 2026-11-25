# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.60", "python-dotenv==1.1.0"]
# ///
"""Build the experience memory: one lesson card per development case, written by GPT-6 Luna.

For each development case, a reflector call sees the opening, the documented findings, the
reference diagnosis and GPT-6 Luna's graded trajectories on that case, and writes a lesson card.
Only development cases are read. Cards are cached per case in data/memory/cards/ and collected
into evaluation/harnesses/memory/dev_lessons.json, which the reported results used; rebuilding it
needs --force.

    uv run experiments/build_memory.py --force
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from models import api_key  # noqa: E402

RELEASE = ROOT / "benchmark/development.json"
RUNS = ROOT / "data/runs/reviewed-cpt-bedside-v4/development"
CACHE = ROOT / "data/memory/cards"
OUT = ROOT / "evaluation/harnesses/memory/dev_lessons.json"
MODEL = "gpt-6-luna"
EFFORT = "medium"
# The prompt asks for shorter text; validation allows 25% slack so a slight overrun is not rejected.
LIMITS = {"presentation": 375, "strategy": 500, "item": 250, "decisive_evidence": 5, "missteps": 4}
REFLECTOR = """You review how a diagnostic agent handled one training case, to write a lesson card
that helps the agent on future, different patients. The agent saw only the opening and the replies
to its own questions and tests; the documented findings below are everything the case contains.
Use only the supplied data. Return one JSON object:
{"presentation": "...", "decisive_evidence": ["..."], "missteps": ["..."], "strategy": "..."}

- presentation: the opening pattern in general clinical terms (age group, key symptoms, timing),
  at most 300 characters, without naming the diagnosis.
- decisive_evidence: up to 5 findings or tests that established the diagnosis, each naming the
  question or test to request and what it showed.
- missteps: up to 4 concrete mistakes in the failing trajectories, such as diagnosing before a
  decisive test, ordering tests the case did not contain, or answering too broadly. Use [] if
  every trajectory was correct.
- strategy: one general lesson for patients with a similar presentation, at most 400 characters.
Each list item is at most 200 characters."""


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def development():
    """The packed development split, read once."""
    return read(RELEASE)


def trajectories(case_id):
    out = []
    for path in sorted(RUNS.glob(f"budget_*/baseline/run*/budget_*/{case_id}.json")):
        row = read(path)
        if "result" not in row:
            continue
        steps = [m["content"] for m in row["result"]["agent_messages"][2:] if m["role"] in ("assistant", "user")]
        out.append({"mode": path.parts[-5], "score": row["result"]["diagnosis_score"],
                    "correct": row["result"]["correct"], "final_diagnosis": row["result"]["model_diagnosis"],
                    "medical_cost_usd": row["medical_cost"]["total_usd"], "steps": steps})
    return out


def validate(card):
    if set(card) != {"presentation", "decisive_evidence", "missteps", "strategy"}:
        raise ValueError(f"Unexpected fields {sorted(card)}")
    for field in ("presentation", "strategy"):
        if not isinstance(card[field], str) or not card[field].strip() or len(card[field]) > LIMITS[field]:
            raise ValueError(f"Invalid {field}")
    for field in ("decisive_evidence", "missteps"):
        items = card[field]
        if not isinstance(items, list) or len(items) > LIMITS[field] or any(
                not isinstance(i, str) or not i.strip() or len(i) > LIMITS["item"] for i in items):
            raise ValueError(f"Invalid {field}")
    if not card["decisive_evidence"]:
        raise ValueError("decisive_evidence is empty")
    return card


def reflect(client, case):
    payload = {"opening": case["case_summary"], "reference_diagnosis": case["final_diagnosis"],
               "documented_findings": case["findings"], "trajectories": case["trajectories"]}
    digest = hashlib.sha256(json.dumps([MODEL, EFFORT, REFLECTOR, payload], sort_keys=True).encode()).hexdigest()
    path = CACHE / f"{case['case_id']}.json"
    if path.exists() and read(path)["input_sha256"] == digest:
        return read(path)
    for _ in range(3):
        response = client.responses.create(
            model=MODEL, reasoning={"effort": EFFORT}, max_output_tokens=4000, store=False,
            input=[{"role": "developer", "content": REFLECTOR},
                   {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        try:
            card = validate(json.loads(response.output_text or ""))
            break
        except ValueError:
            continue
    else:
        raise RuntimeError(f"No valid lesson card for {case['case_id']}")
    record = {"case_id": case["case_id"], "input_sha256": digest, **card, "diagnosis": case["final_diagnosis"],
              "trajectories": len(case["trajectories"]),
              "trajectories_correct": sum(t["correct"] for t in case["trajectories"])}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true",
                        help="overwrite the existing cards (the harness results used the committed ones)")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        parser.error(f"{OUT.relative_to(ROOT)} exists; pass --force to rebuild it")
    cases = development()["cases"]
    for case in cases:
        case["findings"] = [{"name": f["name"], "result": f["result"]}
                            for f in development()["audit"][case["case_id"]]["draft"]["findings"]]
        case["trajectories"] = trajectories(case["case_id"])
        if len(case["trajectories"]) != 4:
            raise SystemExit(f"{case['case_id']} has {len(case['trajectories'])} trajectories; it needs the "
                             "2 baseline runs at $1,000 and $4,000 on development in data/runs")
    from openai import OpenAI
    key = api_key()
    client = OpenAI(api_key=key, timeout=180)
    with ThreadPoolExecutor(args.workers) as pool:
        cards = list(pool.map(lambda case: reflect(client, case), cases))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "built": date.today().isoformat(), "split": "development", "model": MODEL, "reasoning": EFFORT,
        "reflector_prompt": REFLECTOR, "source": "experiments/specs/memory_dev_traces.json",
        "cards": [{k: v for k, v in card.items() if k != "input_sha256"} for card in cards],
    }, indent=2, ensure_ascii=False) + "\n")
    solved = sum(card["trajectories_correct"] > 0 for card in cards)
    print(f"{len(cards)} cards written to {OUT.relative_to(ROOT)}; {solved} cases solved in at least one trajectory")


if __name__ == "__main__":
    main()
