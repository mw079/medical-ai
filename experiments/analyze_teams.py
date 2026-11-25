# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sdbench[cpt-costing] @ git+https://github.com/microsoft/sdbench@304ce9625a046cea97d64e3b1ba5d19217b5095c",
#   "python-dotenv==1.1.0",
# ]
# ///
"""Why did the team designs not beat one agent? Separate generation from selection.

For every team encounter on the test splits, take the turn where the team diagnosed and collect
every diagnosis its members proposed (Independent and Decentralized proposals, Centralized role
recommendations). Each distinct proposal is graded with the encounter's own saved judge prompt,
only the candidate diagnosis swapped, by the same judge model (GPT-6 Luna, reasoning off).

- generation: did any member propose a correct diagnosis?
- selection: when one did, did the team's final answer use a correct one?

Grades are cached in data/analysis/proposal_grades.json, so reruns make no new calls.

    uv run experiments/analyze_teams.py
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from environment import explicit_judge_score  # noqa: E402

RUNS = ROOT / "data/runs/reviewed-cpt-bedside-v4"
CACHE = ROOT / "data/analysis/proposal_grades.json"
OUT = ROOT / "results/multi-agent-test/analysis.json"
JUDGE = "gpt-6-luna"
TEAM_ARMS = [f"{d}_{n}" for d in ("independent", "decentralized", "centralized") for n in (3, 5)]
DIAGNOSIS = re.compile(r"<diagnosis>(.*?)</diagnosis>", re.S)


def normalize(text):
    return " ".join(text.casefold().split()).rstrip(" .")


def proposals(turn):
    """Diagnoses the members proposed in this turn."""
    actions = [p["action"] for p in turn.get("proposals", [])] + [n["action"] for n in turn.get("notes", [])]
    return [m.group(1).strip() for a in actions if (m := DIAGNOSIS.search(a))]


def encounters():
    for mode in ("budget_1000", "budget_4000"):
        for arm in TEAM_ARMS:
            for path in sorted(RUNS.glob(f"*_test/{mode}/{arm}/run1/{mode}/PMC*.json")):
                row = json.loads(path.read_text())
                turns = [c for c in row["model_calls"] if c["role"] == "diagnostic"]
                final = turns[-1]
                yield {"mode": mode, "arm": arm, "case": path.stem, "split": path.parts[-6],
                       "chosen": row["result"]["model_diagnosis"], "correct": bool(row["result"]["correct"]),
                       "chosen_score": row["result"]["diagnosis_score"],
                       "proposed": proposals(final), "judge_messages": row["result"]["check_diagnosis_messages"][:2]}


def grade_all(items):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    jobs = {}
    for item in items:
        for diagnosis in item["proposed"]:
            key = f"{item['case']}|{normalize(diagnosis)}"
            if key not in cache and normalize(diagnosis) != normalize(item["chosen"]):
                jobs.setdefault(key, (item, diagnosis))
    if jobs:
        from openai import OpenAI
        from models import api_key
        key = api_key()
        client = OpenAI(api_key=key, timeout=90)

        def grade(job):
            cache_key, (item, diagnosis) = job
            system, user = item["judge_messages"]
            marker = f"Candidate Diagnosis: {item['chosen']}"
            if marker not in user["content"]:
                raise ValueError(f"Judge prompt for {item['case']} does not name the chosen diagnosis")
            messages = [system, {**user, "content": user["content"].replace(marker, f"Candidate Diagnosis: {diagnosis}")}]
            for _ in range(3):
                response = client.responses.create(model=JUDGE, input=messages, reasoning={"effort": "none"},
                                                   max_output_tokens=900, store=False)
                try:
                    return cache_key, explicit_judge_score(response.output_text or "")
                except ValueError:
                    continue
            raise RuntimeError(f"No parsable grade for {cache_key}")

        print(f"Grading {len(jobs)} new proposals", flush=True)
        with ThreadPoolExecutor(16) as pool:
            for cache_key, score in pool.map(grade, jobs.items()):
                cache[cache_key] = score
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return cache


def summarize(items, cache):
    out = {}
    for item in items:
        scores = [item["chosen_score"] if normalize(d) == normalize(item["chosen"])
                  else cache[f"{item['case']}|{normalize(d)}"] for d in item["proposed"]]
        any_correct = any(s >= 4 for s in scores) or item["correct"]
        s = out.setdefault(item["mode"], {}).setdefault(item["arm"], dict.fromkeys(
            ("encounters", "team_correct", "member_had_correct", "missed_by_selection",
             "no_member_correct", "proposed_diagnoses"), 0))
        s["encounters"] += 1
        s["team_correct"] += item["correct"]
        s["member_had_correct"] += any_correct
        s["missed_by_selection"] += any_correct and not item["correct"]
        s["no_member_correct"] += not any_correct
        s["proposed_diagnoses"] += len({normalize(d) for d in item["proposed"]})
    return out


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    items = list(encounters())
    cache = grade_all(items)
    summary = summarize(items, cache)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    for mode, arms in summary.items():
        print(f"\n{mode}")
        print(f"{'arm':16} {'team correct':>12} {'a member had it':>16} {'lost in selection':>18} {'nobody had it':>14}")
        for arm, s in arms.items():
            print(f"{arm:16} {s['team_correct']:>12} {s['member_had_correct']:>16} "
                  f"{s['missed_by_selection']:>18} {s['no_member_correct']:>14}")


if __name__ == "__main__":
    main()
