"""Compare two arms from repeated runs on the same cases.

Each arm is a folder of run directories (run1..runN). Per case, the score is the share of runs
that were correct. The paired test flips arm labels within each case, so it accounts for both
case difficulty and run-to-run variation. Each run also reports the diagnostic agent's model
calls, tokens and token cost, counted from the saved traces. experiments/run_arms.py uses
these functions for every report.
"""

import json
from pathlib import Path
import random
from statistics import mean

from models import token_cost

USAGE_FIELDS = ("calls", "input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens", "reasoning_tokens")


def agent_usage(model_calls):
    """Sum the diagnostic agent's model calls and tokens in one encounter's trace.

    Retries ("previous_attempts") and team members ("agent_calls") are included. Gatekeeper,
    judge and pricing calls are not: they belong to the environment and are the same for every arm.
    """
    totals = dict.fromkeys(USAGE_FIELDS, 0)

    def add(call):
        # A team's turn record holds no usage itself; its model calls are nested under agent_calls.
        if "usage" in call:
            count(call)
        for nested in (*call.get("previous_attempts", []), *call.get("agent_calls", [])):
            add(nested)

    def count(call):
        usage = call.get("usage") or {}
        inputs = usage.get("input_tokens_details") or {}
        outputs = usage.get("output_tokens_details") or {}
        totals["calls"] += 1
        totals["input_tokens"] += usage.get("input_tokens") or 0
        totals["cached_tokens"] += inputs.get("cached_tokens") or 0
        totals["cache_write_tokens"] += inputs.get("cache_write_tokens") or 0
        totals["output_tokens"] += usage.get("output_tokens") or 0
        totals["reasoning_tokens"] += outputs.get("reasoning_tokens") or 0

    for call in model_calls:
        if call.get("role") == "diagnostic":
            add(call)
    return totals


DEFAULT_MODE = "budget_1000"
IGNORED_SETTINGS = ("playbook", "modes", "model", "reasoning", "agent", "agents", "memory_sha256")  # what may differ between arms


def read(path):
    return json.loads(Path(path).read_text())


def load_arm(arm, mode=DEFAULT_MODE):
    """Return case IDs, per-case run outcomes, comparable settings, and the run count."""
    runs = sorted(p for p in Path(arm).glob("run[0-9]*") if p.is_dir())
    if not runs:
        raise ValueError(f"No run directories in {arm}")
    settings = [read(r / "settings.json") for r in runs]
    ids = settings[0]["case_ids"]
    rows = {}
    model = settings[0]["model"]
    for run, s in zip(runs, settings):
        summary = read(run / "summary.json")["modes"][mode]
        if s["case_ids"] != ids or summary["errors"] or summary["scored"] != len(ids):
            raise ValueError(f"{run} has different cases, errors, or missing cases")
        for i in ids:
            row = read(run / mode / f"{i}.json")
            usage = agent_usage(row["model_calls"])
            rows.setdefault(i, []).append({"correct": bool(row["result"]["correct"]),
                                           "score": row["result"].get("diagnosis_score"),
                                           "diagnosis": row["result"].get("model_diagnosis"),
                                           "cost": row["medical_cost"]["total_usd"],
                                           "agent": usage, "agent_cost": token_cost(usage, model)})
    # Runs of one arm must match exactly (apart from the mode list). Arms may differ in agent.
    if any({k: v for k, v in s.items() if k != "modes"} != {k: v for k, v in settings[0].items() if k != "modes"}
           for s in settings):
        raise ValueError(f"Runs in {arm} have different settings")
    comparable = {k: v for k, v in settings[0].items() if k not in IGNORED_SETTINGS}
    return ids, rows, comparable, len(runs)


def arm_summary(ids, rows, n_runs):
    per_run = [sum(rows[i][k]["correct"] for i in ids) for k in range(n_runs)]
    return {"runs": n_runs, "correct_per_run": per_run,
            "mean_accuracy": round(mean(mean(r["correct"] for r in rows[i]) for i in ids), 4),
            "min_accuracy": round(min(per_run) / len(ids), 4), "max_accuracy": round(max(per_run) / len(ids), 4),
            "avg_medical_cost_usd": round(mean(r["cost"] for i in ids for r in rows[i]), 2),
            **{f"avg_agent_{key}": round(mean(r["agent"][key] for i in ids for r in rows[i]), 1)
               for key in ("calls", "input_tokens", "output_tokens", "reasoning_tokens")},
            "avg_agent_cost_usd": round(mean(r["agent_cost"] for i in ids for r in rows[i]), 5)}


def paired(diffs, permutations=20000, seed=0):
    """Mean difference, two-sided sign-flip p-value, and a 95% bootstrap interval."""
    observed = mean(diffs)
    rng = random.Random(seed)
    extreme = sum(abs(mean(d if rng.random() < 0.5 else -d for d in diffs)) >= abs(observed) - 1e-12
                  for _ in range(permutations))
    boot = sorted(mean(rng.choice(diffs) for _ in diffs) for _ in range(5000))
    return {"delta_points": round(100 * observed, 2),
            "p_value": round((extreme + 1) / (permutations + 1), 4),
            "ci95_points": [round(100 * boot[124], 2), round(100 * boot[4875], 2)]}


def compare(baseline_dir, candidate_dir, mode=DEFAULT_MODE):
    ids, base, base_settings, nb = load_arm(baseline_dir, mode)
    cids, cand, cand_settings, nc = load_arm(candidate_dir, mode)
    if ids != cids or base_settings != cand_settings:
        raise ValueError("Arms differ in cases or evaluation settings")
    def share(rows):
        return {i: mean(r["correct"] for r in rows[i]) for i in ids}

    b, c = share(base), share(cand)
    diffs = [c[i] - b[i] for i in ids]
    return {"cases": len(ids), "mode": mode,
            "baseline": arm_summary(ids, base, nb), "candidate": arm_summary(ids, cand, nc),
            **paired(diffs),
            "cases_better": sorted(i for i in ids if c[i] > b[i]),
            "cases_worse": sorted(i for i in ids if c[i] < b[i]),
            "per_case": {i: {"baseline": b[i], "candidate": c[i]} for i in ids}}
