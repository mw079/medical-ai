"""Run an experiment spec: every arm, mode, split and repeat, then the paired report.

    python3 experiments/run_arms.py experiments/specs/<name>.json [--analyze-only] [--dry-run]

A spec names one policy, the splits with their case counts, the modes, the arms and the number
of runs. An arm is null (the baseline), a playbook file, or {"model", "agent", "agents", "harness"}. Raw runs go to
data/runs/<policy>/<split>/<mode>/<arm>/run<k>/. The report goes to results/<name>/.
Arms alternate within each repeat, so drift over time affects every arm alike. Re-running the
same command resumes. Test splits require "held_out": true in the spec.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from compare_repeated import arm_summary, compare, load_arm, paired  # noqa: E402
from agents import AGENTS, SINGLE_DESIGNS  # noqa: E402
from models import DEFAULT_MODEL, registry  # noqa: E402
from policy import get_policy  # noqa: E402

SPLITS = ("development", "validation", "same_source_test", "external_test")
HELD_OUT = {"same_source_test", "external_test"}
MODES = ("instant", "budget_1000", "budget_4000", "no_budget")
HARNESS_MODES = {"budget_1000", "budget_4000"}
ATTEMPTS = 3


def read(path):
    return json.loads(Path(path).read_text())


def load_spec(path):
    spec = read(path)
    for key in ("name", "policy", "splits", "modes", "arms", "runs"):
        if key not in spec:
            raise ValueError(f"Spec is missing {key!r}")
    get_policy(spec["policy"])
    if not spec["arms"] or "baseline" not in spec["arms"]:
        raise ValueError("Every spec needs a 'baseline' arm to compare against")
    for split, limit in spec["splits"].items():
        if split not in SPLITS or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"Invalid split entry {split!r}: {limit!r}")
    if any(mode not in MODES for mode in spec["modes"]):
        raise ValueError(f"Unknown mode in {spec['modes']}")
    if (set(spec["splits"]) & HELD_OUT) and spec.get("held_out") is not True:
        raise ValueError('Test splits need "held_out": true in the spec')
    if not isinstance(spec["runs"], int) or spec["runs"] < 1:
        raise ValueError("runs must be a positive integer")
    models = registry()["models"]
    for arm in spec["arms"]:
        config = arm_config(spec, arm)
        if config["model"] not in models:
            raise ValueError(f"Arm {arm!r} uses unknown model {config['model']!r}")
        if config["agent"] not in AGENTS or (config["agent"] in SINGLE_DESIGNS) != (config["agents"] == 1):
            raise ValueError(f"Arm {arm!r} needs a known design, with 1 agent for single designs and 2+ for teams")
        if config["harness"] is not None:
            if config["agent"] != "single":
                raise ValueError(f"Arm {arm!r}: playbooks apply to the single agent only")
            if not (ROOT / config["harness"]).is_file():
                raise ValueError(f"Missing harness for arm {arm!r}: {config['harness']}")
            if set(spec["modes"]) - HARNESS_MODES:
                raise ValueError(f"Harness arms run only in {sorted(HARNESS_MODES)}")
    return spec


def arm_config(spec, arm):
    """An arm is null (baseline), a harness path, or {"model", "harness", "agent", "agents"}."""
    value = spec["arms"][arm]
    if value is None or isinstance(value, str):
        return {"model": DEFAULT_MODEL, "harness": value, "agent": "single", "agents": 1}
    if not isinstance(value, dict) or not value.keys() <= {"model", "harness", "agent", "agents"}:
        raise ValueError(f"Invalid arm {arm!r}: {value!r}")
    return {"model": value.get("model", DEFAULT_MODEL), "harness": value.get("harness"),
            "agent": value.get("agent", "single"), "agents": value.get("agents", 1)}


def run_dir(spec, split, mode, arm, k):
    return ROOT / "data/runs" / spec["policy"] / split / mode / arm / f"run{k}"


def complete(directory, mode, limit):
    try:
        summary = read(directory / "summary.json")["modes"][mode]
    except (OSError, KeyError, ValueError):
        return False
    return summary["scored"] == limit and not summary["errors"]


def command(spec, split, mode, arm, k, workers):
    cmd = ["uv", "run", "--locked", "evaluation/evaluate.py", "--split", split,
           "--limit", str(spec["splits"][split]), "--mode", mode, "--policy", spec["policy"],
           "--output", str(run_dir(spec, split, mode, arm, k)), "--workers", str(workers)]
    config = arm_config(spec, arm)
    cmd += ["--model", config["model"]]
    if config["agent"] != "single":
        cmd += ["--agent", config["agent"], "--agents", str(config["agents"])]
    if config["harness"]:
        cmd += ["--harness", config["harness"]]
    if split in HELD_OUT:
        cmd.append("--held-out")
    return cmd


def run_split(spec, split, workers, dry_run):
    limit = spec["splits"][split]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for k in range(1, spec["runs"] + 1):
        for mode in spec["modes"]:
            for arm in spec["arms"]:
                out = run_dir(spec, split, mode, arm, k)
                if complete(out, mode, limit):
                    continue
                cmd = command(spec, split, mode, arm, k, workers)
                if dry_run:
                    print(" ".join(cmd))
                    continue
                out.mkdir(parents=True, exist_ok=True)
                for attempt in range(1, ATTEMPTS + 1):
                    print(f"{split} {mode} {arm} run{k} attempt {attempt}", flush=True)
                    with (out / "run.log").open("a") as log:
                        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
                    if complete(out, mode, limit):
                        break
                else:
                    raise RuntimeError(f"{out} did not finish after {ATTEMPTS} attempts; see run.log")


def arm_record(spec, arm):
    config = arm_config(spec, arm)
    harness = config["harness"]
    return {"model": config["model"], "agent": config["agent"], "agents": config["agents"],
            "harness": None if harness is None else {
                "path": harness, "sha256": hashlib.sha256((ROOT / harness).read_bytes()).hexdigest()}}


CASE_FIELDS = ("mode", "split", "arm", "run", "case_id", "correct", "judge_score", "diagnosis", "medical_cost_usd",
               "agent_calls", "agent_input_tokens", "agent_output_tokens", "agent_cost_usd")


def case_rows(spec):
    """One row per mode, split, arm, run and case: enough to recompute every number in the report."""
    for mode in spec["modes"]:
        for split in spec["splits"]:
            for arm in spec["arms"]:
                ids, rows, _, _ = load_arm(run_dir(spec, split, mode, arm, 1).parent, mode)
                for case_id in ids:
                    for k, r in enumerate(rows[case_id], 1):
                        yield {"mode": mode, "split": split, "arm": arm, "run": k, "case_id": case_id,
                               "correct": int(r["correct"]), "judge_score": r["score"], "diagnosis": r["diagnosis"],
                               "medical_cost_usd": r["cost"], "agent_calls": r["agent"]["calls"],
                               "agent_input_tokens": r["agent"]["input_tokens"],
                               "agent_output_tokens": r["agent"]["output_tokens"],
                               "agent_cost_usd": round(r["agent_cost"], 6)}


def write_cases(spec, out):
    with (out / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(case_rows(spec))


def analyze(spec):
    """Per split and mode: every arm's summary and its paired comparison with the baseline."""
    report = {"name": spec["name"], "policy": spec["policy"], "runs": spec["runs"],
              "arms": {arm: arm_record(spec, arm) for arm in spec["arms"]},
              "results": {}}
    for mode in spec["modes"]:
        pooled = {arm: [] for arm in spec["arms"] if arm != "baseline"}
        for split in spec["splits"]:
            base_dir = run_dir(spec, split, mode, "baseline", 1).parent
            entry = {}
            ids, rows, _, n = load_arm(base_dir, mode)
            entry["baseline"] = arm_summary(ids, rows, n)
            for arm in pooled:
                result = compare(base_dir, run_dir(spec, split, mode, arm, 1).parent, mode)
                entry[arm] = {**result["candidate"], **{k: result[k] for k in (
                    "delta_points", "p_value", "ci95_points", "cases_better", "cases_worse")}}
                pooled[arm] += [v["candidate"] - v["baseline"] for v in result["per_case"].values()]
            report["results"].setdefault(mode, {})[split] = entry
        if len(spec["splits"]) > 1:
            report["results"][mode]["pooled"] = {arm: paired(d) for arm, d in pooled.items() if d}
    out = ROOT / "results" / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "report.md").write_text(markdown(report))
    write_cases(spec, out)
    return report


def markdown(report):
    lines = [f"# {report['name']}", "", f"Policy `{report['policy']}`. Runs per arm are in the Runs column.", "",
             "| Mode | Split | Arm | Runs | Mean accuracy | Range over runs | Avg medical cost | "
             "Agent calls | Agent tokens (in / out) | Agent model cost | Δ vs baseline | p |",
             "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, splits in report["results"].items():
        for split, arms in splits.items():
            if split == "pooled":
                continue
            for arm, r in arms.items():
                delta = f"{r['delta_points']:+.1f}" if "delta_points" in r else "–"
                p = f"{r['p_value']:.2f}" if "p_value" in r else "–"
                lines.append(f"| {mode} | {split} | {arm} | {r['runs']} | {100 * r['mean_accuracy']:.1f}% | "
                             f"{100 * r['min_accuracy']:.0f}–{100 * r['max_accuracy']:.0f}% | "
                             f"${r['avg_medical_cost_usd']:,.0f} | {r['avg_agent_calls']:.1f} | "
                             f"{r['avg_agent_input_tokens']:,.0f} / {r['avg_agent_output_tokens']:,.0f} | "
                             f"${r['avg_agent_cost_usd']:.4f} | {delta} | {p} |")
        for arm, r in splits.get("pooled", {}).items():
            low, high = r["ci95_points"]
            lines.append(f"| {mode} | pooled | {arm} | | | | | | | | {r['delta_points']:+.1f} "
                         f"(95% CI {low:+.1f} to {high:+.1f}) | {r['p_value']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--workers", type=int, default=12, help="parallel cases per run")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the pending commands")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    if not args.analyze_only:
        # Splits run side by side. Within a split, runs are sequential and arms alternate.
        with ThreadPoolExecutor(max_workers=len(spec["splits"])) as pool:
            for job in [pool.submit(run_split, spec, s, args.workers, args.dry_run) for s in spec["splits"]]:
                job.result()
        if args.dry_run:
            return 0
    report = analyze(spec)
    print((ROOT / "results" / spec["name"] / "report.md").read_text())
    return 0 if report else 1


if __name__ == "__main__":
    raise SystemExit(main())
