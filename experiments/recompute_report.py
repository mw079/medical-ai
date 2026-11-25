"""Recompute a report's accuracy, paired differences, p-values and intervals from its cases.csv.

    python3 experiments/recompute_report.py results/<name>   # or: results/*

This needs no raw run traces and no API keys. It checks every recomputed number against the
committed report.json and exits non-zero on a mismatch.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from compare_repeated import paired  # noqa: E402


def recompute(folder):
    """Yield (label, recomputed, reported) for each mode, split and arm, and for the pooled rows."""
    report = json.loads((folder / "report.json").read_text())
    share = defaultdict(lambda: defaultdict(list))  # (mode, split, arm) -> case -> outcomes
    with (folder / "cases.csv").open() as handle:
        for row in csv.DictReader(handle):
            share[row["mode"], row["split"], row["arm"]][row["case_id"]].append(int(row["correct"]))
    for mode, splits in report["results"].items():
        pooled = defaultdict(list)
        for split, arms in splits.items():
            if split == "pooled":
                continue
            base = share[mode, split, "baseline"]
            for arm, reported in arms.items():
                cases = share[mode, split, arm]
                yield f"{mode} {split} {arm} accuracy", round(mean(mean(v) for v in cases.values()), 4), reported["mean_accuracy"]
                if arm == "baseline":
                    continue
                diffs = [mean(cases[i]) - mean(base[i]) for i in base]
                pooled[arm] += diffs
                result = paired(diffs)
                for key in ("delta_points", "p_value", "ci95_points"):
                    yield f"{mode} {split} {arm} {key}", result[key], reported[key]
        for arm, reported in splits.get("pooled", {}).items():
            result = paired(pooled[arm])
            for key in ("delta_points", "p_value", "ci95_points"):
                yield f"{mode} pooled {arm} {key}", result[key], reported[key]


def check(folder):
    """Recompute one folder; return its printed lines and mismatch count."""
    lines, failures, checked = [], 0, 0
    for label, ours, theirs in recompute(folder):
        checked += 1
        if ours != theirs:
            failures += 1
            lines.append(f"MISMATCH {folder.name}: {label}: recomputed {ours}, reported {theirs}")
    return lines + [f"{folder.name}: {checked} numbers checked"], failures


def main(folders):
    folders = [Path(f) for f in folders if (Path(f) / "cases.csv").exists()]
    with ProcessPoolExecutor() as pool:  # folders are independent; each keeps its own seeded RNG
        results = list(pool.map(check, folders))
    for lines, _ in results:
        print("\n".join(lines))
    return 1 if sum(failures for _, failures in results) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or sorted(str(p) for p in Path("results").iterdir())))
