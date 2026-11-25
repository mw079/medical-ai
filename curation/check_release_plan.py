# /// script
# requires-python = ">=3.11"
# dependencies = ["cursor-sdk==1.4.2", "pyarrow==24.0.0", "python-dotenv==1.1.0"]
# ///
"""Check a proposed 200-case release without invoking any models."""

from itertools import combinations
import json
from pathlib import Path

from build_tasks import ROOT, candidates, read_json
from load_sources import content_hash, normalized_title


# The attempt cap is 2.5x each target, corresponding to a 40% accepted yield.
# Development skips the 10 + 30 + 8 reports already inspected in pilots.
BATCHES = {
    "development": (60, 150, 48),
    "validation": (40, 100, 0),
    "same_source_test": (50, 125, 0),
    "external_test": (50, 125, 0),
}


def main() -> None:
    m1 = ROOT / "data/m1"
    manifest = read_json(m1 / "manifest.json")
    inspected = set(manifest["external_inspection_exclusions"])
    for pilot in ("timing", "fresh30", "readiness8"):
        path = ROOT / "data/m2" / pilot / "settings.json"
        if not path.is_file():
            continue
        settings = read_json(path)
        if settings["split"] != "development" or not isinstance(settings["limit"], int):
            raise ValueError(f"Unexpected settings in the {pilot} pilot")
        inspected.update(settings["candidate_order"][:settings["limit"]])
    report_sets = {}
    results = {}
    for split, (target, limit, skip) in BATCHES.items():
        pool = candidates(m1, split)
        selected = pool[skip:skip + limit]
        ids = {report.article_id for report in selected}
        if not 0 < target <= limit == len(selected) == len(ids):
            raise ValueError(f"{split} does not have enough distinct candidates for its cap")
        if ids & inspected:
            raise ValueError(f"{split} overlaps with inspected source reports")
        output = Path("data/m2/release200") / split
        results[split] = {
            "accepted_target": target, "attempt_limit": limit, "skip": skip,
            "eligible_after_skip": len(pool) - skip, "output": str(output),
            "argv": ["uv", "run", "--locked", "curation/build_tasks.py", "--split", split,
                     "--target", str(target), "--limit", str(limit), "--skip", str(skip),
                     "--output", str(output)],
        }
        report_sets[split] = (ids, {content_hash(r.text) for r in selected},
                              {normalized_title(r.title) for r in selected} - {""})

    for (a, keys_a), (b, keys_b) in combinations(report_sets.items(), 2):
        for name, x, y in zip(("article ID", "exact normalized text", "normalized title"), keys_a, keys_b, strict=True):
            if x & y:
                raise ValueError(f"{a} and {b} overlap by {name}")
    if sum(batch["accepted_target"] for batch in results.values()) != 200:
        raise ValueError("Accepted targets must total 200")
    if len({batch["output"] for batch in results.values()}) != len(results):
        raise ValueError("Each split needs its own output directory")
    print(json.dumps({"batches": results, "id_text_title_overlap": 0,
                      "models_called": 0}, indent=2))


if __name__ == "__main__":
    main()
