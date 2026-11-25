"""Read the packed benchmark and prepare the writable files a run needs.

benchmark/<split>.json holds, per split: the SDBench-format cases, each case's accepted curation
record ("audit") and blind review, and the shared gatekeeper cache with its question evidence.
The benchmark folder is read-only. A run works in data/benchmark/<split>/: SDBench reads the
dataset there, and the gatekeeper cache there is seeded from the packed file and extended by new
requests, so committed files never change.
"""

import fcntl
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"
WORK = ROOT / "data/benchmark"
SPLITS = ("development", "validation", "same_source_test", "external_test")


def load_split(split, benchmark=BENCHMARK):
    return json.loads((Path(benchmark) / f"{split}.json").read_text())


def _write_missing(path, text):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".seed")
        tmp.write_text(text)
        tmp.replace(path)


def prepare(split, cache_name, work=WORK, benchmark=BENCHMARK):
    """Return (packed split, working folder) after seeding the working folder."""
    packed = load_split(split, benchmark)
    root = Path(work) / split
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".seed.lock").open("a") as lock:  # several runs may start at once
        fcntl.flock(lock, fcntl.LOCK_EX)
        dataset = root / "data/processed_case_data" / split / "dataset.json"
        text = json.dumps(packed["cases"], indent=2, ensure_ascii=False) + "\n"
        if not dataset.exists() or dataset.read_text() != text:
            dataset.unlink(missing_ok=True)
            _write_missing(dataset, text)
        cache = root / cache_name
        for case_id, record in packed["gatekeeper_cache"].items():
            _write_missing(cache / f"{case_id}.json", json.dumps(record, indent=1, ensure_ascii=False))
        for case_id, ids in packed["question_evidence"].items():
            _write_missing(cache / "question-evidence" / f"{case_id}.json", json.dumps(ids, indent=2) + "\n")
    return packed, root
