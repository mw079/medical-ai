"""Preflight and run the four capped curation batches."""

import argparse
from datetime import UTC, datetime
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/m2/release200"
LOG = RELEASE / "run.log"
STATUS = RELEASE / "status.json"


def write_status(state):
    state["updated_at"] = datetime.now(UTC).isoformat()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATUS)


def run(workers):
    with (RELEASE / ".release.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"state": "preflight", "active_split": None, "splits": {}}
        write_status(state)
        try:
            checked = subprocess.check_output(
                ["uv", "run", "--locked", "curation/check_release_plan.py"], cwd=ROOT, text=True,
            )
            batches = json.loads(checked)["batches"]
            state["splits"] = {
                split: {"target": batch["accepted_target"], "attempt_limit": batch["attempt_limit"],
                        "state": "pending"}
                for split, batch in batches.items()
            }
            for split, batch in batches.items():
                progress = state["splits"][split]
                state.update(state="running", active_split=split)
                progress["state"] = "running"
                write_status(state)
                print(f"Starting {split}: target {progress['target']}, cap {progress['attempt_limit']}", flush=True)
                result = subprocess.run([*batch["argv"], "--workers", str(workers)], cwd=ROOT)
                summary_file = ROOT / batch["output"] / "summary.json"
                if not summary_file.is_file():
                    raise RuntimeError(f"{split} stopped before producing a summary, exit {result.returncode}")
                summary = json.loads(summary_file.read_text())
                progress.update({name: summary[name] for name in ("attempted", "accepted", "rejected", "error")})
                progress["exit_code"] = result.returncode
                if result.returncode == 0 and progress["accepted"] == progress["target"]:
                    progress["state"] = "complete"
                elif progress["error"]:
                    progress["state"] = "processing_errors"
                else:
                    progress["state"] = "shortfall"
                write_status(state)
            state.update(state="complete" if all(p["state"] == "complete" for p in state["splits"].values())
                         else "finished_with_shortfalls", active_split=None)
            write_status(state)
            print(json.dumps(state, indent=2), flush=True)
        except Exception as error:
            state.update(state="failed", reason=str(error))
            write_status(state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent reports within one split")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    RELEASE.mkdir(parents=True, exist_ok=True)
    if args.background:
        with LOG.open("ab") as log:
            job = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--workers", str(args.workers)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
            )
        print(f"PID {job.pid} | log {LOG} | status {STATUS}")
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
