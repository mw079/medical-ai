# /// script
# requires-python = ">=3.11"
# dependencies = ["cursor-sdk==1.4.2", "pyarrow==24.0.0", "python-dotenv==1.1.0"]
# ///
"""Draft, review, and export source-grounded sequential diagnosis tasks.

For each candidate report in a fixed order: a Grok writer drafts a task from the report, local
checks validate the draft (validation.py), a separate Grok session reviews it against the
report, and a blind GPT-6 Luna session (codex_reviewer.py) reviews it again. Only cases that
pass all four steps are exported. Every step is saved per report in <output>/audit and
<output>/codex, so an interrupted run resumes without repeating finished calls.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from cursor_sdk import CursorClient
from dotenv import dotenv_values

from check_licenses import screen_manifest
import codex_reviewer
from cursor_sessions import MODEL_SELECTION, CursorSessions
from load_sources import (
    MEDR_REVISION, STANFORD_REVISION, SourceReport, content_hash, load_medr,
    load_stanford_split, prepare_candidates, load_stanford,
)
from prompts import MEDR_LABEL_GUIDANCE, REVIEW_NO_DRAFT, REVIEWER, WRITER
from validation import VALIDATION_VERSION, parse_object, validate_draft, validate_review

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("development", "validation", "same_source_test", "external_test")


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def source_payload(report: SourceReport) -> dict:
    return {"article_id": report.article_id, "title": report.title,
            "source_label": report.diagnosis, "report": report.text}


def process_case(report: SourceReport, output: Path, invoke) -> dict:
    path = output / "audit" / f"{report.article_id}.json"
    fingerprint = hashlib.sha256(json.dumps(asdict(report), sort_keys=True).encode()).hexdigest()
    record = read_json(path) if path.exists() else {
        "article_id": report.article_id, "source_url": report.source_url,
        "source_title": report.title, "source": report.source, "license_screen": "PMC CC BY filter",
        "source_sha256": fingerprint,
    }
    if record["source_sha256"] != fingerprint:
        raise ValueError(f"Source changed for {report.article_id}")
    record["source_label"] = report.diagnosis
    revalidate = record.get("stage") == "validation" and record.get("validation_version") != VALIDATION_VERSION
    if record.get("status") in ("accepted", "rejected") and "review" in record and not revalidate:
        return record
    record.pop("error", None)
    try:
        if "writer" not in record:
            record["writer"] = invoke(WRITER, source_payload(report))
            write_json(path, record)
        writer_text = record["writer"]["text"]
        stage, reason = "reviewer", None
        try:
            draft = parse_object(writer_text)
            record["draft"] = draft
            validate_draft(draft, report)
            if draft["status"] == "reject":
                stage, reason = "writer", draft["reason"]
        except ValueError as error:
            stage, reason = "validation", str(error)

        if revalidate and record.get("reason") != reason:
            previous = {field: record.pop(field) for field in ("reviewer", "review", "reason") if field in record}
            record.setdefault("prior_validation", []).append(previous)
        payload = {"source": source_payload(report)}
        if reason:
            prompt = REVIEW_NO_DRAFT
            payload.update(writer_output=writer_text, rejection_reason=reason)
        else:
            prompt = REVIEWER + MEDR_LABEL_GUIDANCE if report.source == "MedR-Bench" else REVIEWER
            payload["draft"] = draft
        if "reviewer" not in record:
            record["reviewer"] = invoke(prompt, payload)
            write_json(path, record)
        if record["writer"]["session_id"] == record["reviewer"]["session_id"]:
            raise RuntimeError("Writer and reviewer shared a session")
        try:
            review = parse_object(record["reviewer"]["text"])
            validate_review(review)
        except ValueError:
            # Preserve the malformed response for audit and retry the reviewer on resume.
            record.setdefault("invalid_reviews", []).append(record.pop("reviewer"))
            raise
        record.update(status="accepted" if not reason and review["verdict"] == "accept" else "rejected",
                      stage=stage, review=review, validation_version=VALIDATION_VERSION)
        if reason:
            record["reason"] = reason
    except Exception as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")
    write_json(path, record)
    return record


def export_case(record: dict, split: str) -> dict:
    draft = record["draft"]
    return {
        "case_id": record["article_id"],
        "case_title": record["article_id"],
        "case_summary": draft["opening"].strip(),
        "full_text": "\n".join([draft["opening"].strip()] + [
            f"{finding['name']}: {finding['result']}" for finding in draft["findings"]
        ]),
        "final_diagnosis": draft["diagnosis"].strip(),
        "case_sets": [split],
    }


def candidates(m1: Path, split: str) -> list[SourceReport]:
    manifest_path = m1 / "manifest.json"
    if not manifest_path.exists():
        cache = m1 / "sources"
        manifest, _ = prepare_candidates(load_stanford(cache), load_medr(cache))
        write_json(manifest_path, manifest)
    manifest = read_json(manifest_path)
    licenses_path = m1 / "cc_by_manifest.json"
    if not licenses_path.exists():
        write_json(licenses_path, screen_manifest(manifest))
    licenses = read_json(licenses_path)
    revisions = manifest["source_revisions"]
    if revisions != licenses["source_revisions"] or revisions["MedCaseReasoning"] != STANFORD_REVISION or revisions["MedR-Bench"] != MEDR_REVISION:
        raise ValueError("M1 source revisions do not match the loaders or license snapshot")
    eligible = set(manifest["candidate_ids"][split]) & set(licenses["splits"][split]["cc_by_ids"])
    cache = m1 / "sources"
    reports = load_medr(cache) if split == "external_test" else load_stanford_split(cache, split)
    selected = [report for report in reports if report.article_id in eligible]
    if len(selected) != len(eligible):
        raise ValueError("The candidate manifest does not match the source")
    return sorted(selected, key=lambda report: (content_hash(f"m2 {report.article_id}"), report.article_id))


def review_report(report, output, invoke, second_review):
    record = process_case(report, output, invoke)
    if second_review is None or record["status"] != "accepted":
        return record
    try:
        decision = second_review(report, record["draft"])["review"]["verdict"]
    except Exception as error:
        return {**record, "status": "error", "stage": "codex", "error": str(error)}
    if decision == "accept":
        return record
    # Save the rejection so the audit record matches the export and a resume does not re-review it.
    record = {**record, "status": "rejected", "stage": "codex"}
    write_json(output / "audit" / f"{report.article_id}.json", record)
    return record


def run_batch(args, reports, invoke, second_review=None) -> dict:
    records = []
    accepted = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        limit = args.limit or len(reports)
        while len(records) < min(limit, len(reports)):
            size = args.workers
            if args.target:
                size = min(size, args.target - len(accepted))
                if size == 0:
                    break
            batch = reports[len(records):min(len(records) + size, limit)]
            pending = [executor.submit(review_report, report, args.output, invoke, second_review)
                       for report in batch]
            for future in as_completed(pending):
                record = future.result()
                detail = f" ({record['error']})" if record["status"] == "error" else ""
                print(f"{record['article_id']} {record['status']}{detail}", flush=True)
            for future in pending:
                record = future.result()
                records.append(record)
                if record["status"] == "accepted":
                    accepted.append(export_case(record, args.split))
            write_json(args.output / "data" / "processed_case_data" / args.split / "dataset.json", accepted)
    counts = {state: sum(record["status"] == state for record in records)
              for state in ("accepted", "rejected", "error")}
    summary = {"split": args.split, "attempted": len(records), **counts,
               "case_status": {record["article_id"]: record["status"] for record in records}}
    write_json(args.output / "summary.json", summary)
    return summary


def second_reviewer(output: Path, key: str):
    """Blind second review. MedR-Bench labels get the same guidance the first reviewer gets."""
    def review(report, draft):
        prompt = codex_reviewer.PROMPT + (MEDR_LABEL_GUIDANCE if report.source == "MedR-Bench" else "")
        return codex_reviewer.review_case(source_payload(report), codex_reviewer.blind_draft(draft), key,
                                          output / "codex" / f"{report.article_id}.json", prompt=prompt)
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", type=Path, default=ROOT / "data/m1", help="candidate manifest and source cache")
    parser.add_argument("--output", type=Path, required=True, help="batch folder, one per split and run")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--skip", type=int, default=0, help="skip this many reports in the fixed candidate order")
    parser.add_argument("--limit", type=int, help="maximum number of source reports attempted")
    parser.add_argument("--target", type=int, help="stop after this many accepted cases")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.limit is None and args.target is None:
        parser.error("Pass --limit, --target, or both")
    if any(value is not None and value < 1 for value in (args.limit, args.target, args.workers)):
        parser.error("Limits, targets and workers must be positive")
    if args.skip < 0:
        parser.error("--skip must be non-negative")
    env = dotenv_values(ROOT / ".env")
    cursor_key = os.environ.get("CURSOR_API_KEY") or env.get("CURSOR_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY") or env.get("OPENAI_API_KEY")
    if not cursor_key or not openai_key:
        parser.error("Set CURSOR_API_KEY and OPENAI_API_KEY in the environment or .env")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another run already uses this output directory")
        reports = candidates(args.m1, args.split)[args.skip:]
        if not reports:
            parser.error("--skip leaves no eligible reports")
        settings = {"split": args.split, "model": asdict(MODEL_SELECTION), "cursor_sdk": "1.4.2", "limit": args.limit,
                    "target": args.target, "writer_prompt": WRITER, "reviewer_prompt": REVIEWER,
                    "rejection_prompt": REVIEW_NO_DRAFT,
                    "source_revision": MEDR_REVISION if args.split == "external_test" else STANFORD_REVISION,
                    "candidate_order": [report.article_id for report in reports]}
        if args.skip:
            settings["skip"] = args.skip
        settings["codex_review"] = {"model": codex_reviewer.MODEL, "effort": codex_reviewer.EFFORT,
                                    "prompt": codex_reviewer.PROMPT, "schema": codex_reviewer.SCHEMA}
        if args.split == "external_test":
            settings["codex_review"]["medr_label_guidance"] = MEDR_LABEL_GUIDANCE
        settings_path = args.output / "settings.json"
        if settings_path.exists() and read_json(settings_path) != settings:
            parser.error("Run settings changed. Choose a new --output directory")
        write_json(settings_path, settings)
        with tempfile.TemporaryDirectory(prefix="medical-curation-") as directory:
            workspace = Path(directory)
            # The SDK treats run-scoped cancellation as cloud-routed unless this owned bridge allows fallback.
            with CursorClient.launch_bridge(workspace=workspace, state_root=workspace / "state",
                                            allow_api_key_env_fallback=True, client_timeout=180) as client:
                summary = run_batch(args, reports, CursorSessions(client, workspace, cursor_key),
                                    second_reviewer(args.output, openai_key))
    print(json.dumps({key: value for key, value in summary.items() if key != "case_status"}))
    if summary["error"] or (args.target and summary["accepted"] < args.target):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
