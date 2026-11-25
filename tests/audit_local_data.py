"""Audit the released benchmark: case counts, license screen, split overlap and source records."""

import argparse
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

SPLITS = {"development": 60, "validation": 40, "same_source_test": 50, "external_test": 50}


def read(path):
    return json.loads(path.read_text())


def audit_cases(release, license_manifest, source_cache=None):
    if source_cache is not None:
        from load_sources import content_hash, load_medr, load_stanford
        originals = load_stanford(source_cache)
        originals["external_test"] = load_medr(source_cache)
        original_hashes = {split: {r.article_id: content_hash(r.text) for r in reports}
                           for split, reports in originals.items()}
    # The full PMC license screen is local (curation); without it, check each record's screen label.
    licenses = read(license_manifest)["splits"] if license_manifest.exists() else None
    seen = {kind: set() for kind in ("article ID", "source record fingerprint", "normalized title")}
    if source_cache is not None:
        seen["normalized source text"] = set()
    counts = {}
    for split, expected in SPLITS.items():
        packed = read(release / f"{split}.json")
        rows = packed["cases"]
        if len(rows) != expected:
            raise ValueError(f"{split}: expected {expected} cases, found {len(rows)}")
        eligible = set(licenses[split]["cc_by_ids"]) if licenses else None
        source = "MedR-Bench" if split == "external_test" else "MedCaseReasoning"
        local = {kind: set() for kind in seen}
        for row in rows:
            case_id = row["case_id"]
            if not re.fullmatch(r"PMC\d+", case_id):
                raise ValueError(f"{split}: invalid article ID")
            audit = packed["audit"][case_id]
            second = packed["blind_review"].get(case_id)
            if second is None or second["review"]["verdict"] != "accept":
                raise ValueError(f"{split}: {case_id} lacks an accepting second review")
            if (row["case_sets"] != [split] or audit["article_id"] != case_id or audit["status"] != "accepted"
                    or audit["source"] != source or audit["license_screen"] != "PMC CC BY filter"
                    or (eligible is not None and case_id not in eligible)
                    or audit["source_url"] != f"https://pmc.ncbi.nlm.nih.gov/articles/{case_id}/"):
                raise ValueError(f"{split}: inconsistent source, split or acceptance metadata")
            title = " ".join(re.findall(r"\w+", audit["source_title"].casefold()))
            source_hash = audit["source_sha256"]
            if not title or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
                raise ValueError(f"{split}: missing title or invalid source hash")
            keys = {"article ID": case_id, "source record fingerprint": source_hash, "normalized title": title}
            if source_cache is not None:
                keys["normalized source text"] = original_hashes[split][case_id]
            for kind, value in keys.items():
                if value in seen[kind] or (kind == "article ID" and value in local[kind]):
                    raise ValueError(f"{split}: duplicate or overlapping {kind}")
                local[kind].add(value)
            label = row["final_diagnosis"].strip()
            if not label or re.search(rf"(?<!\w){re.escape(label)}(?!\w)", row["case_summary"], re.I):
                raise ValueError(f"{split}: empty label or exact diagnosis in opening")
        exported = {row["case_id"] for row in rows}
        unreleased = sorted(set(packed["audit"]) - exported)
        if unreleased:
            raise ValueError(f"{split}: audit records without a released case: {unreleased[:3]}")
        for kind, values in local.items():
            seen[kind].update(values)
        counts[split] = len(rows)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "benchmark")
    parser.add_argument("--license-manifest", type=Path, default=ROOT / "data/m1/cc_by_manifest.json")
    parser.add_argument("--source-cache", type=Path, help="Compare cached full source text via the offline runner")
    args = parser.parse_args(argv)
    print(json.dumps({
        "verified_case_counts": audit_cases(args.release, args.license_manifest, args.source_cache),
        "checked_normalized_source_text": args.source_cache is not None,
    }, indent=2))


if __name__ == "__main__":
    main()
