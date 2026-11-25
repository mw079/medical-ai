"""Copy the 200 released cases from the curation workspace into the shareable benchmark/ folder.

Each split becomes one file, benchmark/<split>.json, with what evaluation and the offline audit
read: the SDBench-format cases, the accepted writer/reviewer record and blind review of each case,
and the shared gatekeeper cache (which question releases which finding). It also writes
cases.csv with the source attribution and license screen of every case. With --prices it also
rewrites prices.json.gz, the reviewed price of every test order in the reported runs, so reruns
price them identically.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("development", "validation", "same_source_test", "external_test")
CACHE = "gatekeeper-cache-v1"
PRICES = ROOT / "data/costing_data/estimates-gpt-5-mini-e038cec6defd6725-reviewed-cpt-bedside-v3"
PRICE_FIELDS = ("test_name", "cost", "method", "billing_review", "explanation")


def read(path):
    return json.loads(path.read_text())


def optional(path):
    return read(path) if path.exists() else None


def export(release, licenses, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in SPLITS:
        source = release / split
        cases = read(source / "data/processed_case_data" / split / "dataset.json")
        ids = [row["case_id"] for row in cases]
        eligible = set(licenses["splits"][split]["cc_by_ids"])
        if not set(ids) <= eligible:
            raise ValueError(f"{split}: released cases outside the CC BY screen")
        packed = {"split": split, "cases": cases,
                  "audit": {i: read(source / "audit" / f"{i}.json") for i in ids},
                  "blind_review": {i: read(source / "codex" / f"{i}.json") for i in ids},
                  "gatekeeper_cache": {i: r for i in ids if (r := optional(source / CACHE / f"{i}.json"))},
                  "question_evidence": {i: r for i in ids
                                        if (r := optional(source / CACHE / "question-evidence" / f"{i}.json")) is not None}}
        (output / f"{split}.json").write_text(json.dumps(packed, indent=1, ensure_ascii=False) + "\n")
        for case_id in ids:
            audit = packed["audit"][case_id]
            rows.append({"split": split, "case_id": case_id, "source": audit["source"],
                         "title": " ".join(audit["source_title"].split()), "url": audit["source_url"],
                         "license_screen": audit["license_screen"]})
    with (output / "cases.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_prices(cache_dir, path):
    """Keep the fields a run uses from each reviewed price, keyed by its cache file name."""
    prices = {}
    for file in sorted(cache_dir.glob("*.json")):
        record = read(file)
        prices[file.stem] = {key: record[key] for key in PRICE_FIELDS if key in record}
        if "cpt_estimate" in record:
            prices[file.stem]["cpt_codes"] = record["cpt_estimate"].get("cpt_codes")
    data = json.dumps(prices, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    return len(prices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "data/m2/release200")
    parser.add_argument("--licenses", type=Path, default=ROOT / "data/m1/cc_by_manifest.json")
    parser.add_argument("--prices", type=Path, nargs="?", const=PRICES,
                        help="also rewrite prices.json.gz from this reviewed price cache (default: the policy v4 "
                             "cache). Off by default: the local cache grows with every run, and the shipped "
                             "file should hold only the prices of the reported runs.")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark")
    args = parser.parse_args()
    cases = export(args.release, read(args.licenses), args.output)
    print(f"Exported {cases} cases to {args.output}")
    if args.prices:
        prices = export_prices(args.prices, args.output / "prices.json.gz")
        print(f"Exported {prices} reviewed prices")


if __name__ == "__main__":
    main()
