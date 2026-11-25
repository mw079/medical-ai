"""Count original reports labeled CC BY in PMC across the M1 candidate pools."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BATCH_SIZE = 500
TARGETS = {"development": 60, "validation": 40, "same_source_test": 50, "external_test": 50}


def find_cc_by(article_ids: set[str]) -> set[str]:
    article_filter = " OR ".join(f"{article_id[3:]}[uid]" for article_id in sorted(article_ids))
    params = urlencode({
        "db": "pmc",
        "term": f'"cc by license"[filter] AND ({article_filter})',
        "retmode": "json",
        "retmax": len(article_ids),
        "tool": "medical_ai_m1_license_check",
    })
    request = Request(ESEARCH, data=params.encode())
    for attempt in range(3):
        time.sleep(0.4)
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)["esearchresult"]
            break
        except HTTPError as error:
            if attempt == 2 or (error.code != 429 and error.code < 500):
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)

    matches = {f"PMC{article_id}" for article_id in result["idlist"]}
    if result.get("errorlist") or len(matches) != int(result["count"]) or not matches <= article_ids:
        raise ValueError(f"Unexpected PMC search result for batch starting with {min(article_ids)}")
    return matches


def screen_manifest(manifest: dict) -> dict:
    pools = {split: set(manifest["candidate_ids"][split]) for split in TARGETS}
    medr = pools["external_test"].union(*(manifest[key] for key in (
        "shared_pmcids", "additional_title_matches", "additional_exact_text_matches", "external_inspection_exclusions",
    )))
    if len(medr) != manifest["source_counts"]["MedR-Bench"]:
        raise ValueError("External candidate and exclusion lists do not account for every MedR-Bench report")

    all_ids = sorted(set().union(*pools.values(), medr))
    cc_by = set()
    for start in range(0, len(all_ids), BATCH_SIZE):
        cc_by.update(find_cc_by(set(all_ids[start:start + BATCH_SIZE])))

    results = {}
    for split, pool in pools.items():
        results[split] = {"pool_size": len(pool), "cc_by_count": len(pool & cc_by),
                          "target": TARGETS[split], "cc_by_ids": sorted(pool & cc_by)}
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "metadata_source": ESEARCH,
        "filter": '"cc by license"[filter]',
        "source_revisions": manifest["source_revisions"],
        "splits": results,
        "medr_full": {"pool_size": len(medr), "cc_by_count": len(medr & cc_by)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "data/m1/manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    snapshot = screen_manifest(manifest)
    args.manifest.with_name("cc_by_manifest.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    for split, result in snapshot["splits"].items():
        print(split, f"{result['cc_by_count']}/{result['pool_size']}", f"target {result['target']}")
    medr = snapshot["medr_full"]
    print("MedR-Bench before exclusions", f"{medr['cc_by_count']}/{medr['pool_size']}")


if __name__ == "__main__":
    main()
