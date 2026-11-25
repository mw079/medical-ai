# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow==24.0.0"]
# ///
"""Load the two pinned source datasets and build the candidate manifest.

Stanford MedCaseReasoning keeps its own train, validation and test splits (development,
validation and same-source test here). MedR-Bench supplies the external test pool after removing
reports that match a Stanford report by PMC ID, normalized title or normalized text, and five
reports that were read during design (chosen by hash, before reading).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.request import urlopen

import pyarrow.parquet as parquet


STANFORD_REVISION = "469a5365bc534b5b2b9cfbc52b2ef2a10f43a339"
MEDR_REVISION = "ff60ab440afd2f2bc0c603b3a65d715ec83138a7"
SDBENCH_REVISION = "304ce9625a046cea97d64e3b1ba5d19217b5095c"
STANFORD_SPLITS = {"train": "development", "val": "validation", "test": "same_source_test"}

@dataclass(frozen=True)
class SourceReport:
    source: str
    article_id: str
    source_url: str
    title: str
    text: str
    diagnosis: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"PMC\d+", self.article_id):
            raise ValueError(f"Invalid article ID in {self.source}: {self.article_id!r}")
        for field in ("text", "diagnosis", "source_url"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing {field} for {self.article_id}")


def download(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def pmc_url(article_id: str) -> str:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{article_id}/"


def load_stanford_split(cache: Path, split: str) -> list[SourceReport]:
    source = "MedCaseReasoning"
    upstream = next(name for name, candidate in STANFORD_SPLITS.items() if candidate == split)
    base = f"https://huggingface.co/datasets/zou-lab/MedCaseReasoning/resolve/{STANFORD_REVISION}/data"
    filename = f"{upstream}-00000-of-00001.parquet"
    path = download(f"{base}/{filename}", cache / source / STANFORD_REVISION / filename)
    columns = ["pmcid", "title", "text", "final_diagnosis"]
    return [
        SourceReport(source, row["pmcid"], pmc_url(row["pmcid"]), row["title"], row["text"], row["final_diagnosis"])
        for row in parquet.read_table(path, columns=columns).to_pylist()
    ]


def load_stanford(cache: Path) -> dict[str, list[SourceReport]]:
    return {split: load_stanford_split(cache, split) for split in STANFORD_SPLITS.values()}


def unique_entries(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate key in MedR-Bench JSON")
    return result


def load_medr(cache: Path) -> list[SourceReport]:
    source = "MedR-Bench"
    filename = "diagnosis_957_cases_with_rare_disease_491.json"
    base = f"https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/{MEDR_REVISION}/data/MedRBench"
    path = download(f"{base}/{filename}", cache / source / MEDR_REVISION / filename)
    with path.open() as file:
        records = json.load(file, object_pairs_hook=unique_entries)
    return [
        SourceReport(source, article_id, pmc_url(article_id), row["raw_case"].strip().splitlines()[0].strip(),
                     row["raw_case"], row["generate_case"]["final_diagnosis"])
        for article_id, row in records.items()
    ]


def ids_for(reports: list[SourceReport]) -> set[str]:
    ids = {report.article_id for report in reports}
    if len(ids) != len(reports):
        raise ValueError("Duplicate article IDs within the source or across Stanford splits")
    return ids


def normalized_title(title: str) -> str:
    return " ".join(re.findall(r"\w+", (title or "").casefold()))


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def review_sample(reports: list[SourceReport]) -> list[SourceReport]:
    # Choose before reading the reports. The hash fixes the order across Python versions.
    return sorted(reports, key=lambda report: content_hash(f"m1-review {report.article_id}"))[:5]


def prepare_candidates(stanford: dict[str, list[SourceReport]], medr: list[SourceReport]) -> tuple[dict, dict]:
    stanford_reports = [report for reports in stanford.values() for report in reports]
    stanford_ids = ids_for(stanford_reports)
    exact_overlap = stanford_ids & ids_for(medr)
    stanford_titles = {normalized_title(report.title) for report in stanford_reports}
    stanford_hashes = {content_hash(report.text) for report in stanford_reports}
    title_matches, text_matches = set(), set()
    for report in medr:
        if report.article_id in exact_overlap:
            continue
        title = normalized_title(report.title)
        if title and title in stanford_titles:
            title_matches.add(report.article_id)
        if content_hash(report.text) in stanford_hashes:
            text_matches.add(report.article_id)

    excluded = exact_overlap | title_matches | text_matches
    external_pool = [report for report in medr if report.article_id not in excluded]
    samples = {"development": review_sample(stanford["development"]), "external_inspection": review_sample(external_pool)}
    inspected = ids_for(samples["external_inspection"])
    candidate_ids = {split: sorted(ids_for(reports)) for split, reports in stanford.items()}
    candidate_ids["external_test"] = sorted(ids_for(external_pool) - inspected)
    manifest = {
        "source_revisions": {"MedCaseReasoning": STANFORD_REVISION, "MedR-Bench": MEDR_REVISION, "SDBench": SDBENCH_REVISION},
        "source_counts": {"MedCaseReasoning": len(stanford_reports), "MedR-Bench": len(medr)},
        "shared_pmcids": sorted(exact_overlap),
        "additional_title_matches": sorted(title_matches),
        "additional_exact_text_matches": sorted(text_matches),
        "external_inspection_exclusions": sorted(inspected),
        "candidate_ids": candidate_ids,
    }
    return manifest, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "m1")
    args = parser.parse_args()
    cache = args.output / "sources"
    manifest, _ = prepare_candidates(load_stanford(cache), load_medr(cache))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Source counts", manifest["source_counts"])
    print("Matches", {key: len(manifest[key]) for key in ("shared_pmcids", "additional_title_matches", "additional_exact_text_matches")})
    print("Candidate counts", {split: len(ids) for split, ids in manifest["candidate_ids"].items()})

if __name__ == "__main__":
    main()
