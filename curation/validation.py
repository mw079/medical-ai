"""Checks that run on every draft and review without calling a model.

A draft must quote the source verbatim for its opening, diagnosis, cutoff and every finding, keep
the reference diagnosis out of the opening, use the source's diagnosis label, and keep a
consistent encounter timeline. A review must have a valid verdict and issue kinds.
"""

from __future__ import annotations

import json
import re

from load_sources import SourceReport

# Bump when these rules change. Saved cases checked under an older version are re-validated.
VALIDATION_VERSION = 5
ISSUE_KINDS = {"leakage", "unsupported", "diagnosis", "insufficient", "timeline", "multiple_patients"}


def parse_object(text: str) -> dict:
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), re.DOTALL)
    value = json.loads(fenced[1] if fenced else text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def string(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or empty {field}")
    return value.strip()


def normalized(text: str) -> str:
    return " ".join(text.split())


def normalized_quote(text: str) -> str:
    # Figure references are editorial annotations. Leave clinical parentheticals intact.
    text = re.sub(r"\(\s*(?:fig(?:ure)?s?\.?)\s*\d+[a-z]?"
                  r"(?:\s*(?:and|,|[-–])\s*\d+[a-z]?)*\s*\)", "", text, flags=re.I)
    return re.sub(r" +(\)|[.,;:!?](?= |$))", r"\1", normalized(text))


def label_words(text: str) -> str:
    # The source sometimes joins words in title case, such as ObstructiveUropathy.
    separated = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return " ".join(re.findall(r"\w+", separated.casefold()))


def year(value: dict, field: str) -> int | None:
    if field not in value:
        raise ValueError(f"Missing {field}; use null for an undocumented year")
    result = value[field]
    if result is not None and (type(result) is not int or not 1800 <= result <= 2200):
        raise ValueError(f"Invalid {field}")
    return result


def validate_timing(draft: dict, findings: list[dict]) -> None:
    encounter = draft.get("encounter")
    if not isinstance(encounter, dict):
        raise ValueError("Missing encounter timing")
    string(encounter.get("start"), "encounter start")
    string(encounter.get("cutoff"), "encounter cutoff")
    start, cutoff = year(encounter, "start_year"), year(encounter, "cutoff_year")
    if start is not None and cutoff is not None and cutoff < start:
        raise ValueError("Encounter cutoff predates its opening")
    for finding in findings:
        name = finding["name"]
        phase = finding.get("phase")
        if phase not in ("history", "presentation", "workup"):
            raise ValueError(f"{name} has an invalid encounter phase")
        date = year(finding, "year")
        if date is None:
            continue
        if cutoff is not None and date > cutoff:
            raise ValueError(f"{name} occurs after the encounter cutoff")
        if start is not None and (
            (phase == "history" and date > start)
            or (phase == "presentation" and date != start)
            or (phase == "workup" and date < start)
        ):
            raise ValueError(f"{name} contradicts its encounter phase")


def validate_draft(draft: dict, source: SourceReport) -> None:
    if draft.get("status") == "reject":
        string(draft.get("reason"), "reason")
        return
    if draft.get("status") != "draft":
        raise ValueError("Invalid draft status")
    opening = string(draft.get("opening"), "opening")
    diagnosis = string(draft.get("diagnosis"), "diagnosis")
    diagnosis_words = label_words(diagnosis)
    source_words = label_words(source.diagnosis)
    # An expanded label still needs a reviewer to confirm clinical equivalence.
    short_medr_label = (source.source == "MedR-Bench" and bool(diagnosis_words)
                        and f" {diagnosis_words} " in f" {source_words} ")
    if (diagnosis_words.replace(" ", "") != source_words.replace(" ", "")
            and f" {source_words} " not in f" {diagnosis_words} " and not short_medr_label):
        raise ValueError("Diagnosis differs from the source label")
    opening_words = f" {label_words(opening)} "
    if any(words and f" {words} " in opening_words for words in (diagnosis_words, source_words)):
        raise ValueError("Opening contains the reference diagnosis")
    findings = draft.get("findings")
    if not isinstance(findings, list) or not 3 <= len(findings) <= 8:
        raise ValueError("Expected 3 to 8 findings")
    if not all(isinstance(finding, dict) for finding in findings):
        raise ValueError("Each finding must be an object")
    names = [string(finding.get("name"), "finding name").casefold() for finding in findings]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate finding names")
    for finding in findings:
        string(finding.get("result"), "finding result")
    validate_timing(draft, findings)
    quotes = [("opening", draft.get("opening_quote")), ("diagnosis", draft.get("diagnosis_quote")),
              ("cutoff", draft["encounter"].get("cutoff_quote"))]
    quotes.extend((finding["name"], finding.get("quote")) for finding in findings)
    report = normalized_quote(source.text)
    for number, (name, quote) in enumerate(quotes, 1):
        passage = normalized_quote(string(quote, f"quote {number}"))
        if not passage or passage not in report:
            raise ValueError(f"Quote {number} for {name} is missing from the source")


def validate_review(review: dict) -> None:
    verdict = review.get("verdict")
    issues = review.get("issues")
    if verdict not in ("accept", "reject") or not isinstance(issues, list) or bool(issues) != (verdict == "reject"):
        raise ValueError("Invalid review verdict or issues")
    for issue in issues:
        if not isinstance(issue, dict) or issue.get("kind") not in ISSUE_KINDS:
            raise ValueError("Invalid issue kind")
        string(issue.get("detail"), "issue detail")


def complete_response(text: str) -> bool:
    if not text.rstrip().endswith(("}", "```")):
        return False
    try:
        value = parse_object(text)
    except ValueError:
        return False
    if value.get("status") == "draft":
        return {"opening", "opening_quote", "diagnosis", "diagnosis_quote", "encounter", "findings"} <= value.keys()
    if value.get("status") == "reject":
        return isinstance(value.get("reason"), str) and bool(value["reason"].strip())
    try:
        validate_review(value)
    except ValueError:
        return False
    return True
