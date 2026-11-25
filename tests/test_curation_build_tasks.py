"""Offline checks of the curation pipeline: validation rules, both reviews and resuming."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from build_tasks import export_case, process_case, review_report, run_batch
from cursor_sessions import finish_cursor_response
from load_sources import SourceReport
from prompts import REVIEW_NO_DRAFT, REVIEWER
from validation import complete_response, validate_draft


SOURCE = SourceReport(
    "fixture", "PMC1", "https://example.org", "A test case",
    "In 2002, a 43-year-old woman presented with one week of fever and productive cough. "
    "Oxygen saturation was 91% on room air. Chest radiography showed a right lower lobe "
    "consolidation. Blood culture grew Streptococcus pneumoniae. "
    "The team diagnosed pneumococcal pneumonia. "
    "At a separate visit for recurrent fever in 2016, chest CT showed a left upper lobe opacity.",
    "pneumococcal pneumonia",
)
DRAFT = {
    "status": "draft",
    "opening": "A 43-year-old woman presents with one week of fever and productive cough.",
    "opening_quote": "In 2002, a 43-year-old woman presented with one week of fever and productive cough.",
    "diagnosis": SOURCE.diagnosis,
    "diagnosis_quote": "The team diagnosed pneumococcal pneumonia.",
    "encounter": {"start": "Initial presentation in 2002", "start_year": 2002,
                  "cutoff": "End of the 2002 diagnostic workup", "cutoff_year": 2002,
                  "cutoff_quote": "The team diagnosed pneumococcal pneumonia."},
    "findings": [
        {"name": "Oxygen saturation", "result": "91% on room air", "phase": "presentation", "year": 2002,
         "quote": "Oxygen saturation was 91% on room air."},
        {"name": "Chest radiograph", "result": "Right lower lobe consolidation", "phase": "workup", "year": 2002,
         "quote": "Chest radiography showed a right lower lobe consolidation."},
        {"name": "Blood culture", "result": "Grew Streptococcus pneumoniae", "phase": "workup", "year": 2002,
         "quote": "Blood culture grew Streptococcus pneumoniae."},
    ],
}


FUTURE_FINDING = {
    "name": "Chest CT", "result": "Left upper lobe opacity", "phase": "workup", "year": 2016,
    "quote": "At a separate visit for recurrent fever in 2016, chest CT showed a left upper lobe opacity.",
}


class PipelineTests(unittest.TestCase):
    def test_export_requires_both_reviews_to_accept(self):
        reports = [replace(SOURCE, article_id=f"PMC{index}") for index in range(1, 5)]
        checked = []

        def first_gate(report, output, invoke):
            return {"article_id": report.article_id, "draft": DRAFT,
                    "status": "rejected" if report.article_id == "PMC3" else "accepted"}

        def second_gate(report, draft):
            checked.append(report.article_id)
            if report.article_id == "PMC4":
                raise TimeoutError("review unavailable")
            return {"review": {"verdict": "accept" if report.article_id == "PMC1" else "reject"}}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = SimpleNamespace(output=output, split="development", workers=1, limit=4, target=None)
            with patch("build_tasks.process_case", side_effect=first_gate):
                result = run_batch(args, reports, None, second_gate)
            exported = json.loads((output / "data/processed_case_data/development/dataset.json").read_text())
        self.assertEqual(checked, ["PMC1", "PMC2", "PMC4"])
        self.assertEqual(result["case_status"], {
            "PMC1": "accepted", "PMC2": "rejected", "PMC3": "rejected", "PMC4": "error",
        })
        self.assertEqual([item["case_id"] for item in exported], ["PMC1"])

    def test_second_review_rejection_is_saved_and_kept_on_resume(self):
        def accepted(report, output, invoke):
            return {"article_id": report.article_id, "draft": DRAFT, "status": "accepted",
                    "stage": "reviewer", "review": {"verdict": "accept", "issues": []}}

        second_gate = Mock(return_value={"review": {"verdict": "reject"}})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("build_tasks.process_case", side_effect=accepted):
                result = review_report(SOURCE, output, None, second_gate)
            saved = json.loads((output / "audit/PMC1.json").read_text())
            self.assertEqual((result["status"], result["stage"]), ("rejected", "codex"))
            self.assertEqual((saved["status"], saved["stage"]), ("rejected", "codex"))
            # On resume the saved rejection is returned without another review.
            with patch("build_tasks.process_case", side_effect=lambda r, o, i: saved):
                self.assertEqual(review_report(SOURCE, output, None, second_gate)["status"], "rejected")
        second_gate.assert_called_once()

    def test_rejects_literal_leak_wrong_label_and_made_up_quote(self):
        for field, value in (
            ("opening", "The patient has pneumococcal pneumonia."),
            ("diagnosis", "viral bronchitis"),
            ("opening_quote", "No patient matching this quote."),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_draft({**DRAFT, field: value}, SOURCE)

    def test_accepts_source_extraction_whitespace_before_punctuation(self):
        validate_draft(DRAFT, replace(SOURCE, text=SOURCE.text.replace("air.", "air .")))

    def test_allows_readable_label_and_source_backed_expansion(self):
        source = replace(SOURCE, diagnosis="PneumococcalPneumonia")
        validate_draft(DRAFT, source)
        expanded = {**DRAFT, "diagnosis": "Acute pneumococcal pneumonia"}
        validate_draft(expanded, source)
        with self.assertRaisesRegex(ValueError, "Opening contains"):
            validate_draft({**expanded, "opening": "She has pneumococcal pneumonia."}, source)

    def test_medr_allows_diagnosis_inside_narrative_label(self):
        source = replace(SOURCE, source="MedR-Bench", diagnosis=(
            "The final diagnosis is pneumococcal pneumonia. The culture supported it."
        ))
        validate_draft(DRAFT, source)
        with self.assertRaisesRegex(ValueError, "Diagnosis differs"):
            validate_draft({**DRAFT, "diagnosis": "viral bronchitis"}, source)
        with self.assertRaisesRegex(ValueError, "Diagnosis differs"):
            validate_draft(DRAFT, replace(source, source="Stanford"))

    def test_ignores_figure_citation_but_keeps_clinical_text(self):
        source = replace(SOURCE, text=SOURCE.text.replace("91% on room air.", "91% on room air (Figures 7 and 8 )."))
        validate_draft(DRAFT, source)
        with self.assertRaisesRegex(ValueError, "Oxygen saturation"):
            validate_draft(DRAFT, replace(source, text=source.text.replace("91%", "99%")))

    def test_complete_response_stops_active_cursor_run(self):
        self.assertFalse(complete_response('{"status":"draft"}'))
        self.assertFalse(complete_response('{"verdict":"reject","issues":[]}'))
        response = '{"verdict":"accept","issues":[]}'
        self.assertTrue(complete_response(response))
        run = Mock(id="run-id", status="running", usage=None)
        run.wait.return_value = SimpleNamespace(status="cancelled")
        saved = finish_cursor_response(run, "session-id", response)
        run.cancel.assert_called_once_with()
        self.assertEqual(saved["cursor_status"], "cancelled")
        self.assertEqual(saved["text"], response)

    def test_same_encounter_workup_passes_and_later_visit_fails(self):
        validate_draft(DRAFT, SOURCE)
        leaked = {**DRAFT, "findings": [*DRAFT["findings"], FUTURE_FINDING]}
        with self.assertRaisesRegex(ValueError, "after the encounter cutoff"):
            validate_draft(leaked, SOURCE)

    def test_reviewer_failure_resumes_with_saved_writer(self):
        calls = []

        def invoke(prompt, payload):
            calls.append(prompt)
            if len(calls) == 1:
                return {"session_id": "writer", "text": json.dumps(DRAFT)}
            if len(calls) == 2:
                raise TimeoutError("temporary")
            return {"session_id": "reviewer", "text": '{"verdict":"accept","issues":[]}'}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertEqual(process_case(SOURCE, path, invoke)["status"], "error")
            accepted = process_case(SOURCE, path, invoke)
            self.assertEqual(accepted["status"], "accepted")
            self.assertEqual(process_case(SOURCE, path, invoke), accepted)
            self.assertEqual(calls[1:], [REVIEWER, REVIEWER])
            self.assertEqual(export_case(accepted, "development")["case_sets"], ["development"])

    def test_writer_rejection_still_gets_a_separate_review(self):
        calls = []

        def invoke(prompt, payload):
            calls.append(prompt)
            if len(calls) == 1:
                return {"session_id": "writer", "text": '{"status":"reject","reason":"insufficient"}'}
            return {"session_id": "reviewer", "text": (
                '{"verdict":"reject","issues":[{"kind":"insufficient","detail":"Draft omitted."}]}'
            )}

        with tempfile.TemporaryDirectory() as directory:
            result = process_case(SOURCE, Path(directory), invoke)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["stage"], "writer")
        self.assertEqual(calls[1:], [REVIEW_NO_DRAFT])


if __name__ == "__main__":
    unittest.main()
