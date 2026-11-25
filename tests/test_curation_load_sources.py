"""Offline checks of source loading and split isolation."""

import unittest

from curation.load_sources import SourceReport, prepare_candidates


def report(article_id: str, title: str = "", text: str = "Some source text", diagnosis: str = "Example diagnosis"):
    return SourceReport("fixture", article_id, f"https://example.org/{article_id}", title, text, diagnosis)


class SourceLoadingTests(unittest.TestCase):
    def test_report_requires_source_text_and_diagnosis(self):
        for fields in ({"text": " "}, {"diagnosis": ""}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                report("PMC1", **fields)

    def test_duplicate_stanford_article_across_splits_fails(self):
        stanford = {"development": [report("PMC1")], "validation": [report("PMC1")], "same_source_test": []}
        with self.assertRaisesRegex(ValueError, "Duplicate article IDs"):
            prepare_candidates(stanford, [])

    def test_overlap_and_review_examples_are_excluded_deterministically(self):
        stanford = {
            "development": [report("PMC1", "First case", "Some source text")],
            "validation": [report("PMC2", "Validation case", "Validation text")],
            "same_source_test": [report("PMC3", "Test case", "Test text")],
        }
        medr = [report("PMC2"), report("PMC4", "FIRST CASE!", "Different text"), report("PMC5", "", "  Test TEXT ")]
        medr.extend(report(f"PMC{i}", f"Article {i}", f"Body {i}") for i in range(10, 18))
        manifest, _ = prepare_candidates(stanford, medr)
        reversed_manifest, _ = prepare_candidates(stanford, list(reversed(medr)))
        self.assertEqual(manifest, reversed_manifest)
        self.assertEqual(manifest["shared_pmcids"], ["PMC2"])
        self.assertEqual(manifest["additional_title_matches"], ["PMC4"])
        self.assertEqual(manifest["additional_exact_text_matches"], ["PMC5"])
        inspected = set(manifest["external_inspection_exclusions"])
        external_test = set(manifest["candidate_ids"]["external_test"])
        self.assertEqual(len(inspected), 5)
        self.assertEqual(len(external_test), 3)
        self.assertTrue(inspected.isdisjoint(external_test))


if __name__ == "__main__":
    unittest.main()
