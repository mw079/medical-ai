"""Check lesson-card validation and embedding retrieval without network access."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from build_memory import validate
from memory import ExperienceMemory, format_cards

WORDS = ("fever", "cough", "rash", "weakness")


class FakeEmbeddings:
    """Embeds text as counts of a few keywords, so similarity is predictable."""

    def __init__(self):
        self.calls = 0
        self.embeddings = self

    def create(self, model, input):
        self.calls += 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=[t.lower().count(w) + 0.01 for w in WORDS])
                                     for t in input])


def card(presentation, diagnosis):
    return {"case_id": diagnosis, "presentation": presentation, "decisive_evidence": ["a test"],
            "missteps": [], "strategy": "a lesson", "diagnosis": diagnosis}


class MemoryTests(unittest.TestCase):
    def test_retrieves_the_most_similar_cards_and_caches_embeddings(self):
        with TemporaryDirectory() as tmp:
            cards = Path(tmp) / "cards.json"
            cards.write_text(json.dumps({"cards": [card("fever and cough", "pneumonia"),
                                                   card("rash", "eczema"),
                                                   card("leg weakness", "myelopathy")]}))
            client = FakeEmbeddings()
            memory = ExperienceMemory(client, cards, Path(tmp) / "emb.json")
            top = memory.retrieve("cough with fever for a week", k=2)
            self.assertEqual(top[0]["diagnosis"], "pneumonia")
            self.assertEqual(len(top), 2)
            again = ExperienceMemory(FakeEmbeddings(), cards, Path(tmp) / "emb.json")
            self.assertEqual(again.client.calls, 0)  # card embeddings came from the cache
            self.assertIn("Diagnosis in that case: pneumonia", format_cards(top))

    def test_card_validation(self):
        good = {"presentation": "Adult with fever", "decisive_evidence": ["Chest X-ray: consolidation"],
                "missteps": [], "strategy": "Image early."}
        self.assertEqual(validate(good), good)
        for bad in ({**good, "extra": 1}, {**good, "decisive_evidence": []},
                    {**good, "presentation": "x" * 376}, {**good, "missteps": ["y" * 251]}):
            with self.assertRaises(ValueError):
                validate(bad)


if __name__ == "__main__":
    unittest.main()
