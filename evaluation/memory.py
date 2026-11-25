"""Experience memory: retrieve lesson cards from development cases by embedding similarity.

Cards come from experiments/build_memory.py (evaluation/harnesses/memory/dev_lessons.json). Each
card is embedded once from its presentation and decisive evidence; embeddings are cached on disk
by text hash. A query is the case opening plus the findings released so far.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARDS = ROOT / "evaluation/harnesses/memory/dev_lessons.json"
EMBEDDINGS = ROOT / "data/memory/embeddings.json"
EMBEDDING_MODEL = "text-embedding-3-small"


def card_text(card):
    return card["presentation"] + "\n" + "\n".join(card["decisive_evidence"])


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class ExperienceMemory:
    def __init__(self, client, cards_path=DEFAULT_CARDS, cache_path=EMBEDDINGS):
        self.client = client
        self.cards = json.loads(Path(cards_path).read_text())["cards"]
        self.cache_path = Path(cache_path)
        self.lock = Lock()
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        vectors = self._embed([card_text(card) for card in self.cards], persist=True)
        self.vectors = list(zip(self.cards, vectors, strict=True))

    def _embed(self, texts, *, persist=False):
        keys = [hashlib.sha256(f"{EMBEDDING_MODEL}\n{t}".encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        if missing:
            response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=[t for _, t in missing])
            with self.lock:
                for (key, _), item in zip(missing, response.data, strict=True):
                    self.cache[key] = item.embedding
                if persist:  # several evaluation processes may start at once; write atomically
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = self.cache_path.with_name(f"{self.cache_path.name}.{os.getpid()}.tmp")
                    tmp.write_text(json.dumps(self.cache))
                    tmp.replace(self.cache_path)
        return [self.cache[k] for k in keys]

    def retrieve(self, query, k=3):
        """The k most similar cards to the query, most similar first, with their similarity."""
        (vector,) = self._embed([query])
        ranked = sorted(((cosine(vector, v), card) for card, v in self.vectors), key=lambda x: -x[0])
        return [{**card, "similarity": round(score, 4)} for score, card in ranked[:k]]


def format_cards(cards):
    """Render retrieved cards for a prompt. The source diagnosis is labeled as a past case."""
    blocks = []
    for n, card in enumerate(cards, 1):
        blocks.append(
            f"Past case {n} (similarity {card['similarity']:.2f}): {card['presentation']}\n"
            f"  Diagnosis in that case: {card['diagnosis']}\n"
            f"  Decisive evidence: {'; '.join(card['decisive_evidence'])}\n"
            f"  Missteps seen: {'; '.join(card['missteps']) or 'none'}\n"
            f"  Lesson: {card['strategy']}")
    return "\n".join(blocks)
