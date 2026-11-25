"""Price test orders with SDBench's chargemaster, plus an independent billing-code check.

SDBench maps each order to CPT codes in the hospital price list. A second model call must approve
that the codes cover the order; otherwise SDBench's LLM price estimate is used. Prices are cached
per order and shared by every run and policy, so prices never drift between arms. The reviewed
prices of every order in the reported runs ship in benchmark/prices.json.gz; a new order is priced
and reviewed on first use. KeywordIndex replaces the pinned matcher's slow keyword search with
an exact, cached equivalent.
"""

from concurrent.futures import Future
import fcntl
import gzip
import hashlib
import heapq
import json
import math
import re
from threading import Lock

from fuzzywuzzy import fuzz, utils
from sdbench.costing import llm_estimater
from sdbench.costing.costing import estimate_unified_test_costs

from environment import MODEL, ROOT, read_json, write_json

SHIPPED_PRICES = ROOT / "benchmark/prices.json.gz"


def shipped_prices(path=SHIPPED_PRICES):
    """Reviewed prices by cache file stem (curation/export_benchmark.py), or {} if absent."""
    return json.loads(gzip.decompress(path.read_bytes())) if path.exists() else {}


def review_cpt(client, test, estimate):
    """Require a second opinion on whether the selected codes cover the order."""
    selection = estimate["cpt_estimate"]["cpt_selection"]
    codes = estimate["cpt_estimate"]["cpt_codes"]
    if "87591" in codes and not re.search(r"gonorr|\bGC\b", test, re.IGNORECASE):
        return False, "87591 is a gonorrhea assay and the order does not request it."
    payload = {"test": test, "codes": codes,
               "descriptions": [r["procedure_name"] for r in selection["selected_results"]],
               "matching_rationale": selection["reasoning"]}
    instructions = (
        "Audit whether these US billing codes cover the requested medical test. "
        "Reject unrelated, overbroad, or partial matches. Interpret the known meaning "
        "of each code independently of the hospital's shorthand description. "
        "A genetic panel and a pathogen DNA assay are different tests. Reject when "
        "uncertain, including a generic code used only because no suitable code was found. "
        "Reply with exactly APPROVE or REJECT."
    )
    for _ in range(2):
        response = client.responses.create(
            model=MODEL, reasoning={"effort": "none"}, max_output_tokens=80, store=False,
            input=[{"role": "developer", "content": instructions},
                   {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        )
        verdict = (response.output_text or "").strip().upper()
        if verdict in {"APPROVE", "REJECT"}:
            return verdict == "APPROVE", "Independent CPT review: " + verdict.lower()
    raise ValueError(f"Could not verify billing codes for {test!r}")


class GroundedPrices:
    def __init__(self, config, cache_dir, previous_cache, client, shipped=None):
        self.config = config
        self.shipped = shipped or {}
        self.cache_dir = cache_dir
        self.previous_cache = previous_cache
        self.client = client
        self.pending = {}
        self.lock = Lock()

    def __call__(self, test):
        with self.lock:
            existing = self.pending.get(test)
            if existing is None:
                request = self.pending[test] = Future()
        if existing is not None:
            return existing.result()
        try:
            filename = hashlib.sha256(test.encode()).hexdigest() + ".json"
            path = self.cache_dir / filename
            # Both held-out splits can use the same reviewed price cache concurrently.
            locks = self.cache_dir / "locks"
            locks.mkdir(parents=True, exist_ok=True)
            with (locks / (path.stem + ".lock")).open("a") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                if path.exists():
                    result = read_json(path)
                elif path.stem in self.shipped:
                    result = self.shipped[path.stem]
                else:
                    result = self.verified_price(test, filename)
                if result["cost"] is None or not math.isfinite(result["cost"]) or result["cost"] <= 0:
                    raise ValueError(f"SDBench could not price {test!r}: {result['explanation']}")
                if not path.exists():
                    write_json(path, result)
        except Exception as error:
            request.set_exception(error)
            with self.lock:
                del self.pending[test]
            raise
        request.set_result(result)
        return result

    def verified_price(self, test, filename):
        previous = self.previous_cache / filename
        original = read_json(previous) if previous.exists() else estimate_unified_test_costs(test, self.config)
        if original["method"] != "cpt_lookup":
            return original
        accepted, note = review_cpt(self.client, test, original)
        if accepted:
            return {**original, "billing_review": note}
        fallback = llm_estimater.cost_test(test, self.config)
        return {**original, "method": "llm_fallback" if "cost" in fallback else "failure",
                "cost": fallback.get("cost"), "llm": fallback,
                "billing_review": note, "explanation": fallback.get("error", note)}


class KeywordIndex:
    """Reuse normalized procedure text and skip partial scores that cannot change the top k."""

    def __init__(self, charges, clean_price):
        rows = charges[["Procedure Code (CPT / HCPCS)", "Procedure Name", "Gross Charge",
                        "Hospital Inpatient / Outpatient / Both"]].itertuples(index=False, name=None)
        self.rows = []
        for code, name, price, setting in rows:
            name = str(name)
            lower = name.lower()
            normalized = utils.full_process(lower, force_ascii=True)
            self.rows.append((str(code), name, clean_price(price), str(setting), lower,
                              set(re.findall(r"\b\w+\b", lower)), normalized,
                              " ".join(sorted(normalized.split()))))

    def search(self, query, top_k=15):
        if top_k <= 0:
            return []
        query = query.lower()
        query_words = set(re.findall(r"\b\w+\b", query))
        normalized = utils.full_process(query, force_ascii=True)
        sorted_query = " ".join(sorted(normalized.split()))
        candidates = []
        for i, (_, _, _, _, lower, words, normalized_name, sorted_name) in enumerate(self.rows):
            ratio = fuzz.ratio(query, lower)
            token_sort = fuzz.ratio(sorted_query, sorted_name)
            token_set = (fuzz.token_set_ratio(normalized, normalized_name, full_process=False)
                         if normalized and normalized_name else 0)
            contains = query in lower
            overlap = len(query_words & words) / len(query_words) if query_words else 0
            # The exact score can only decrease when the partial ratio replaces 100.
            upper = (ratio * 0.2 + 100 * 0.2 + token_sort * 0.2 + token_set * 0.2
                     + (100 if contains else 0) * 0.1 + overlap * 100 * 0.1)
            if upper > 30:
                candidates.append((upper, i, ratio, token_sort, token_set, contains, overlap))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)

        best = []
        for upper, i, ratio, token_sort, token_set, contains, overlap in candidates:
            if len(best) == top_k and upper < best[0][0]:
                break
            code, name, cost, setting, lower, *_ = self.rows[i]
            partial = fuzz.partial_ratio(query, lower)
            score = (ratio * 0.2 + partial * 0.2 + token_sort * 0.2 + token_set * 0.2
                     + (100 if contains else 0) * 0.1 + overlap * 100 * 0.1)
            if score <= 30:
                continue
            ranked = (score, -i, {"cpt_code": code, "procedure_name": name, "gross_charge": cost,
                                  "setting": setting, "keyword_score": score, "fuzzy_ratio": ratio,
                                  "partial_ratio": partial, "token_sort_ratio": token_sort,
                                  "token_set_ratio": token_set, "contains_match": contains,
                                  "word_overlap": overlap})
            if len(best) < top_k:
                heapq.heappush(best, ranked)
            elif ranked[:2] > best[0][:2]:
                heapq.heapreplace(best, ranked)
        return [result for _, _, result in sorted(best, reverse=True)]
