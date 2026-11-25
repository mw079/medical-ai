"""Budget modes and their enforcement around each diagnostic turn.

A test batch must fit the remaining budget at full price before the gatekeeper sees it; blocked
batches go back to the agent with the prices, and after three blocks the agent must diagnose.
Tests answered "not documented" are refunded afterwards. The final bill is recomputed from the
SDBench trace and must match what was approved.
"""

from collections import Counter
import math

from sdbench.agents.baseline import extract_test_list_from_response

from environment import UNAVAILABLE
from policy import get_policy

MODES = {"instant": 0, "budget_1000": 1000, "budget_4000": 4000, "no_budget": None}
TEST_PREFIX = "Test to order: "


class MedicalPolicy:
    """Enforce one mode's budget and rules around each diagnostic turn."""

    def __init__(self, mode, price_batch, diagnose, playbook=None, policy=None, case_id=None):
        self.mode = mode
        self.cap = MODES[mode]
        self.price_batch = price_batch
        self.diagnose = diagnose
        self.policy = policy or get_policy()
        self.visit_cost = self.policy.visit_cost
        self.visit_spend = 0 if mode == "instant" else self.visit_cost
        self.tests = {}
        self.rejected = []
        self.playbook = playbook or {}
        self.evidence = []  # filled by the gatekeeper: {"request", "result"}
        self.case_id = case_id
        self.scratch = {}  # an agent's own notes for this encounter, kept across turns

    def undocumented(self):
        """Ordered tests whose every reply so far was 'not documented'. These are not charged."""
        replies = {}
        for entry in self.evidence:
            if entry["request"].startswith(TEST_PREFIX):
                name = entry["request"][len(TEST_PREFIX):]
                replies.setdefault(name, []).append(entry["result"])
        return {name for name, results in replies.items() if all(r == UNAVAILABLE for r in results)}

    @property
    def spent(self):
        free = self.undocumented()
        return self.visit_spend + sum(info["cost"] for name, info in self.tests.items() if name not in free)

    def __call__(self, messages):
        instruction = (
            "\nUse questions only for patient history or routine physical examination. "
            "Use a separate <test> tag for each diagnostic procedure."
        )
        final = self.mode == "instant"
        if final:
            instruction += "\nGive one final diagnosis based on the opening alone."
        elif self.cap is not None:
            instruction += (
                f"\nThe estimated medical budget has ${self.cap - self.spent:,.2f} remaining. "
                f"Each test batch includes a ${self.visit_cost} follow-up visit plus the test prices. "
                "A batch must fit at full price; tests reported as not documented are refunded afterward. "
                "Spend only on requests likely to change the diagnosis."
            )
        stage = "opening" if len(messages) == 2 else "follow_up"
        if guidance := self.playbook.get(stage):
            instruction += "\nClinical decision guidance: " + guidance
        # Facts an agent may use for its own control logic; the instruction text stays unchanged.
        context = {"case_id": self.case_id, "mode": self.mode, "cap": self.cap, "visit_cost": self.visit_cost,
                   "remaining": None if self.cap is None else self.cap - self.spent,
                   "turns_used": sum(m.get("role") == "assistant" for m in messages),
                   "max_turns": self.policy.max_turns,
                   "undocumented_tests": sorted(self.undocumented()), "scratch": self.scratch,
                   # Real chargemaster prices for candidate tests, for agents that weigh cost.
                   "price_tests": self.price_batch, "ordered_tests": sorted(self.tests)}
        retries = []
        feedback = ""
        for _ in range(3):
            call = self.diagnose(messages, instruction=instruction + feedback, final=final, context=context)
            tests = [item[len(TEST_PREFIX):] for item in
                     extract_test_list_from_response(call["text"]) if item.startswith(TEST_PREFIX)]
            if not tests or self.mode == "no_budget":
                break
            names = [name for name in dict.fromkeys(tests[:self.policy.max_tests_per_turn])
                     if name not in self.tests]
            new_tests = dict(zip(names, self.price_batch(names), strict=True))
            charge = self.visit_cost + sum(item["cost"] for item in new_tests.values())
            if self.spent + charge <= self.cap:
                self.visit_spend += self.visit_cost
                self.tests.update(new_tests)
                break
            prices = ", ".join(f"{name}: ${info['cost']:,.2f}" for name, info in new_tests.items())
            feedback = (
                f"\nThe requested batch was blocked. It would add ${charge:,.2f}. "
                f"New test prices: {prices}. Choose a cheaper test, ask a question, or diagnose."
            )
            self.rejected.append({"tests": tests, "reason": feedback.strip()})
            retries.extend(call.get("previous_attempts", []))
            retries.append({key: value for key, value in call.items() if key != "previous_attempts"})
        else:
            call = self.diagnose(messages, instruction=instruction + "\nGive your final diagnosis now.", final=True,
                                 context=context)
        return {**call, "previous_attempts": retries + call.get("previous_attempts", [])}

    def total(self, result):
        tests = list(dict.fromkeys(t.removeprefix(TEST_PREFIX) for t in result["requested_tests_only"]))
        if self.mode == "instant" and tests:
            raise RuntimeError(f"Restricted mode issued a test: {self.mode}")
        visits = 0 if self.mode == "instant" else result["num_test_rounds"] + 1
        items = [*self.price_batch(tests)]
        replies = {}
        for log in result["test_logs"]:
            if log["test_name"].startswith(TEST_PREFIX):
                replies.setdefault(log["test_name"][len(TEST_PREFIX):], []).append(log["result"])
        free = {name for name, results in replies.items() if all(r == UNAVAILABLE for r in results)}
        test_cost = sum(item["cost"] for name, item in zip(tests, items, strict=True) if name not in free)
        total = round(visit_cost := visits * self.visit_cost, 2) + test_cost
        if self.cap is not None and (not math.isclose(total, self.spent) or total > self.cap):
            raise RuntimeError("Final SDBench prices disagree with the enforced budget")
        methods = Counter(item["method"] for item in items)
        return {"visits": visits, "visit_cost_usd": visit_cost, "test_cost_usd": round(test_cost, 2),
                "total_usd": round(total, 2), "methods": dict(methods), "tests": items,
                "uncharged_undocumented_tests": sorted(free & set(tests)),
                "unavailable_orders": sum(log["test_name"].startswith(TEST_PREFIX)
                                          and log["result"] == UNAVAILABLE for log in result["test_logs"]),
                "blocked_batches": self.rejected}
