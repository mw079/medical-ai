"""Check the sequential panel team's role steps with a fake model client."""

import json
from types import SimpleNamespace
import unittest

from agents.panel_team import PanelTeam

OPENING = [{"role": "system", "content": "SDBench"}, {"role": "user", "content": "Fever and cough."}]
ASKED = OPENING + [{"role": "assistant", "content": "<question>Any sputum?</question>"},
                   {"role": "user", "content": "Results: yellow sputum"}]
DX = "<diagnosis>Pneumonia</diagnosis>"


class FakeClient:
    def __init__(self, orchestrator, challenge=None, keep=None, tests=("Chest X-ray", "Procalcitonin", "Chest CT"),
                 differential=("Pneumonia", "Lung abscess"), reconciled="Lung abscess"):
        self.orchestrator, self.challenge = list(orchestrator), challenge or {"object": False}
        self.keep, self.tests, self.differential, self.reconciled = keep, list(tests), differential, reconciled
        self.prompts, self.responses = [], self

    def answer(self, developer):
        if "You are Dr. Experience" in developer:
            return {"applies": True, "hint": "A similar case was settled by a chest CT."}
        if "You are Dr. Hypothesis" in developer:
            return {"differential": [{"diagnosis": d, "probability": 0.4} for d in self.differential]}
        if "You are Dr. Test-Chooser" in developer:
            return {"questions": ["Any sputum?", "Any night sweats?"], "tests": self.tests}
        if "You are Dr. Stewardship" in developer:
            return {"keep": self.keep if self.keep is not None else self.tests}
        if "You are Dr. Challenger" in developer:
            return self.challenge
        if "weigh every diagnosis considered" in developer:
            return {"diagnosis": self.reconciled}
        if "You lead the diagnostic panel" in developer:
            return {"action": self.orchestrator.pop(0)}
        raise AssertionError("unexpected prompt")

    def create(self, **request):
        developer = request["input"][0]["content"]
        self.prompts.append(developer)
        return SimpleNamespace(model=request["model"], id="r", status="completed", output_text=json.dumps(self.answer(developer)),
                               service_tier=None, usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5}))


def context(**changes):
    prices = {"Chest X-ray": 100, "Procalcitonin": 80, "Chest CT": 900}
    base = {"mode": "budget_1000", "cap": 1000, "remaining": 700, "visit_cost": 300, "turns_used": 1, "max_turns": 10,
            "undocumented_tests": ["Procalcitonin"], "ordered_tests": [], "scratch": {},
            "price_tests": lambda names: [{"cost": prices[n]} for n in names]}
    return {**base, **changes}


class PanelTeamTests(unittest.TestCase):
    def test_roles_feed_the_orchestrator_and_code_enforces_dead_ends_and_budget(self):
        client = FakeClient(["<test>Chest X-ray</test>"])
        turn = PanelTeam(client, "gpt-6-luna")(ASKED, context=context())
        proposals = turn["proposals"]
        self.assertEqual(proposals["tests"], ["Chest X-ray", "Chest CT"])  # Procalcitonin was not documented
        self.assertEqual(proposals["approved_tests"], ["Chest X-ray"])   # CT does not fit $700 with the visit
        self.assertEqual(proposals["questions"], ["Any night sweats?"])  # already asked about sputum
        self.assertEqual(turn["text"], "<test>Chest X-ray</test>")
        roles = [c["role"] for c in turn["agent_calls"]]
        self.assertEqual(roles, ["dr_hypothesis", "dr_test_chooser", "dr_stewardship", "orchestrator", "dr_challenger"])

    def test_challenger_objection_to_an_early_diagnosis_forces_more_evidence(self):
        client = FakeClient([DX, "<test>Chest X-ray</test>"],
                            challenge={"object": True, "reason": "No imaging yet.", "better": "Order a chest X-ray."})
        turn = PanelTeam(client, "gpt-6-luna")(ASKED, context=context())
        self.assertEqual((turn["first_action"], turn["text"]), (DX, "<test>Chest X-ray</test>"))
        self.assertIn("No imaging yet.", client.prompts[-1])

    def test_differential_persists_and_the_final_turn_reconciles_all_candidates(self):
        ctx = context()
        team = PanelTeam(FakeClient(["<test>Chest X-ray</test>"], differential=("Pneumonia", "Lung abscess")), "gpt-6-luna")
        team(ASKED, context=ctx)
        team.client = FakeClient([], differential=("Pneumonia", "Empyema"), reconciled="Lung abscess")
        final = ASKED + [{"role": "user", "content": "No more tests available. Diagnose."}]
        turn = team(final, context=ctx)
        self.assertEqual(ctx["scratch"]["considered"], ["Pneumonia", "Lung abscess", "Empyema"])
        self.assertEqual(turn["text"], "<diagnosis>Lung abscess</diagnosis>")
        self.assertIn("Your previous differential", team.client.prompts[0])
        self.assertIn("Lung abscess", team.client.prompts[0])


class ChecklistTests(unittest.TestCase):
    def test_an_action_that_only_repeats_ordered_tests_is_replaced(self):
        client = FakeClient(["<test>Chest X-ray</test>", "<test>Sputum culture</test>"], tests=("Chest CT",))
        turn = PanelTeam(client, "gpt-6-luna")(ASKED, context=context(ordered_tests=["Chest X-ray"]))
        self.assertEqual(turn["text"], "<test>Sputum culture</test>")
        self.assertEqual(turn["checklist"], {"repeated": ["Chest X-ray"]})
        self.assertIn("must not be repeated: Chest X-ray", client.prompts[-1])


class FakeMemory:
    def retrieve(self, query, k=3):
        return [{"case_id": cid, "similarity": 0.5, "presentation": f"past {cid}", "diagnosis": f"dx {cid}",
                 "decisive_evidence": ["a test"], "missteps": [], "strategy": "a lesson"}
                for cid in ("PMC1", "PMC2", "PMC3", "PMC4")][:k]


class ExperienceTests(unittest.TestCase):
    def test_cards_stay_with_dr_experience_and_only_the_hint_reaches_the_others(self):
        client = FakeClient(["<test>Chest X-ray</test>"])
        team = PanelTeam(client, "gpt-6-luna", memory=FakeMemory(), playbook=["Do not stop at a syndrome."])
        turn = team(ASKED, context=context(case_id="PMC1"))
        self.assertEqual([c["case_id"] for c in turn["retrieved"]], ["PMC2", "PMC3", "PMC4"])
        experience, *others = client.prompts
        self.assertIn("You are Dr. Experience", experience)
        self.assertIn("past PMC2", experience)
        self.assertNotIn("past PMC1", experience)
        for prompt in others:
            self.assertNotIn("past PMC2", prompt)
        orchestrator = next(p for p in others if "You lead the diagnostic panel" in p)
        self.assertIn("Dr. Experience: A similar case was settled by a chest CT.", orchestrator)
        self.assertIn("- Do not stop at a syndrome.", orchestrator)

    def test_without_memory_there_is_no_experience_call(self):
        client = FakeClient(["<test>Chest X-ray</test>"])
        PanelTeam(client, "gpt-6-luna")(ASKED, context=context())
        self.assertFalse(any("Dr. Experience" in p for p in client.prompts))


if __name__ == "__main__":
    unittest.main()
