"""Check the evidence-first agent's control logic with a fake model client."""

import json
from types import SimpleNamespace
import unittest

from agents.evidence import EvidenceAgent, state_text

OPENING = [{"role": "system", "content": "SDBench"}, {"role": "user", "content": "Fever and cough."}]
ASKED = OPENING + [{"role": "assistant", "content": "<question>Any sputum?</question>"},
                   {"role": "user", "content": "Results: yellow sputum"}]
CONTEXT = {"mode": "budget_1000", "cap": 1000, "remaining": 700, "visit_cost": 300,
           "turns_used": 1, "max_turns": 10, "undocumented_tests": ["Procalcitonin"]}
DX = "<diagnosis>Pneumonia</diagnosis>"


class FakeClient:
    def __init__(self, agent_actions, gate=None, reconciled=None, differential=("Pneumonia",)):
        self.agent_actions, self.gate, self.prompts = list(agent_actions), gate, []
        self.reconciled, self.differential = reconciled, list(differential)
        self.responses = self

    def create(self, **request):
        developer = request["input"][0]["content"]
        self.prompts.append(developer)
        if "You review a proposed final diagnosis" in developer:
            text = json.dumps(self.gate)
        elif "weigh every diagnosis considered" in developer:
            text = json.dumps({"assessments": [], "diagnosis": self.reconciled})
        else:
            text = json.dumps({"differential": [{"diagnosis": d, "probability": 0.3} for d in self.differential],
                               "action": self.agent_actions.pop(0)})
        usage = {"input_tokens": 10, "output_tokens": 5}
        return SimpleNamespace(model=request["model"], id="r", status="completed", output_text=text,
                               service_tier=None, usage=SimpleNamespace(model_dump=lambda: usage))


def run(client, messages, context=CONTEXT, final=False):
    return EvidenceAgent(client, "gpt-6-luna")(messages, context=context, final=final)


class EvidenceAgentTests(unittest.TestCase):
    def test_non_diagnosis_actions_pass_through_with_one_call(self):
        client = FakeClient(["<test>Chest X-ray</test>"])
        turn = run(client, ASKED)
        self.assertEqual(turn["text"], "<test>Chest X-ray</test>")
        self.assertEqual(len(client.prompts), 1)
        self.assertIn("$700 of the $1,000 budget left", client.prompts[0])
        self.assertIn("not documented so far: Procalcitonin", client.prompts[0])

    def test_diagnosis_before_any_question_is_sent_back_to_ask(self):
        client = FakeClient([DX, "<question>Any chest pain?</question>"])
        turn = run(client, OPENING, context={**CONTEXT, "turns_used": 0})
        self.assertEqual((turn["text"], turn["gate"]["decision"]), ("<question>Any chest pain?</question>", "ask_first"))

    def test_gate_sends_the_agent_back_for_a_test(self):
        client = FakeClient([DX, "<test>Chest X-ray</test>"],
                            gate={"decision": "continue", "reason": "No imaging yet.", "diagnosis": "Pneumonia"})
        turn = run(client, ASKED)
        self.assertEqual(turn["text"], "<test>Chest X-ray</test>")
        self.assertIn("No imaging yet.", client.prompts[-1])

    def test_gate_approves_and_refines(self):
        client = FakeClient([DX], gate={"decision": "approve", "reason": "Confirmed.",
                                        "diagnosis": "Pneumococcal pneumonia of the right lower lobe"})
        turn = run(client, ASKED)
        self.assertEqual(turn["text"], "<diagnosis>Pneumococcal pneumonia of the right lower lobe</diagnosis>")

    def test_gate_cannot_block_without_budget_or_turns(self):
        for context in ({**CONTEXT, "remaining": 350}, {**CONTEXT, "turns_used": 9}):
            client = FakeClient([DX], gate={"decision": "continue", "reason": "More.", "diagnosis": "Pneumonia"})
            turn = run(client, ASKED, context=context)
            self.assertEqual((turn["text"], turn["gate"]["decision"]), (DX, "approve_forced"))

    def test_final_turn_is_never_blocked_and_keeps_the_agents_answer(self):
        client = FakeClient([DX], gate={"decision": "continue", "reason": "More.", "diagnosis": "Not established"})
        final = ASKED + [{"role": "user", "content": "No more tests available. Diagnose."}]
        self.assertEqual(run(client, final)["text"], DX)
        client = FakeClient([DX], gate={"decision": "approve", "reason": "Fits.", "diagnosis": "Lobar pneumonia"})
        self.assertEqual(run(client, final)["text"], "<diagnosis>Lobar pneumonia</diagnosis>")

    def test_an_uncommitted_approval_is_rejected(self):
        client = FakeClient([DX], gate={"decision": "approve", "reason": "Unclear.", "diagnosis": "Undetermined mass"})
        turn = run(client, ASKED)
        self.assertEqual((turn["text"], turn["gate"]["decision"]), (DX, "unavailable"))

    def test_state_text_without_budget(self):
        text = state_text({"mode": "no_budget", "cap": None, "remaining": None, "turns_used": 2, "max_turns": 10}, OPENING)
        self.assertEqual(text, "8 turns left. no history questions asked yet.")


class FakeMemory:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=3):
        self.queries.append(query)
        cards = [{"case_id": cid, "similarity": s, "presentation": f"past {cid}", "diagnosis": f"dx {cid}",
                  "decisive_evidence": ["a test"], "missteps": [], "strategy": "a lesson"}
                 for cid, s in (("PMC1", 0.9), ("PMC2", 0.8), ("PMC3", 0.7), ("PMC4", 0.6))]
        return cards[:k]


class MemoryTests(unittest.TestCase):
    def test_playbook_and_cards_reach_the_prompt_and_the_own_case_is_left_out(self):
        client, memory = FakeClient(["<test>Chest X-ray</test>"]), FakeMemory()
        agent = EvidenceAgent(client, "gpt-6-luna", memory=memory, playbook=["Do not stop at a syndrome."])
        messages = ASKED + [{"role": "user", "content": "Results:\n- CBC: Not documented in this encounter."}]
        turn = agent(messages, context={**CONTEXT, "case_id": "PMC1"})
        self.assertEqual([c["case_id"] for c in turn["retrieved"]], ["PMC2", "PMC3", "PMC4"])
        self.assertIn("- Do not stop at a syndrome.", client.prompts[0])
        self.assertIn("past PMC2", client.prompts[0])
        self.assertNotIn("past PMC1", client.prompts[0])
        self.assertIn("yellow sputum", memory.queries[0])
        self.assertNotIn("Not documented", memory.queries[0])

    def test_gate_prompt_asks_only_for_a_decision(self):
        client = FakeClient([DX], gate={"decision": "approve", "reason": "Fits.", "diagnosis": "Pneumonia"})
        agent = EvidenceAgent(client, "gpt-6-luna", memory=FakeMemory(), playbook=["A rule."])
        agent(ASKED, context=CONTEXT)
        gate_prompt = client.prompts[-1]
        self.assertNotIn('{"action"', gate_prompt)
        self.assertTrue(gate_prompt.rstrip().endswith('"diagnosis": "..."}.'))


class ReconcileTests(unittest.TestCase):
    def test_reconciliation_can_pick_an_earlier_candidate(self):
        context = {**CONTEXT, "scratch": {}}
        agent = EvidenceAgent(FakeClient(["<test>Chest X-ray</test>"], differential=("Pneumonia", "Lung abscess")),
                              "gpt-6-luna", reconcile=True)
        agent(ASKED, context=context)  # turn 1 records the candidates
        self.assertEqual(context["scratch"]["considered"], ["Pneumonia", "Lung abscess"])
        client = FakeClient([DX], gate={"decision": "approve", "reason": "Fits.", "diagnosis": "Pneumonia"},
                            reconciled="Lung abscess of the right lower lobe")
        agent.client = client
        turn = agent(ASKED, context=context)
        self.assertEqual(turn["text"], "<diagnosis>Lung abscess of the right lower lobe</diagnosis>")
        self.assertEqual(turn["reconcile"]["candidates"], ["Pneumonia", "Lung abscess"])
        self.assertIn("Candidates: Pneumonia; Lung abscess", client.prompts[-1])

    def test_one_candidate_needs_no_reconciliation(self):
        client = FakeClient([DX], gate={"decision": "approve", "reason": "Fits.", "diagnosis": "Pneumonia"})
        turn = EvidenceAgent(client, "gpt-6-luna", reconcile=True)(ASKED, context={**CONTEXT, "scratch": {}})
        self.assertEqual((turn["text"], len(client.prompts)), (DX, 2))  # agent + gate only

    def test_an_uncommitted_reconciliation_keeps_the_proposal(self):
        context = {**CONTEXT, "scratch": {"considered": ["Lung abscess"]}}
        client = FakeClient([DX], gate={"decision": "approve", "reason": "Fits.", "diagnosis": "Pneumonia"},
                            reconciled="Undetermined lung process")
        turn = EvidenceAgent(client, "gpt-6-luna", reconcile=True)(ASKED, context=context)
        self.assertEqual(turn["text"], DX)


if __name__ == "__main__":
    unittest.main()
