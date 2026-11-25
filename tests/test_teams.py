"""Check the team designs' selection, voting and logging with a fake model client."""

import json
from threading import Lock
from types import SimpleNamespace
import unittest

from agents import make_agent
from agents.teams import CentralizedTeam, DecentralizedTeam, IndependentTeam
from compare_repeated import agent_usage

MESSAGES = [{"role": "system", "content": "SDBench system prompt"},
            {"role": "user", "content": "Patient presentation: fever and cough."}]
CBC, XRAY, DX = "<test>CBC</test>", "<test>Chest X-ray</test>", "<diagnosis>Pneumonia</diagnosis>"


class FakeClient:
    """Answers by prompt kind. Proposals and votes are handed out in order from queues."""

    def __init__(self, proposals=(), votes=(), referee=1, orchestrator=XRAY, note=CBC):
        self.proposals, self.votes = list(proposals), list(votes)
        self.referee, self.orchestrator, self.note = referee, orchestrator, note
        self.lock, self.calls = Lock(), []
        self.responses = self

    def answer(self, developer):
        if "You are the referee" in developer:
            return json.dumps({"choice": self.referee})
        if "Vote for the proposal" in developer:
            return self.votes.pop(0)
        if "You lead a panel" in developer:
            return json.dumps({"action": self.orchestrator})
        if "on a diagnostic panel" in developer:
            return json.dumps({"assessment": "Brief view.", "action": self.note})
        return self.proposals.pop(0)

    def create(self, **request):
        with self.lock:
            developer = request["input"][0]["content"]
            text = self.answer(developer)
            self.calls.append(developer)
        usage = {"input_tokens": 10, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 2}}
        return SimpleNamespace(model=request["model"], id="r", status="completed", output_text=text,
                               service_tier="default", usage=SimpleNamespace(model_dump=lambda: usage))


def action(text, reason="because"):
    return json.dumps({"action": text, "reason": reason})


class IndependentTests(unittest.TestCase):
    def test_identical_proposals_skip_the_referee(self):
        client = FakeClient(proposals=[action(CBC)] * 3)
        turn = IndependentTeam(client, "gpt-6-luna", 3)(MESSAGES)
        self.assertEqual((turn["text"], turn["referee"]), (CBC, "unanimous"))
        self.assertEqual(len(client.calls), 3)

    def test_referee_picks_a_whole_proposal_by_number(self):
        client = FakeClient(proposals=[action(CBC), action(XRAY), action(CBC)], referee=2)
        turn = IndependentTeam(client, "gpt-6-luna", 3)(MESSAGES)
        self.assertEqual(turn["text"], turn["proposals"][1]["action"])
        self.assertEqual(turn["choice"], 2)
        self.assertEqual([call["role"] for call in turn["agent_calls"]][-1], "referee")

    def test_invalid_proposal_is_retried_once_then_dropped(self):
        # Agents answer in parallel: the first two calls are valid, the third agent fails twice.
        client = FakeClient(proposals=[action(CBC), action(XRAY), "not json", "still not json"], referee=1)
        turn = IndependentTeam(client, "gpt-6-luna", 3)(MESSAGES)
        self.assertEqual(len(turn["proposals"]), 2)
        self.assertEqual(agent_usage([{"role": "diagnostic", **turn}])["calls"], 5)  # 4 proposal calls + referee

    def test_final_turn_accepts_only_diagnoses(self):
        client = FakeClient(proposals=[action(CBC), action(CBC), action(DX), action(DX)], referee=1)
        turn = IndependentTeam(client, "gpt-6-luna", 2)(MESSAGES, final=True)
        self.assertEqual(turn["text"], DX)


class DecentralizedTests(unittest.TestCase):
    def test_majority_vote_wins(self):
        client = FakeClient(proposals=[action(CBC), action(XRAY), action(DX)],
                            votes=[json.dumps({"vote": 2}), json.dumps({"vote": 2}), json.dumps({"vote": 1})])
        turn = DecentralizedTeam(client, "gpt-6-luna", 3)(MESSAGES)
        self.assertEqual(turn["text"], turn["proposals"][1]["action"])
        self.assertEqual(sorted(turn["votes"]), [1, 2, 2])
        self.assertIn("reason", turn["proposals"][0])

    def test_tie_and_invalid_votes_go_to_the_lowest_number(self):
        client = FakeClient(proposals=[action(CBC), action(XRAY), action(DX)],
                            votes=[json.dumps({"vote": 3}), json.dumps({"vote": 2}), "x", "x"])
        turn = DecentralizedTeam(client, "gpt-6-luna", 3)(MESSAGES)
        self.assertEqual(turn["choice"], 2)
        self.assertIn(None, turn["votes"])

    def test_out_of_range_vote_is_not_counted(self):
        client = FakeClient(proposals=[action(CBC), action(XRAY)],
                            votes=[json.dumps({"vote": 2}), json.dumps({"vote": 7}), json.dumps({"vote": 9})])
        turn = DecentralizedTeam(client, "gpt-6-luna", 2)(MESSAGES)
        self.assertEqual(turn["choice"], 2)


class CentralizedTests(unittest.TestCase):
    def test_orchestrator_may_choose_a_new_request(self):
        client = FakeClient(note=CBC, orchestrator=XRAY)
        turn = CentralizedTeam(client, "gpt-6-luna", 3)(MESSAGES)
        self.assertEqual(turn["text"], XRAY)
        self.assertEqual([n["role"] for n in turn["notes"]], ["Dr. Hypothesis", "Dr. Test-Chooser", "Dr. Challenger"])
        self.assertEqual(agent_usage([{"role": "diagnostic", **turn}])["calls"], 4)

    def test_five_agents_add_stewardship_and_checklist(self):
        client = FakeClient()
        turn = CentralizedTeam(client, "gpt-6-luna", 5)(MESSAGES)
        self.assertEqual([n["role"] for n in turn["notes"]][3:], ["Dr. Stewardship", "Dr. Checklist"])

    def test_unsupported_team_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            CentralizedTeam(FakeClient(), "gpt-6-luna", 4)
        with self.assertRaises(ValueError):
            IndependentTeam(FakeClient(), "gpt-6-luna", 1)
        with self.assertRaises(KeyError):
            make_agent("hybrid", "gpt-6-luna", 3)


if __name__ == "__main__":
    unittest.main()
