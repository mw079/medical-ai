"""Multi-agent designs on one shared encounter: Independent, Decentralized and Centralized.

Each design is called once per SDBench turn and returns one action, like the single agent, so
the budget rules, gatekeeper and judge treat every design the same way. The designs follow
Kim et al. (2025), adapted to a conversation.

- Independent: n identical agents propose in parallel. A referee picks the proposal most of
  them agree with. No referee call when all valid proposals are identical.
- Decentralized: n identical agents propose with a one-line reason, then each sees all
  proposals and votes for one. Code counts the votes; a tie goes to the lowest number.
- Centralized: n role agents (from evaluation/harnesses/panel.py, plus Checklist for n = 5) each give an
  assessment and a recommended action. An orchestrator reads them and writes the action.

Every model call is returned under "agent_calls" with its role, output and token usage.
"""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json

from .base import PROMPT, action_from_response, conversation_input, is_final, parse_object, request

DECENTRALIZED_REASON = (
    '\nAlso give a one-line reason. Return {"action": "...", "reason": "..."}.'
)
REFEREE = """
You are the referee for a team of diagnosticians who each proposed the next action independently.
Find the proposal that most of the team agrees with in substance: the same kind of action and
largely the same questions, tests or diagnosis. Do not judge which proposal is clinically best.
Do not write an action this time. Return only {"choice": <proposal number>}.
"""
VOTE = """
Your team proposed the next actions listed in TEAM PROPOSALS. Vote for the proposal most likely to
lead to the correct diagnosis within the remaining budget. You may vote for your own.
Do not write an action this time. Return only {"vote": <proposal number>}.
"""
ORCHESTRATOR = """
You lead a panel of physicians. Their assessments of this case are in PANEL NOTES. Read them, then
choose the next action yourself. You may choose questions or tests that no panelist recommended.
"""
ROLE_FORMAT = """
Give your assessment in at most 80 words, from your role only, and the action you recommend.
Return {"assessment": "...", "action": "..."}.
"""
# Worded as in evaluation/harnesses/panel.py. Checklist follows the MAI-DxO description and is used only with 5.
ROLES = {
    "Dr. Hypothesis": "List the three most likely diagnoses with rough probabilities, updated with "
                      "the evidence so far.",
    "Dr. Test-Chooser": "Pick the questions or tests that best separate the leading diagnoses.",
    "Dr. Challenger": "Name the most likely error in the current thinking (anchoring, a missed common "
                      "cause, a dangerous diagnosis not yet considered) and what finding would "
                      "disprove the leading diagnosis.",
    "Dr. Stewardship": "Check the remaining budget and prefer the cheapest request that could change "
                       "the ranking.",
    "Dr. Checklist": "Check that requested test names are specific and orderable, that nothing reported "
                     "as not documented is ordered again, and that the plan is consistent with the "
                     "evidence so far.",
}
ROLE_SETS = {3: ["Dr. Hypothesis", "Dr. Test-Chooser", "Dr. Challenger"],
             5: ["Dr. Hypothesis", "Dr. Test-Chooser", "Dr. Challenger", "Dr. Stewardship", "Dr. Checklist"]}


def _choice(key, count):
    def parse(text):
        value = parse_object(text)[key]
        if type(value) is not int or not 1 <= value <= count:
            raise ValueError(f"{key} must be a proposal number from 1 to {count}")
        return value
    return parse


class Team:
    design = ""

    def __init__(self, client, model, agents):
        if agents < 2:
            raise ValueError("A team needs at least two agents")
        self.client = client
        self.model = model
        self.agents = agents

    def _parallel(self, jobs):
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            return list(pool.map(lambda job: request(self.client, self.model, *job[0], role=job[1]), jobs))

    def _propose(self, messages, instruction, final, *, extra=""):
        """Every agent proposes once in parallel. Returns valid proposals (in agent order) and calls."""
        def parse(text):
            data = parse_object(text)
            return {"action": action_from_response(text, final=final), "reason": str(data.get("reason", ""))}
        user = conversation_input(messages)
        results = self._parallel([((PROMPT + instruction + extra, user, parse), f"agent_{i}")
                                  for i in range(1, self.agents + 1)])
        proposals, calls = [], []
        for i, (value, member_calls) in enumerate(results, 1):
            calls += member_calls
            if value is not None:
                proposals.append({"agent": i, **value})
        if not proposals:
            raise ValueError(f"No valid proposal from any of the {self.agents} agents")
        return proposals, calls

    def _result(self, action, calls, **record):
        return {"text": action, "model": self.model, "design": self.design, "agents": self.agents,
                **record, "agent_calls": calls}


def _numbered(proposals, with_reason=False):
    return "\n".join(f"{n}. {p['action']}" + (f"  (reason: {p['reason']})" if with_reason else "")
                     for n, p in enumerate(proposals, 1))


class IndependentTeam(Team):
    design = "independent"

    def __call__(self, messages, *, instruction="", final=False, context=None):
        final = is_final(messages, final)
        proposals, calls = self._propose(messages, instruction, final)
        actions = [p["action"] for p in proposals]
        if len(set(actions)) == 1:
            return self._result(actions[0], calls, proposals=proposals, choice=1, referee="unanimous")
        user = conversation_input(messages) + "\n\nTEAM PROPOSALS\n" + _numbered(proposals)
        choice, referee_calls = request(self.client, self.model, PROMPT + instruction + REFEREE, user,
                                        _choice("choice", len(proposals)), role="referee")
        calls += referee_calls
        if choice is None:
            raise ValueError("The referee gave no valid proposal number after two attempts")
        return self._result(actions[choice - 1], calls, proposals=proposals, choice=choice, referee="model")


class DecentralizedTeam(Team):
    design = "decentralized"

    def __call__(self, messages, *, instruction="", final=False, context=None):
        final = is_final(messages, final)
        proposals, calls = self._propose(messages, instruction, final, extra=DECENTRALIZED_REASON)
        user = conversation_input(messages) + "\n\nTEAM PROPOSALS\n" + _numbered(proposals, with_reason=True)
        results = self._parallel([((PROMPT + instruction + VOTE, user, _choice("vote", len(proposals))),
                                   f"voter_{i}") for i in range(1, self.agents + 1)])
        votes = []
        for vote, member_calls in results:
            calls += member_calls
            votes.append(vote)  # None means no valid vote
        counts = Counter(v for v in votes if v is not None)
        # Most votes wins; a tie (including no valid votes) goes to the lowest proposal number.
        winner = min(range(1, len(proposals) + 1), key=lambda n: (-counts[n], n))
        return self._result(proposals[winner - 1]["action"], calls, proposals=proposals,
                            votes=votes, choice=winner)


class CentralizedTeam(Team):
    design = "centralized"

    def __init__(self, client, model, agents):
        super().__init__(client, model, agents)
        if agents not in ROLE_SETS:
            raise ValueError(f"Centralized roles are defined for {sorted(ROLE_SETS)} agents")
        self.roles = ROLE_SETS[agents]

    def __call__(self, messages, *, instruction="", final=False, context=None):
        final = is_final(messages, final)

        def note(text):
            data = parse_object(text)
            return {"assessment": str(data.get("assessment", "")).strip(),
                    "action": action_from_response(text, final=final)}

        user = conversation_input(messages)
        results = self._parallel([
            ((PROMPT + instruction + f"\nYou are {role} on a diagnostic panel. {ROLES[role]}" + ROLE_FORMAT,
              user, note), role) for role in self.roles])
        notes, calls = [], []
        for role, (value, member_calls) in zip(self.roles, results):
            calls += member_calls
            if value is not None:
                notes.append({"role": role, **value})
        if not notes:
            raise ValueError("No valid assessment from any panel member")
        panel = json.dumps(notes, ensure_ascii=False, indent=1)
        action, lead_calls = request(self.client, self.model, PROMPT + instruction + ORCHESTRATOR,
                                     user + "\n\nPANEL NOTES\n" + panel,
                                     lambda text: action_from_response(text, final=final), role="orchestrator")
        calls += lead_calls
        if action is None:
            raise ValueError("The orchestrator gave no valid action after two attempts")
        return self._result(action, calls, notes=notes)
