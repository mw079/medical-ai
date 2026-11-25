"""Sequential panel team: MAI-DxO's roles as separate GPT-6 Luna agents with distinct jobs.

The first Centralized team failed because every role solved the whole case: all five read the
full transcript, each recommended an action, and they recommended the same one. Here each role has
its own output, the roles run in order so later ones react to earlier ones, and code supplies
facts the model would otherwise guess (prices, dead ends, budget and turn limits).

One turn:
0. Dr. Experience (memory variant only) reads the retrieved past cases in its own context and
   passes at most a two-sentence hint; the playbook rules go to the orchestrator and Challenger.
1. Dr. Hypothesis updates a ranked differential that is kept across turns.
2. Dr. Test-Chooser proposes questions and tests that separate the leading diagnoses.
3. Dr. Checklist (code) drops requests already ordered or answered "not documented".
4. Dr. Stewardship sees real chargemaster prices and keeps only tests worth their cost.
5. The orchestrator drafts one action from these outputs.
6. Dr. Challenger attacks the draft; on an objection the orchestrator revises once.
7. A diagnosis is confirmed by reconciliation over every diagnosis considered in the encounter,
   and code decides whether the Challenger may still block it (budget and turns left).
"""

import json
import re

from .base import PROMPT, action_from_response, conversation_input, is_final, parse_object, request
from .evidence import GATE_INTRO, EvidenceAgent, kind, state_text

EXPERIENCE = """
You are Dr. Experience on a diagnostic panel. You see lessons from past patients; the others do
not. Past cases (other patients):
{cards}
Say whether any lesson applies to this patient. If one does, give a hint of at most two sentences:
which diagnosis to consider or which question or test settled a similar case. If none applies,
say so; an irrelevant hint misleads the panel.
Return only {{"applies": true or false, "hint": "..."}}."""
HYPOTHESIS = """
You are Dr. Hypothesis on a diagnostic panel. Case state: {state}
Your previous differential: {previous}{hint}
Update it with the evidence in the conversation: keep diagnoses that still fit, drop those the
evidence rules out, add new ones. Rank at most 5, with probabilities, and for each name the
finding that supports it most and the finding that argues against it most.
Do not choose an action. Return only {{"differential": [{{"diagnosis": "...", "probability": 0.0,
"supports": "...", "against": "..."}}]}}."""
TEST_CHOOSER = """
You are Dr. Test-Chooser on a diagnostic panel. Case state: {state}
Leading diagnoses: {leading}{hint}
Propose what best separates the top two: up to 5 history or examination questions, and up to 4
diagnostic tests, each a single specific, orderable item. Do not propose anything already reported
as not documented. Do not choose an action. Return only {{"questions": ["..."], "tests": ["..."],
"reason": "..."}}."""
STEWARDSHIP = """
You are Dr. Stewardship on a diagnostic panel. Case state: {state}
Leading diagnoses: {leading}
Proposed tests with their real prices: {priced}
Keep only tests whose result could change the leading diagnosis enough to justify the cost.
Return only {{"keep": ["exact test names from the list"], "reason": "..."}}."""
ORCHESTRATOR = """
You lead the diagnostic panel. Case state: {state}
Differential (Dr. Hypothesis): {differential}
Questions proposed (Dr. Test-Chooser): {questions}
Tests approved for cost (Dr. Stewardship): {tests}{hint}{playbook}
Choose one action: ask questions, order tests, or give a single final diagnosis. Prefer the
panel's proposals. Diagnose only when the leading diagnosis is clearly established."""
CHALLENGER = """
You are Dr. Challenger on a diagnostic panel. Case state: {state}
Differential: {differential}
The panel's draft action: {draft}{playbook}
Attack the draft: is it anchored on one diagnosis, missing a dangerous alternative, wasting budget,
or stopping before decisive evidence? Object only for a concrete reason.
Return only {{"object": true or false, "reason": "...", "better": "what to do instead"}}."""
REPEATS = ("\nThese requests were already made or reported as not documented and must not be repeated: "
           "{repeats}. Choose different questions or tests, or diagnose.")
REVISE = "\nDr. Challenger objected to your draft ({draft}): {reason} Suggested instead: {better} Revise your action."


def normalize(name):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", name.casefold()).split())


class PanelTeam(EvidenceAgent):
    name = "panel_team"

    def __init__(self, client, model, agents=5, *, memory=None, playbook=None):
        if agents != 5:
            raise ValueError("The panel team has five roles")
        super().__init__(client, model, 1, gate=True, reconcile=True, memory=memory, playbook=playbook)

    def _experience(self, messages, context, calls, record):
        """Dr. Experience: retrieved cards stay in this role's context; only a short hint leaves it."""
        if self.memory is None:
            return ""
        from memory import format_cards
        from .evidence import UNAVAILABLE
        lines = [line for m in messages[1:] if m["role"] == "user" for line in m["content"].splitlines()
                 if line.strip() and UNAVAILABLE not in line]
        own = context.get("case_id")
        cards = [c for c in self.memory.retrieve("\n".join(lines)[:2000], k=4) if c["case_id"] != own][:3]
        record["retrieved"] = [{"case_id": c["case_id"], "similarity": c["similarity"]} for c in cards]

        def parse(text):
            data = parse_object(text)
            hint = str(data.get("hint", "")).strip()
            if len(hint) > 400 or "\n" in hint:
                raise ValueError("The hint must be at most two short sentences")
            return {"applies": bool(data.get("applies")), "hint": hint}
        advice = self._ask(EXPERIENCE.format(cards=format_cards(cards)), messages, parse, "dr_experience", calls)
        record["experience"] = advice
        return f"\nDr. Experience: {advice['hint']}" if advice and advice["applies"] and advice["hint"] else ""

    def _ask(self, prompt, messages, parse, role, calls):
        value, more = request(self.client, self.model, GATE_INTRO + prompt, conversation_input(messages),
                              parse, role=role)
        calls += more
        return value

    def __call__(self, messages, *, instruction="", final=False, context=None):
        final = is_final(messages, final)
        context = context if context is not None else {}
        scratch = context.setdefault("scratch", {})
        state = state_text(context, messages)
        calls, record = [], {"model": self.model, "design": self.name, "state": state, "gate": None}
        hint = self._experience(messages, context, calls, record)
        playbook = ("\nPlaybook from past cases:\n" + "\n".join(f"- {r}" for r in self.playbook)) if self.playbook else ""

        # 1. Dr. Hypothesis: a differential kept across turns.
        def parse_differential(text):
            items = parse_object(text)["differential"]
            if not isinstance(items, list) or not items:
                raise ValueError("empty differential")
            return [i for i in items if isinstance(i, dict) and str(i.get("diagnosis", "")).strip()][:5]
        previous = scratch.get("differential", [])
        differential = self._ask(HYPOTHESIS.format(state=state, previous=json.dumps(previous) if previous else "none yet",
                                                   hint=hint),
                                 messages, parse_differential, "dr_hypothesis", calls) or previous
        scratch["differential"] = differential
        considered = scratch.setdefault("considered", [])
        for item in differential:
            name = str(item["diagnosis"]).strip()
            if "\n" not in name and name.casefold() not in {n.casefold() for n in considered}:
                considered.append(name)
        leading = "; ".join(f"{i['diagnosis']} ({i.get('probability', '?')})" for i in differential[:3]) or "none"
        record["differential"] = differential

        if final:  # forced diagnosis: confirm the best supported candidate directly
            action = f"<diagnosis>{differential[0]['diagnosis'] if differential else 'Unknown'}</diagnosis>"
            return self._finish(record, action, calls, messages, context, playbook)

        # 2. Dr. Test-Chooser: discriminating requests.
        def parse_requests(text):
            data = parse_object(text)
            def clean(xs):
                return [str(x).strip() for x in xs if str(x).strip() and "\n" not in str(x) and "<" not in str(x)]
            return {"questions": clean(data.get("questions", []))[:5], "tests": clean(data.get("tests", []))[:4]}
        proposed = self._ask(TEST_CHOOSER.format(state=state, leading=leading, hint=hint), messages, parse_requests,
                             "dr_test_chooser", calls) or {"questions": [], "tests": []}

        # 3. Dr. Checklist (code): drop dead ends and repeats before anything is priced.
        dead = {normalize(t) for t in context.get("undocumented_tests", []) + context.get("ordered_tests", [])}
        asked = {normalize(q) for m in messages if m["role"] == "assistant"
                 for q in re.findall(r"<question>(.*?)</question>", m["content"], re.S)}
        tests = [t for t in proposed["tests"] if normalize(t) not in dead]
        questions = [q for q in proposed["questions"] if normalize(q) not in asked]

        # 4. Dr. Stewardship: real prices, keep tests worth their cost.
        approved = []
        if tests and context.get("price_tests"):
            prices = context["price_tests"](tests)
            priced = "; ".join(f"{t}: ${p['cost']:,.0f}" for t, p in zip(tests, prices))
            keep = self._ask(STEWARDSHIP.format(state=state, leading=leading, priced=priced), messages,
                             lambda text: [str(x) for x in parse_object(text)["keep"]], "dr_stewardship", calls)
            approved = [t for t in tests if keep is None or t in keep]
            remaining = context.get("remaining")
            if remaining is not None:  # code keeps the batch affordable: drop the most expensive first
                cost = dict(zip(tests, (p["cost"] for p in prices)))
                while approved and context.get("visit_cost", 300) + sum(cost[t] for t in approved) > remaining:
                    approved.remove(max(approved, key=cost.get))
        elif tests:
            approved = tests
        record["proposals"] = {"questions": questions, "tests": tests, "approved_tests": approved}

        # 5. Orchestrator draft, 6. Dr. Challenger, one revision on objection.
        lead = ORCHESTRATOR.format(state=state, differential=leading, questions="; ".join(questions) or "none",
                                   tests="; ".join(approved) or "none", hint=hint, playbook=playbook)
        draft = self._ask_action(messages, instruction + lead, final, calls)
        blockable = self.can_block(context, messages)

        def parse_challenge(text):
            data = parse_object(text)
            return {"object": bool(data.get("object")), "reason": str(data.get("reason", "")).strip(),
                    "better": str(data.get("better", "")).strip()}
        challenge = self._ask(CHALLENGER.format(state=state, differential=leading, draft=draft, playbook=playbook), messages,
                              parse_challenge, "dr_challenger", calls)
        record["challenge"] = challenge
        action = draft
        if challenge and challenge["object"] and (kind(draft) != "diagnosis" or blockable):
            revised = self._ask_action(messages, instruction + lead + REVISE.format(draft=draft, **{
                k: challenge[k] for k in ("reason", "better")}), final, calls,
                forbid_diagnosis=kind(draft) == "diagnosis")
            action = revised or draft
        record["first_action"] = draft

        # Dr. Checklist (code) again, on the action itself: a revision must not repeat a request.
        def repeats(act):
            if kind(act) == "diagnosis":
                return []
            tags = re.findall(r"<(test|question)>(.*?)</\1>", act, re.S)
            return [item for tag, item in tags if normalize(item) in (dead if tag == "test" else asked)]
        if repeats(action) and len(repeats(action)) == len(re.findall(r"<(?:test|question)>", action)):
            retry = self._ask_action(messages, instruction + lead + REPEATS.format(repeats="; ".join(repeats(action))),
                                     final, calls, forbid_diagnosis=False)
            if retry and not (repeats(retry) and len(repeats(retry)) == len(re.findall(r"<(?:test|question)>", retry))):
                record["checklist"] = {"repeated": repeats(action)}
                action = retry

        # 7. Confirm a diagnosis by reconciliation over every candidate considered.
        if kind(action) == "diagnosis":
            record["gate"] = {"decision": "challenged" if action != draft else "approve"}
            return self._finish(record, action, calls, messages, context, playbook)
        return {**record, "text": action, "agent_calls": calls}

    def _ask_action(self, messages, instruction, final, calls, *, forbid_diagnosis=False):
        def parse(text):
            action = action_from_response(text, final=final)
            if forbid_diagnosis and kind(action) == "diagnosis":
                raise ValueError("A diagnosis was not allowed this time")
            return action
        value, more = request(self.client, self.model, PROMPT + instruction, conversation_input(messages),
                              parse, role="orchestrator")
        calls += more
        if value is None and not forbid_diagnosis:
            raise ValueError("The orchestrator gave no valid action after two attempts")
        return value
