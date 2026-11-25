"""Evidence-first single agent: case state, ask before diagnosing, and a closure gate.

One GPT-6 Luna agent acts each turn, as in the
baseline, with three additions aimed at the measured failure (stopping with evidence and budget
left):

- Case state: each turn the agent sees the budget and turns left and the tests already reported
  as not documented, and returns a ranked differential with its action.
- Ask before diagnosing: a diagnosis is refused until at least one round of history or
  examination questions has been asked (questions cost nothing).
- Closure gate: when the agent wants to diagnose while budget and turns remain, a second call
  reviews the evidence. It either sends the agent back for the most useful questions or tests, or
  approves and states the diagnosis as specifically as the evidence supports. Code decides when
  the gate may block: never on the last turn, when budget is too low for another visit, or in
  instant mode.
- Final reconciliation: before a diagnosis is committed, one call weighs every diagnosis the
  agent ranked during the encounter against all the evidence and picks the best supported. On
  development, the correct diagnosis appeared in the agent's own rankings in 15 more cases than
  it finally chose.

The agent object is shared across encounters. Per-encounter notes (the diagnoses considered) live
in context["scratch"], which MedicalPolicy keeps for one encounter.
"""

import re

from .base import PROMPT, action_from_response, conversation_input, is_final, parse_object, request

UNAVAILABLE = "Not documented in this encounter."

STATE = """
Case state: {state}
Before acting, rank the three most likely diagnoses with probabilities. Gather the evidence that
separates them: ask about history and examination findings (free) before ordering tests, and
order the test that best distinguishes the top two. Do not repeat requests reported as not
documented. Return {{"differential": [{{"diagnosis": "...", "probability": 0.0}}], "action": "..."}}."""
ASK_FIRST = ("\nYou have not asked any history or examination questions yet. They are free. Ask the "
             "questions that best separate your leading diagnoses before diagnosing.")
GATE_INTRO = """You review a diagnostic agent's work on the supplied SDBench conversation. Use only the
patient summary and the replies in that conversation; do not use tools or outside sources."""
GATE = """
You review a proposed final diagnosis before it is committed. Case state: {state}
Proposed diagnosis: {proposal}
Approve if the evidence already establishes this diagnosis (a confirmatory finding, or a picture
that no affordable question or test could change). Otherwise name the questions or test that would
best confirm it or rule out the strongest alternative.
If you approve, restate it as one committed diagnosis, as specific as the evidence supports
(cause, site, subtype), without adding anything the evidence does not show. Never answer
"undetermined", "not established" or a list of options.
Do not write an action. Return only {{"decision": "approve" or "continue", "reason": "...", "diagnosis": "..."}}."""
RECONCILE = """
Before the final diagnosis is committed, weigh every diagnosis considered during this encounter
against all the evidence in the conversation. Candidates: {candidates}
For each candidate, note the findings that support it and those it contradicts or leaves
unexplained. Then choose the candidate that best explains all the findings, and state it as one
committed diagnosis, as specific as the evidence supports (cause, site, subtype), adding nothing
the evidence does not show. Never answer "undetermined", "not established" or a list of options.
Do not write an action. Return only {{"assessments": [{{"diagnosis": "...", "supports": "...",
"against": "..."}}], "diagnosis": "..."}}."""
UNCOMMITTED = re.compile(r"\b(undetermined|not (yet )?established|unconfirmed|unknown|uncertain)\b", re.I)
CONTINUE = ("\nA reviewer judged the evidence insufficient to diagnose: {reason} Do not diagnose "
            "yet. Ask questions or order tests now.")


def kind(action):
    return re.match(r"\s*<(\w+)>", action)[1]


def state_text(context, messages):
    context = context or {}
    parts = []
    if context.get("remaining") is not None:
        parts.append(f"${context['remaining']:,.0f} of the ${context['cap']:,} budget left "
                     f"(each test round adds a ${context['visit_cost']} visit)")
    if context.get("max_turns"):
        parts.append(f"{context['max_turns'] - context['turns_used']} turns left")
    if context.get("undocumented_tests"):
        parts.append("not documented so far: " + "; ".join(context["undocumented_tests"]))
    asked = any(m["role"] == "assistant" and "<question>" in m["content"] for m in messages)
    parts.append("history questions asked" if asked else "no history questions asked yet")
    return ". ".join(parts) + "."


class EvidenceAgent:
    name = "evidence"

    def __init__(self, client, model, agents=1, *, gate=True, reconcile=False, memory=None, playbook=None):
        if agents != 1:
            raise ValueError("The evidence-first agent is a single agent")
        self.client = client
        self.model = model
        self.gate = gate
        self.reconcile = reconcile
        self.memory = memory      # ExperienceMemory: retrieved lesson cards
        self.playbook = playbook  # general rules distilled from the cards

    def experience(self, messages, context):
        """Playbook rules and the most similar past cases, as prompt text, plus the retrieved IDs."""
        if self.memory is None and not self.playbook:
            return "", []
        text, retrieved = "", []
        if self.playbook:
            text += "\nPlaybook from past cases:\n" + "\n".join(f"- {rule}" for rule in self.playbook)
        if self.memory is not None:
            from memory import format_cards
            lines = [line for m in messages[1:] if m["role"] == "user" for line in m["content"].splitlines()
                     if line.strip() and UNAVAILABLE not in line]
            query = "\n".join(lines)[:2000]
            own = (context or {}).get("case_id")
            cards = [c for c in self.memory.retrieve(query, k=4) if c["case_id"] != own][:3]  # never its own card
            retrieved = [{"case_id": c["case_id"], "similarity": c["similarity"]} for c in cards]
            text += ("\nSimilar past cases (other patients; use a lesson only where it fits this patient):\n"
                     + format_cards(cards))
        return text, retrieved

    def _act(self, messages, instruction, final, *, forbid_diagnosis=False):
        def parse(text):
            action = action_from_response(text, final=final)
            if forbid_diagnosis and kind(action) == "diagnosis":
                raise ValueError("A diagnosis was not allowed this time")
            differential = parse_object(text).get("differential", [])
            return action, differential if isinstance(differential, list) else []
        return request(self.client, self.model, PROMPT + instruction, conversation_input(messages), parse, role="agent")

    def _finish(self, record, action, calls, messages, context, lessons):
        """Commit a diagnosis, after reconciling it with every diagnosis considered so far."""
        proposal = re.search(r"<diagnosis>(.*?)</diagnosis>", action, re.S)[1].strip()
        considered = list((context or {}).get("scratch", {}).get("considered", []))
        candidates = list(dict.fromkeys(considered + [proposal]))
        if not self.reconcile or len({c.casefold() for c in candidates}) < 2:
            return {**record, "text": action, "agent_calls": calls}

        def parse(text):
            chosen = str(parse_object(text).get("diagnosis", "")).strip()
            if not chosen or "\n" in chosen or "<" in chosen or UNCOMMITTED.search(chosen):
                raise ValueError("The reconciled diagnosis must be one committed diagnosis")
            return chosen

        chosen, more = request(self.client, self.model,
                               GATE_INTRO + lessons + RECONCILE.format(candidates="; ".join(candidates)),
                               conversation_input(messages), parse, role="reconcile")
        calls += more
        chosen = chosen or proposal  # keep the proposal if reconciliation failed twice
        record["reconcile"] = {"candidates": candidates, "proposal": proposal, "chosen": chosen}
        return {**record, "text": f"<diagnosis>{chosen}</diagnosis>", "agent_calls": calls}

    def can_block(self, context, messages):
        """Code decides whether the gate may send the agent back for more evidence."""
        context = context or {}
        if context.get("mode") == "instant" or is_final(messages):
            return False
        if context.get("max_turns") and context["max_turns"] - context["turns_used"] <= 1:
            return False
        remaining = context.get("remaining")
        return remaining is None or remaining >= context.get("visit_cost", 300) + 100

    def __call__(self, messages, *, instruction="", final=False, context=None):
        final = is_final(messages, final)
        state = state_text(context, messages)
        lessons, retrieved = self.experience(messages, context)
        instruction = instruction + lessons
        result, calls = self._act(messages, instruction + STATE.format(state=state), final)
        if result is None:
            raise ValueError(f"{self.model} returned no valid diagnostic action after two attempts")
        action, differential = result
        if context is not None:  # remember every diagnosis ranked in this encounter
            notes = context.setdefault("scratch", {}).setdefault("considered", [])
            for item in differential:
                name = str(item.get("diagnosis", "")).strip() if isinstance(item, dict) else ""
                if name and "\n" not in name and name.casefold() not in {n.casefold() for n in notes}:
                    notes.append(name)
        record = {"model": self.model, "design": self.name, "state": state, "differential": differential,
                  "first_action": action, "gate": None, "retrieved": retrieved}
        if kind(action) != "diagnosis":
            return {**record, "text": action, "agent_calls": calls}
        if not self.gate:
            return self._finish(record, action, calls, messages, context, lessons)

        blockable = self.can_block(context, messages)
        asked = any(m["role"] == "assistant" and "<question>" in m["content"] for m in messages)
        if blockable and not asked:
            retry, more = self._act(messages, instruction + STATE.format(state=state) + ASK_FIRST, final,
                                    forbid_diagnosis=True)
            calls += more
            if retry is not None:
                record["gate"] = {"decision": "ask_first"}
                return {**record, "text": retry[0], "agent_calls": calls}

        proposal = re.search(r"<diagnosis>(.*?)</diagnosis>", action, re.S)[1].strip()

        def parse_gate(text):
            data = parse_object(text)
            if data.get("decision") not in ("approve", "continue"):
                raise ValueError("decision must be approve or continue")
            refined = str(data.get("diagnosis") or proposal).strip()
            if "\n" in refined or not refined or "<" in refined:
                raise ValueError("Invalid refined diagnosis")
            if data["decision"] == "approve" and UNCOMMITTED.search(refined):
                raise ValueError("An approved diagnosis must be committed")
            return {"decision": data["decision"], "reason": str(data.get("reason", "")).strip(), "diagnosis": refined}

        verdict, gate_calls = request(self.client, self.model,
                                      # The gate has its own prompt: the agent's action format would
                                      # compete with the decision format once memory text is added.
                                      GATE_INTRO + lessons + GATE.format(state=state, proposal=proposal),
                                      conversation_input(messages), parse_gate, role="closure_gate")
        calls += gate_calls
        if verdict is None:  # the gate failed twice: keep the agent's own diagnosis
            record["gate"] = {"decision": "unavailable"}
            return self._finish(record, action, calls, messages, context, lessons)
        if verdict["decision"] == "continue" and blockable:
            retry, more = self._act(messages, instruction + STATE.format(state=state)
                                    + CONTINUE.format(reason=verdict["reason"]), final, forbid_diagnosis=True)
            calls += more
            if retry is not None:
                record["gate"] = verdict
                return {**record, "text": retry[0], "agent_calls": calls}
        if verdict["decision"] == "continue":
            # The gate wanted more evidence but may not block (no budget or turns left, or the agent
            # insisted). Keep the agent's own diagnosis: the gate's text is not a committed answer.
            record["gate"] = {**verdict, "decision": "approve_forced"}
            return self._finish(record, action, calls, messages, context, lessons)
        record["gate"] = verdict
        return self._finish(record, f"<diagnosis>{verdict['diagnosis']}</diagnosis>", calls, messages, context, lessons)

