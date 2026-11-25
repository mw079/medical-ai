"""Shared pieces for diagnostic agents: the adapter prompt, action validation, one model call.

SDBench's baseline agent builds the conversation. An agent receives it as JSON under a short
adapter prompt and returns one action, which is validated before SDBench sees it.
"""

import json
import re

from models import model_config

PROMPT = """Use the supplied SDBench conversation for your next diagnostic turn.
Follow its system message and use only its patient summary and replies. Do not use tools,
files, web search, or outside sources. Return one JSON object with an action string containing
only the tags requested by SDBench. For example {"action":"<question>When did this begin?</question>"}.
Use exactly one diagnosis tag if diagnosing. If there are no more tests available, diagnose now.
"""
ACTION = re.compile(r"\s*<(question|test|diagnosis)>\s*([^<>]+?)\s*</\1>\s*", re.I)


def parse_object(text):
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), re.DOTALL)
    value = json.loads(fenced[1] if fenced else text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def action_from_response(text, *, final=False):
    """Return the action string, or raise if it is not exactly one valid SDBench turn."""
    data = parse_object(text)
    action = data.get("action")
    if not isinstance(action, str):
        raise ValueError("Missing diagnostic action")
    actions = list(ACTION.finditer(action))
    if not actions or not all(m[2].strip() for m in actions) or ACTION.sub("", action).strip():
        raise ValueError("Invalid SDBench action tags")
    if any(len(match[2].splitlines()) != 1 or "\n" in match[2] or "\r" in match[2]
           for match in actions):
        raise ValueError("Diagnostic actions must be single-line")
    kinds = {m[1].lower() for m in actions}
    if len(kinds) != 1 or ("diagnosis" in kinds and len(actions) != 1):
        raise ValueError("Mixed action types or multiple diagnoses")
    if final and "diagnosis" not in kinds:
        raise ValueError("A final diagnosis is required")
    return action.strip()


def is_final(messages, final=False):
    return final or messages[-1]["content"].startswith("No more tests available.")


def conversation_input(messages):
    return "INPUT JSON\n" + json.dumps({"messages": messages}, ensure_ascii=False)


def call_model(client, model, developer, user, *, max_output_tokens):
    """One call to a model from models.json. Returns its output text and a record with token usage."""
    config = model_config(model)
    response = client.responses.create(
        model=model, input=[{"role": "developer", "content": developer}, {"role": "user", "content": user}],
        reasoning={"effort": config["reasoning"]}, max_output_tokens=max_output_tokens, store=False,
        **config["options"],
    )
    record = {"model": response.model, "response_id": response.id, "status": response.status,
              "reasoning_effort": config["reasoning"], "service_tier": getattr(response, "service_tier", None),
              "usage": response.usage.model_dump() if response.usage else None}
    return (response.output_text or "" if response.status == "completed" else None), record


def request(client, model, developer, user, parse, *, role):
    """Call the model and parse its answer, retrying once with a larger output budget.

    Returns (parsed value or None, call records). Each record carries the role, the raw output
    and token usage, so team calls can be audited and counted.
    """
    calls = []
    for limit in (4096, 8192):
        text, call = call_model(client, model, developer, user, max_output_tokens=limit)
        calls.append({**call, "role": role, "output": text})
        try:
            if text is not None:
                return parse(text), calls
        except (ValueError, TypeError, KeyError):
            pass
    return None, calls
