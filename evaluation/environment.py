"""The evaluation environment: SDBench's agent loop and judge with an evidence-only gatekeeper.

The gatekeeper may release only findings stored for the case. Repeated requests reuse the first
stored match. Patient questions may release only history and routine bedside findings, decided
once per case. The judge is SDBench's prompt with a strict score parser.
"""

import fcntl
import hashlib
import json
from pathlib import Path
import re

from openai.types.chat import ChatCompletion
from sdbench.agents.baseline import extract_diagnosis_from_response
from sdbench.config import SDBenchConfig
from sdbench.harness import run_case

from policy import get_policy


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-6-luna"
UNAVAILABLE = "Not documented in this encounter."
JUDGE_PLACEHOLDERS = (
    "{{REFERENCE_DIAGNOSIS}}", "{{CANDIDATE_DIAGNOSIS}}", "{{CASE_CONTEXT}}", "{{TEST_LOGS}}",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Write atomically so a crash never leaves a half-written result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


# Judge score

SCORE_MARKER = re.compile(r"\*\*SCORE\*\*:", re.IGNORECASE)
FINAL_SCORE = re.compile(r"\*\*SCORE\*\*:\s*(?:([1-5])|\[([1-5])\]|\*\*([1-5])\*\*)\s*\Z",
                         re.IGNORECASE)


def explicit_judge_score(text):
    match = FINAL_SCORE.search(text) if isinstance(text, str) else None
    if match is None or len(SCORE_MARKER.findall(text)) != 1:
        raise ValueError("Judge must return exactly one final explicit score from 1 to 5")
    return int(next(value for value in match.groups() if value is not None))


def verify_saved_grade(result):
    score = explicit_judge_score(result["check_diagnosis_output"])
    if result["diagnosis_score"] != score or result["correct"] != (score >= 4):
        raise ValueError("Saved grade disagrees with the explicit judge score")
    return score


# Which findings a patient question may release

SPECIALIZED_FINDING = re.compile(
    r"biomicroscop|slit[\s\-‐-―]*lamp|fundoscop|funduscop|ophthalmoscop|\bfundus\b|"
    r"retinal (?:exam|imaging|photograph|scan)|"
    r"tonometr|gonioscop|retinoscop|intraocular pressure|perimetr|optical coherence|"
    r"fluorescein|angiograph|dermoscop|endoscop|laryngoscop|fluoroscop|ultrason|"
    r"echocardiog|\b(?:CT|MRI|OCT)\b|histopatholog|immunohistochem|biops",
    re.IGNORECASE,
)


def routine_candidates(findings):
    return {i: f for i, f in enumerate(findings)
            if not SPECIALIZED_FINDING.search(f"{f['name']} {f['result']}")}


def question_evidence(client, findings):
    """Allow only findings a clinician could obtain by talking or routine examination."""
    candidates = routine_candidates(findings)
    if not candidates:
        return []
    instructions = (
        "Return a JSON array of the integer keys of findings obtainable from patient "
        "history or routine bedside examination. Exclude specialized or instrumented "
        "eye exams, including slit-lamp and retinal examinations. Exclude lab results, "
        "imaging, biopsies, other specialty procedures, or surgery, even if historical. "
        "Exclude mixed findings containing any such results, and anything uncertain. Include no prose."
    )
    for _ in range(3):
        response = client.responses.create(
            model=MODEL, reasoning={"effort": "none"},
            input=[{"role": "developer", "content": instructions},
                   {"role": "user", "content": json.dumps(candidates, ensure_ascii=False)}],
            max_output_tokens=200, store=False,
        )
        try:
            ids = json.loads(response.output_text or "")
        except ValueError:
            continue
        if isinstance(ids, list) and all(type(i) is int for i in ids):
            return sorted(candidates.keys() & set(ids))
    raise ValueError("Could not classify question evidence")


def completion(text: str, *, model: str = MODEL) -> ChatCompletion:
    return ChatCompletion(
        id="sdbench-environment", created=0, model=model, object="chat.completion",
        choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
    )


def request_key(text):
    """Normalize a request so trivially different spellings share one stored reply."""
    return " ".join(text.casefold().split()).rstrip(" .?")


class EvidenceCache:
    """Persist the selector's finding IDs per case and normalized request.

    The first match for a request is kept, so the same request gets the same evidence in every
    mode, run, and harness that shares the file. The file is tied to the case's findings.
    """

    def __init__(self, path, findings, question_findings):
        self.path = Path(path)
        digest = hashlib.sha256(json.dumps(findings, sort_keys=True).encode()).hexdigest()
        self.identity = {"findings_sha256": digest,
                         "question_findings": None if question_findings is None else sorted(question_findings)}

    def _access(self, update=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_name(self.path.name + ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {**self.identity, "replies": {}}
            if {key: data.get(key) for key in self.identity} != self.identity:
                raise ValueError(f"Gatekeeper cache {self.path} was built from different evidence")
            if update:
                for key, row in update.items():
                    data["replies"].setdefault(key, row)
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
                tmp.replace(self.path)
            return dict(data["replies"])

    def get(self):
        return self._access()

    def put(self, update):
        return self._access(update)


def evaluate_case(case, audit, client, *, diagnostic=None, question_findings=None,
                  gatekeeper_cache=None, evidence_log=None, policy=None):
    policy = policy or get_policy()
    findings = [{"name": f["name"], "result": f["result"]} for f in audit["draft"]["findings"]]
    evidence_lines = [f"{f['name']}: {f['result']}" for f in findings]
    if any(line not in case["full_text"] for line in evidence_lines):
        raise ValueError("Export and audit evidence differ")
    trace = []

    def ask(role, messages, *, response_format=None):
        options = {"text": {"format": response_format}} if response_format else {}
        response = client.responses.create(
            model=MODEL, input=messages, reasoning={"effort": "none"},
            max_output_tokens=900, store=False, **options,
        )
        if response.status != "completed" or not response.output_text:
            raise RuntimeError(f"{role} model returned {response.status} or empty text")
        trace.append({"role": role, "messages": messages, "response": response.output_text,
                      "model": response.model, "usage": response.usage.model_dump() if response.usage else None})
        return response.output_text.strip()

    def diagnose(messages):
        if not any(entry["role"] == "diagnostic" for entry in trace):
            if len(messages) != 2 or case["case_summary"] not in messages[1]["content"]:
                raise ValueError("Unexpected diagnostic opening")
            opening = "\n".join(message["content"] for message in messages).casefold()
            if case["final_diagnosis"].casefold() in opening:
                raise ValueError("Diagnostic opening contains the reference diagnosis")
            if any(f["result"].casefold() in opening for f in findings
                   if f["result"].casefold() not in case["case_summary"].casefold()):
                raise ValueError("Diagnostic opening contains unreleased evidence")
        if diagnostic is None:
            call = {"text": ask("diagnostic", list(messages)), "model": MODEL}
        else:
            call = diagnostic(list(messages))
            trace.append({"role": "diagnostic", "messages": list(messages), "response": call["text"], **call})
        diagnosis = extract_diagnosis_from_response(call["text"])
        if diagnosis and any(token in diagnosis for token in JUDGE_PLACEHOLDERS):
            raise ValueError("Diagnostic answer contains a reserved judge prompt placeholder")
        return completion(call["text"], model=call["model"])

    cache = EvidenceCache(gatekeeper_cache, findings, question_findings) if gatekeeper_cache else None
    question_rule = ""
    if question_findings is not None:
        question_rule = (
            " For requests prefixed 'Question for the patient:', use only question_finding_ids. "
            "Do not give lab, imaging, pathology, or other diagnostic test results as answers to questions."
        )

    def match(answers, request_payload, *, response_format=None):
        return json.loads(ask("gatekeeper", [
            {"role": "system", "content": (
                "Match each request to zero-indexed IDs of directly relevant documented findings. "
                + answers +
                "Use [] when the request has no documented component. If a request combines several tests "
                "or questions, return all relevant findings that answer its documented parts. "
                "For example, a request for a repeat CBC and blood smear can match a documented manual "
                "blood film even when a separate repeat CBC is unavailable. "
                "Do not substitute unrelated tests or infer a result." + question_rule
            )},
            {"role": "user", "content": json.dumps(request_payload, ensure_ascii=False)},
        ], response_format=response_format))

    def select(requests):
        """Ask the selector for finding IDs, one row per (id, request) pair."""
        payload = {"requests": [text for _, text in requests], "findings": findings}
        if question_findings is not None:
            payload["question_finding_ids"] = sorted(question_findings)

        def valid(rows):
            return isinstance(rows, list) and len(rows) == len(requests) and all(
                isinstance(row, list)
                and all(type(index) is int and 0 <= index < len(findings) for index in row)
                for row in rows
            )

        try:
            selected = match("Return one JSON array of arrays, one inner array per request in order. ", payload)
        except json.JSONDecodeError:
            if question_findings is None:
                raise
            selected = None

        if question_findings is not None and not valid(selected):
            # Pin one output field to each request so an empty match stays in its own row.
            keyed_requests = {f"request_{i}": request for i, request in requests}
            properties = {}
            for key, request in keyed_requests.items():
                allowed = sorted(question_findings) if request.startswith("Question for the patient:") else list(range(len(findings)))
                properties[key] = ({"type": "array", "items": {"type": "integer", "enum": allowed}}
                                   if allowed else {"type": "array", "items": {"type": "integer"}, "maxItems": 0})
            response_format = {
                "type": "json_schema", "name": "evidence_matches", "strict": True,
                "schema": {"type": "object", "properties": properties, "required": list(properties),
                           "additionalProperties": False},
            }
            response = match("Map each request key to an array of matching finding IDs. ",
                             {**payload, "requests": keyed_requests}, response_format=response_format)
            selected = [response.get(key) for key in keyed_requests] if isinstance(response, dict) else None
        if not valid(selected):
            raise ValueError("Gatekeeper returned invalid evidence IDs")
        return selected

    def gatekeep(messages):
        # SDBench sends numbered requests. A selector can only choose stored evidence.
        requests = re.findall(r"^ID (\d+): (.+)$", messages[-1]["content"], re.MULTILINE)
        if not requests:
            raise ValueError("Missing numbered SDBench requests")
        if cache is None:
            selected = select(requests)
        else:
            # Repeated requests reuse the first stored match, within and across runs.
            keys = [request_key(text) for _, text in requests]
            known = cache.get()
            new = list({request_key(text): (i, text) for i, text in requests
                        if request_key(text) not in known}.values())
            if new:
                known = cache.put({request_key(text): row for (_, text), row in zip(new, select(new))})
            selected = [known[key] for key in keys]
        blocks = []
        for (request_id, request), row in zip(requests, selected, strict=True):
            if question_findings is not None and request.startswith("Question for the patient:"):
                row = [i for i in row if i in question_findings]
            result = " | ".join(evidence_lines[index] for index in dict.fromkeys(row)) or UNAVAILABLE
            if re.search(r"</?(?:result|test_result|is_synthetic)\b", result, re.IGNORECASE):
                raise ValueError("Evidence cannot contain gatekeeper protocol tags")
            if evidence_log is not None:
                evidence_log.append({"request": request, "result": result})
            blocks.append(f'<test_result id="{request_id}"><result>{result}</result>'
                          '<is_synthetic>false</is_synthetic></test_result>')
        return completion("\n".join(blocks))

    def judge(messages):
        latest = next((entry for entry in reversed(trace) if entry["role"] == "diagnostic"), None)
        if latest is None or not extract_diagnosis_from_response(latest["response"]):
            raise ValueError("Judge called before a diagnostic commitment")
        if case["final_diagnosis"] not in messages[-1]["content"]:
            raise ValueError("Judge prompt is missing the reference diagnosis")
        answer = ask("judge", list(messages))
        score = explicit_judge_score(answer)
        # SDBench's fallback scans for the first digit. Give it a canonical score marker.
        return completion(FINAL_SCORE.sub(f"**SCORE**: {score}", answer))

    config = SDBenchConfig(max_turns=policy.max_turns,
                           max_questions_per_turn=policy.max_questions_per_turn,
                           max_tests_per_turn=policy.max_tests_per_turn,
                           compute_costs=False, gatekeeper_prompt_version="v6_fixed",
                           gatekeeper_model=MODEL, judge_prompt_version="v4_reasoning_with_tests",
                           judge_model=MODEL)
    result = run_case(config, case, "baseline", diagnose, gatekeeper_llm=gatekeep, judge_llm=judge)
    verify_saved_grade(result)
    known = set(evidence_lines)
    if not all(not log["is_synthetic"] and (
        log["result"] == UNAVAILABLE or set(log["result"].split(" | ")) <= known
    ) for log in result.get("test_logs", [])):
        raise ValueError("Gatekeeper released undocumented or synthetic evidence")
    return {"case_id": case["case_id"],
            "config": {"environment_model": MODEL, "policy": policy.name, "max_turns": policy.max_turns,
                       "max_questions_per_turn": policy.max_questions_per_turn,
                       "max_tests_per_turn": policy.max_tests_per_turn,
                       "runner": "SDBench baseline", "gatekeeper": "documented evidence selector"},
            "result": result, "model_calls": trace}
