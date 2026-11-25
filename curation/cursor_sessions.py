"""Run the writer and first reviewer through the Cursor SDK, one fresh sandboxed session per call.

Each call returns the first complete JSON answer, cancels the run, and records its session ID,
status and token usage. Writer and reviewer never share a session.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile

from cursor_sdk import (
    AgentOptions, CursorAgentError, CursorClient, LocalAgentOptions, ModelParameterValue, ModelSelection, RunResult,
    SandboxOptions, SDKAssistantMessage, SDKUsageMessage, TextDeltaUpdate,
)

from validation import complete_response

MODEL = "grok-4.7"
MODEL_SELECTION = ModelSelection(id=MODEL, params=[ModelParameterValue(id="reasoning_effort", value="medium")])


def finish_cursor_response(run, session_id: str, response: str) -> dict:
    record = {"session_id": session_id, "run_id": run.id, "text": response}
    try:
        if run.status == "running":
            run.cancel()
        result = run.wait()
        record["cursor_status"] = result.status
    except CursorAgentError as error:
        record["cursor_status"] = run.status
        record["stop_error"] = type(error).__name__
    usage = run.usage
    if usage is not None:
        record["usage"] = asdict(usage)
    return record


class CursorSessions:
    def __init__(self, client: CursorClient, workspace: Path, key: str, *, model=MODEL_SELECTION):
        self.client = client
        self.workspace = workspace
        self.key = key
        self.model = model

    def __call__(self, prompt: str, payload: dict) -> dict:
        with tempfile.TemporaryDirectory(dir=self.workspace) as directory:
            local = LocalAgentOptions(cwd=directory, sandbox_options=SandboxOptions(enabled=True))
            with self.client.agents.create(AgentOptions(api_key=self.key, model=self.model, local=local)) as agent:
                run = agent.send(prompt + "\nINPUT JSON\n" + json.dumps(payload, ensure_ascii=False))
                assistant_text: list[str] = []
                text_deltas: list[str] = []
                for event in run.events():
                    if event.result_is_full:
                        result = RunResult.from_json(event.result)
                        if result.status != "finished" or (result.model and result.model.id != self.model.id):
                            raise RuntimeError(f"Cursor run {run.id} ended with {result.status} ({result.error_code})")
                        return finish_cursor_response(run, agent.agent_id, result.result)
                    message = event.sdk_message
                    update = event.interaction_update
                    if isinstance(message, SDKUsageMessage):
                        assistant_text.clear()
                        text_deltas.clear()
                        continue
                    if isinstance(message, SDKAssistantMessage):
                        fragment = "".join(block.text for block in message.message.content
                                           if block.type == "text")
                        buffer = assistant_text
                    elif isinstance(update, TextDeltaUpdate):
                        fragment = update.text
                        buffer = text_deltas
                    else:
                        continue
                    buffer.append(fragment)
                    if not run.id or not fragment.rstrip().endswith(("}", "```")):
                        continue
                    # SDK messages can contain either a full answer or a single text fragment.
                    # Return a complete top-level response before the bridge retries the turn.
                    for response in (fragment, "".join(buffer)):
                        if complete_response(response):
                            return finish_cursor_response(run, agent.agent_id, response)
                raise RuntimeError(f"Cursor run {run.id} ended without an answer")
