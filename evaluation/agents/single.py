"""Single agent: one model call chooses each action. This is SDBench's baseline design."""

from .base import PROMPT, action_from_response, call_model, conversation_input, is_final


class SingleAgent:
    name = "single"

    def __init__(self, client, model, agents=1):
        if agents != 1:
            raise ValueError("The single agent uses exactly one agent")
        self.client = client
        self.model = model

    def __call__(self, messages, *, instruction="", final=False, context=None):
        """One diagnostic turn. Retries once with a larger output budget on an invalid action."""
        final = is_final(messages, final)
        prior_calls = []
        for limit in (4096, 8192):
            text, call = call_model(self.client, self.model, PROMPT + instruction,
                                    conversation_input(messages), max_output_tokens=limit)
            try:
                if text is not None:
                    return {**call, "text": action_from_response(text, final=final),
                            "previous_attempts": prior_calls}
            except ValueError:
                pass
            prior_calls.append(call)
        raise ValueError(f"{self.model} returned no valid diagnostic action after two attempts")
