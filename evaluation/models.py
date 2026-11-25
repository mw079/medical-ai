"""Diagnostic models: which provider serves each one, its reasoning level, and its token prices.

Everything lives in models.json, so adding or switching a model is a config change. Providers
must offer an OpenAI-compatible Responses API. API keys stay in the environment or .env; the
config only names the variable. The gatekeeper and judge do not use this file: they are fixed
by the environment (environment.MODEL), so every agent model is graded the same way.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name("models.json")
DEFAULT_MODEL = "gpt-6-luna"
PRICE_FIELDS = ("input", "cached_input", "cache_write", "output")


def registry():
    return json.loads(CONFIG.read_text())


def model_config(name):
    """The model's entry merged with its provider's settings. Raises for an unknown model."""
    config = registry()
    if name not in config["models"]:
        raise ValueError(f"Unknown model {name!r}. Add it to {CONFIG.name}: {sorted(config['models'])}")
    model = config["models"][name]
    provider = config["providers"][model["provider"]]
    return {"name": name, **model, "base_url": provider["base_url"], "key_env": provider["key_env"],
            "options": provider.get("options", {}), "timeout_seconds": provider.get("timeout_seconds", 90)}


def api_key(variable="OPENAI_API_KEY"):
    """An API key from the environment, else from .env. Raises if neither has it."""
    from dotenv import dotenv_values

    key = os.environ.get(variable) or dotenv_values(ROOT / ".env").get(variable)
    if not key:
        raise ValueError(f"Set {variable} in the environment or .env")
    return key


def client_for(name):
    from openai import OpenAI

    model = model_config(name)
    key = api_key(model["key_env"])
    # Long reasoning can take minutes; the timeout is set per provider in models.json.
    return OpenAI(api_key=key, base_url=model["base_url"], timeout=model["timeout_seconds"])


def token_cost(usage, name):
    """Dollar cost of summed token usage (see usage.agent_usage) at the model's listed prices.

    Reasoning tokens are part of output_tokens for both providers. Cached and cache-write input
    tokens are billed at their own rates. A price the provider does not list must not be needed.
    """
    prices = model_config(name)["usd_per_million_tokens"]
    if set(prices) != set(PRICE_FIELDS):
        raise ValueError(f"{name} needs prices for {PRICE_FIELDS}")
    parts = {
        "input": usage["input_tokens"] - usage["cached_tokens"] - usage["cache_write_tokens"],
        "cached_input": usage["cached_tokens"],
        "cache_write": usage["cache_write_tokens"],
        "output": usage["output_tokens"],
    }
    if parts["input"] < 0:
        raise ValueError("Cached tokens exceed input tokens")
    total = 0.0
    for field, tokens in parts.items():
        if tokens and prices[field] is None:
            raise ValueError(f"No {field} price listed for {name}")
        total += tokens * (prices[field] or 0) / 1_000_000
    return total
