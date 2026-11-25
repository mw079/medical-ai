"""Check experiment specs, paired statistics, token counting and the model registry offline."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import run_arms
from compare_repeated import paired
from models import PRICE_FIELDS, registry, token_cost
from compare_repeated import agent_usage


def spec(**changes):
    base = {"name": "t", "policy": "reviewed-cpt-bedside-v4", "splits": {"validation": 2},
            "modes": ["budget_1000"], "arms": {"baseline": None}, "runs": 1}
    return {**base, **changes}


def load(value):
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.json"
        path.write_text(json.dumps(value))
        return run_arms.load_spec(path)


class SpecTests(unittest.TestCase):
    def test_valid_spec_and_alternating_order(self):
        loaded = load(spec(arms={"baseline": None, "panel": "evaluation/harnesses/panel.py"}, runs=2))
        self.assertEqual(run_arms.command(loaded, "validation", "budget_1000", "panel", 2, 4)[-2:],
                         ["--harness", "evaluation/harnesses/panel.py"])
        self.assertTrue(str(run_arms.run_dir(loaded, "validation", "budget_1000", "panel", 2)).endswith(
            "data/runs/reviewed-cpt-bedside-v4/validation/budget_1000/panel/run2"))

    def test_test_splits_need_explicit_held_out_flag(self):
        with self.assertRaisesRegex(ValueError, "held_out"):
            load(spec(splits={"external_test": 50}))
        loaded = load(spec(splits={"external_test": 50}, held_out=True))
        self.assertIn("--held-out", run_arms.command(loaded, "external_test", "budget_1000", "baseline", 1, 4))

    def test_rejects_unsafe_or_incomplete_specs(self):
        for bad, message in (
            (spec(arms={"panel": "evaluation/harnesses/panel.py"}), "baseline"),
            (spec(arms={"baseline": None, "x": "evaluation/harnesses/missing.py"}), "Missing harness"),
            (spec(modes=["instant"], arms={"baseline": None, "panel": "evaluation/harnesses/panel.py"}), "Harness arms"),
            (spec(policy="v9"), "Unknown policy"),
            (spec(runs=0), "runs"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                load(bad)


class StatisticsTests(unittest.TestCase):
    def test_paired_statistics(self):
        self.assertEqual(paired([0.0] * 10)["delta_points"], 0)
        self.assertEqual(paired([0.0] * 10)["p_value"], 1.0)
        strong = paired([1.0] * 20)
        self.assertEqual(strong["delta_points"], 100)
        self.assertLess(strong["p_value"], 0.001)


class UsageTests(unittest.TestCase):
    def test_counts_retries_and_team_calls_but_not_the_environment(self):
        usage = {"input_tokens": 10, "output_tokens": 4, "output_tokens_details": {"reasoning_tokens": 1}}
        trace = [
            {"role": "gatekeeper", "usage": usage},
            {"role": "diagnostic", "usage": usage, "previous_attempts": [{"usage": usage}],
             "agent_calls": [{"usage": usage}, {"usage": None}]},
        ]
        self.assertEqual(agent_usage(trace), {"calls": 4, "input_tokens": 30, "cached_tokens": 0,
                                              "cache_write_tokens": 0, "output_tokens": 12,
                                              "reasoning_tokens": 3})

    def test_token_cost_uses_cached_and_output_prices(self):
        usage = {"calls": 1, "input_tokens": 1_000_000, "cached_tokens": 400_000, "cache_write_tokens": 0,
                 "output_tokens": 1_000_000, "reasoning_tokens": 900_000}
        # 600K uncached at $1, 400K cached at $0.05, 1M output (reasoning included) at $5.
        self.assertAlmostEqual(token_cost(usage, "gpt-6.1-terra"), 0.6 + 0.02 + 5.0)
        config = registry()
        config["models"]["gpt-6.1-terra"]["usd_per_million_tokens"]["cache_write"] = None
        with patch("models.registry", return_value=config), self.assertRaisesRegex(ValueError, "cache_write"):
            token_cost({**usage, "cache_write_tokens": 10}, "gpt-6.1-terra")
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            token_cost(usage, "not-a-model")


class ModelArmTests(unittest.TestCase):
    def test_model_arms_pass_the_model_and_default_to_luna(self):
        loaded = load(spec(arms={"baseline": None, "terra": {"model": "gpt-6.1-terra"}}))
        self.assertEqual(run_arms.command(loaded, "validation", "budget_1000", "terra", 1, 4)[-2:],
                         ["--model", "gpt-6.1-terra"])
        self.assertIn("gpt-6-luna", run_arms.command(loaded, "validation", "budget_1000", "baseline", 1, 4))

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown model"):
            load(spec(arms={"baseline": None, "x": {"model": "not-a-model"}}))

    def test_every_model_entry_has_a_provider_and_all_prices(self):
        config = registry()
        for name, model in config["models"].items():
            with self.subTest(model=name):
                self.assertIn(model["provider"], config["providers"])
                self.assertEqual(set(model["usd_per_million_tokens"]), set(PRICE_FIELDS))
                self.assertTrue(model["price_source"])
