# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sdbench[cpt-costing] @ git+https://github.com/microsoft/sdbench@304ce9625a046cea97d64e3b1ba5d19217b5095c",
#   "python-dotenv==1.1.0",
# ]
# ///
"""Run diagnostic encounters under one evaluation policy and budget mode.

Each (mode, case) result is saved as JSON and reused on resume. A run folder is tied to its
settings, so changing the policy, cases, harness or prices needs a new --output folder.
"""

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from unittest.mock import patch

from openai import OpenAI
from sdbench.case_data import get_case
from sdbench.config import CostingConfig, SDBenchConfig
from sdbench.costing import cpt_matcher, llm_estimater
from sdbench.costing.costing import get_cpt_fuzzy_matcher

from agents import AGENTS, SINGLE_DESIGNS, make_agent
from budget import MODES, MedicalPolicy
from environment import MODEL, ROOT, evaluate_case, question_evidence, read_json, verify_saved_grade, write_json
from models import DEFAULT_MODEL, api_key, model_config, registry
from policy import DEFAULT_POLICY, POLICIES, get_policy
from pricing import GroundedPrices, KeywordIndex, shipped_prices
from release import prepare


HARNESS_MODES = {"budget_1000", "budget_4000"}  # modes where a playbook may be evaluated
HELD_OUT = {"same_source_test", "external_test"}
PRICE_FILE = ROOT / "data/costing_data/910511770-central-hospital-standard-charges-wa-en.xlsx"



def summarize(output, ids, selected_modes=MODES):
    modes = {}
    for mode in selected_modes:
        outcomes = [read_json(output / mode / f"{i}.json") for i in ids]
        scored = [row for row in outcomes if "result" in row]
        for row in scored:
            try:
                verify_saved_grade(row["result"])
            except ValueError as error:
                raise ValueError(f"Cannot summarize {mode}/{row['case_id']}: {error}") from error
        costs = [row["medical_cost"] for row in scored]
        n = len(scored)
        methods = Counter()
        for row in costs:
            methods.update(row["methods"])
        correct = sum(row["result"]["correct"] for row in scored)
        modes[mode] = {"scored": n, "correct": correct, "accuracy": correct / n if n else None,
                       "avg_medical_cost_usd": round(mean(c["total_usd"] for c in costs), 2) if n else None,
                       "avg_test_cost_usd": round(mean(c["test_cost_usd"] for c in costs), 2) if n else None,
                       "cost_methods": dict(methods),
                       "unavailable_orders": sum(c["unavailable_orders"] for c in costs),
                       "blocked_batches": sum(len(c["blocked_batches"]) for c in costs),
                       "errors": {row["case_id"]: row["error"] for row in outcomes if "error" in row}}
    write_json(output / "summary.json", {"cases": ids, "modes": modes})
    print(json.dumps(modes, indent=2), flush=True)
    return int(any(row["errors"] for row in modes.values()))


def read_playbook(path):
    """Read a declarative harness without importing or executing candidate code."""
    if path.stat().st_size > 9000:
        raise ValueError("Harness exceeds the size limit")
    module = ast.parse(path.read_text())
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
        raise ValueError("Harness must contain only a PLAYBOOK assignment")
    assignment = module.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id != "PLAYBOOK":
        raise ValueError("Harness must assign the PLAYBOOK literal")
    data = ast.literal_eval(assignment.value)
    if not isinstance(data, dict) or not data.keys() <= {"opening", "follow_up"} or any(
        not isinstance(text, str) or len(text) > 3500 for text in data.values()
    ):
        raise ValueError("PLAYBOOK may contain short opening and follow_up guidance")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "validation", *sorted(HELD_OUT)), required=True)
    parser.add_argument("--limit", type=int, required=True, help="number of cases, in export order")
    parser.add_argument("--output", type=Path, required=True, help="run folder, one per configuration")
    parser.add_argument("--mode", choices=MODES, help="one budget mode; default runs all four")
    parser.add_argument("--policy", choices=sorted(POLICIES), default=DEFAULT_POLICY)
    parser.add_argument("--model", choices=sorted(registry()["models"]), default=DEFAULT_MODEL,
                        help="diagnostic agent model from evaluation/models.json")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="single", help="agent design")
    parser.add_argument("--agents", type=int, default=1, help="number of agents in a team design")
    parser.add_argument("--harness", type=Path, help="playbook file; omit for the baseline")
    parser.add_argument("--held-out", action="store_true",
                        help="confirm a deliberate run on a test split (final comparison only)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chargemaster", type=Path, default=PRICE_FILE)
    args = parser.parse_args()
    policy = get_policy(args.policy)
    selected_modes = {args.mode: MODES[args.mode]} if args.mode else MODES
    if (args.split in HELD_OUT) != args.held_out:
        parser.error("Pass --held-out exactly when the split is a test split")
    if (args.agent in SINGLE_DESIGNS) != (args.agents == 1):
        parser.error("Use --agents 1 with a single-agent design and 2 or more with a team design")
    if args.harness and args.agent != "single":
        parser.error("Playbooks apply to the single agent only")
    if args.harness and args.mode not in HARNESS_MODES:
        parser.error(f"Harnesses are evaluated only in {sorted(HARNESS_MODES)}; pass one --mode")
    playbook = read_playbook(args.harness) if args.harness else {}
    packed, batch = prepare(args.split, policy.gatekeeper_cache)
    output = args.output
    os.environ["SDBENCH_DATA_PATH"] = str(batch / "data")
    dataset = batch / "data" / "processed_case_data" / args.split / "dataset.json"
    ids = [row["case_id"] for row in packed["cases"][:args.limit]]
    if len(ids) != args.limit or min(args.limit, args.workers) < 1:
        parser.error("Choose a valid case count and positive worker count")
    if not args.chargemaster.is_file():
        parser.error(f"Missing SDBench hospital price list: {args.chargemaster}")
    # Only the costing fields of this config are used. The encounter config is in environment.py.
    config = CostingConfig.from_sdbench_config(SDBenchConfig(
        max_turns=policy.max_turns, gatekeeper_prompt_version="v6_fixed", gatekeeper_model=MODEL,
        judge_prompt_version="v4_reasoning_with_tests", judge_model=MODEL,
        costing_method="unified_cpt_llm", costing_model="gpt-5-mini",
        costing_source_file_path=str(args.chargemaster.resolve()),
    ))
    price_hash = hashlib.sha256(args.chargemaster.read_bytes()).hexdigest()
    agent_model = model_config(args.model)
    settings = {"case_ids": ids, "split": args.split, "modes": selected_modes, "model": args.model,
                "reasoning": agent_model["reasoning"], "costing_method": config.costing_method,
                "costing_model": config.costing_model, "visit_cost_usd": policy.visit_cost,
                "policy_version": policy.name, "billing_review_model": MODEL,
                "gatekeeper_cache": policy.gatekeeper_cache, "charge_undocumented_tests": False,
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "chargemaster_sha256": price_hash, **policy.settings()}
    if args.harness:
        settings["playbook"] = playbook
    if args.agent != "single":
        settings.update(agent=args.agent, agents=args.agents)
    if args.agent in ("evidence_memory", "evidence_playbook", "panel_team_memory"):
        # A run folder is tied to the exact memory it used.
        memory_dir = ROOT / "evaluation/harnesses/memory"
        settings["memory_sha256"] = hashlib.sha256(b"".join(
            (memory_dir / name).read_bytes() for name in ("dev_lessons.json", "playbook.json"))).hexdigest()
    settings_path = output / "settings.json"
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".settings.lock").open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if settings_path.exists() and read_json(settings_path) != settings:
            parser.error("Settings changed. Choose a new output directory.")
        if not settings_path.exists():
            write_json(settings_path, settings)
    try:
        key = api_key()
    except ValueError as error:
        parser.error(str(error))
    os.environ["OPENAI_API_KEY"] = key  # SDBench's embedding client reads this.
    cases = {i: get_case(args.split, i) for i in ids}
    audits = {i: packed["audit"][i] for i in ids}
    previous_prices = args.chargemaster.resolve().parent / f"estimates-{config.costing_model}-{price_hash[:16]}"

    with OpenAI(api_key=key, timeout=90) as client:
        # Reviewed prices are shared across policies so a policy change never re-prices tests.
        reviewed = previous_prices.with_name(f"{previous_prices.name}-{policy.price_review_cache}")
        price = GroundedPrices(config, reviewed, previous_prices, client, shipped_prices())

        def direct_completion(*, model, messages, debug=False, **kwargs):
            # Keep the pinned CPT matching and fallback prompts. Use the existing OpenAI key.
            return client.chat.completions.create(model=model, messages=list(messages))

        diagnostic = make_agent(args.agent, args.model, args.agents)

        def classify(case_id):
            # Classify once per case and share it, so every run allows the same question evidence.
            shared = batch / policy.gatekeeper_cache / "question-evidence" / f"{case_id}.json"
            if not shared.exists():
                allowed_ids = question_evidence(client, audits[case_id]["draft"]["findings"])
                shared.parent.mkdir(parents=True, exist_ok=True)
                with (shared.parent / ".lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if not shared.exists():  # another process may have stored it first
                        write_json(shared, allowed_ids)
            allowed_ids = read_json(shared)
            write_json(output / "question-evidence" / f"{case_id}.json", allowed_ids)
            return case_id, allowed_ids

        def run_one(mode, case_id):
            path = output / mode / f"{case_id}.json"
            if path.exists() and "medical_cost" in (saved := read_json(path)):
                try:
                    verify_saved_grade(saved["result"])
                except ValueError as error:
                    raise ValueError(f"Cannot reuse {path}: {error}") from error
                return mode, case_id, saved
            rules = MedicalPolicy(mode, price_batch, diagnostic, playbook, policy, case_id=case_id)
            try:
                result = evaluate_case(cases[case_id], audits[case_id], client, diagnostic=rules,
                                       question_findings=allowed[case_id], policy=policy,
                                       gatekeeper_cache=batch / policy.gatekeeper_cache / f"{case_id}.json",
                                       evidence_log=rules.evidence)
                result["medical_cost"] = rules.total(result["result"])
                result["config"].update(agent_model=args.model, agent=args.agent, agents=args.agents)
            except Exception as error:
                result = {"case_id": case_id, "error": f"{type(error).__name__}: {error}"}
            write_json(path, result)
            return mode, case_id, result

        with (patch.object(cpt_matcher, "create_chat_completion", direct_completion),
              patch.object(llm_estimater, "create_chat_completion", direct_completion),
              ThreadPoolExecutor(max_workers=args.workers) as pricing_pool,
              ThreadPoolExecutor(max_workers=args.workers) as pool):
            matcher = get_cpt_fuzzy_matcher(config)
            if matcher is None:
                raise RuntimeError("Could not initialize SDBench's grounded CPT costing")
            # Bound the pinned matcher's network calls so a stalled lookup can be retried.
            matcher._openai_client.timeout = 45
            matcher._openai_client.max_retries = 1

            # Keyword expansions often repeat across tests. Preserve the upstream scorer and
            # copy results because the matcher adds metadata to each returned dictionary.
            keyword_index = KeywordIndex(matcher._df, matcher._clean_price)
            cached_keywords = lru_cache(maxsize=2000)(keyword_index.search)

            def reuse_keywords(query, top_k=15):
                return [dict(item) for item in cached_keywords(query.lower(), top_k)]

            matcher.keyword_search = reuse_keywords

            def price_batch(names):
                return list(pricing_pool.map(price, names))

            allowed = dict(pool.map(classify, ids))
            jobs = [pool.submit(run_one, mode, i) for i in ids for mode in selected_modes]
            for job in as_completed(jobs):
                mode, i, result = job.result()
                print(mode, i, result.get("error") or
                      (result["result"]["diagnosis_score"], result["medical_cost"]["total_usd"]), flush=True)
    return summarize(output, ids, selected_modes)


if __name__ == "__main__":
    raise SystemExit(main())
