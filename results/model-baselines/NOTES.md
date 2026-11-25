# Model baselines: GPT-6 Luna vs GPT-6.1 Terra (held-out, 100 cases)

SDBench's baseline agent, same prompt and environment, medium reasoning. Gatekeeper and judge
are GPT-6 Luna for both. 3 runs per arm. All 12 Terra runs finished on the first attempt with
no errors. Luna's runs are those of `panel-test`.

| Mode | Luna | Terra | Δ pooled (95% CI) | p |
| --- | ---: | ---: | --- | ---: |
| $1,000 | 37.0% | 45.3% | +8.3 (+2.3 to +14.3) | 0.01 |
| $4,000 | 46.3% | **75.3%** | **+29.0** (+20.0 to +37.7) | <0.001 |

Accuracies are the mean of the two test splits. Terra was ahead on both splits in both modes.

## What changes with the model

- **Terra buys more evidence.** At $4,000 it spent $2,565 and $2,046 per case against Luna's
  $1,634 and $1,328, and took more turns (6.8 vs 4.6 agent calls on the Stanford split). Luna
  diagnoses early and leaves most of the budget unused; Terra uses it.
- **The budget matters more for Terra.** Luna gains 9 points from $1,000 to $4,000; Terra
  gains 30.
- **Model cost stays small.** Terra cost $0.009–0.018 per case in tokens, about 8–12 times
  Luna's $0.001–0.002, and still under 2 cents. Terra wrote fewer output tokens per case
  than Luna.
- **The judge is GPT-6 Luna**, so it is not grading its own answers when Terra is the agent.
  If it favoured its own style, that would work against Terra, not for it.

Reproduce: `python3 experiments/run_arms.py experiments/specs/model_baselines.json`
