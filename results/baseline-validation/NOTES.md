# Baseline: single GPT-6 Luna (validation, 40 cases)

| Mode | Runs | Mean accuracy | Range | Avg cost |
| --- | ---: | ---: | --- | ---: |
| $1,000 | 3 | 32.5% | 30–35% | $669 |
| $4,000 | 3 | **55.0%** | 52–57% | $1,599 |

All 240 encounters finished on the first attempt, with no errors. None reached the 10-turn
limit. The most turns used was 7 at $1,000 and 8 at $4,000.

## Reading

- **Money is the main lever.** Accuracy rises from 32.5% at $1,000 to 55% at $4,000. The $4,000
  baseline solves 17 cases in all 3 runs, the $1,000 baseline 10.
- **Ceiling from the cases:** 13 cases are never solved even at $4,000. That points to limits
  in the cases themselves (at most 8 findings), not in the agent's instructions.
- This is the reference every validation comparison uses (`results/evidence-validation/`).

Reproduce: `python3 experiments/run_arms.py experiments/specs/baseline_validation.json`
