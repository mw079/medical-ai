# Results

One folder per experiment spec in `experiments/specs/`, in the order they were run. Each holds
`report.md` / `report.json` (accuracy, cost, tokens, paired differences against the baseline)
and `cases.csv` (one row per mode, split, arm, run and case). `NOTES.md` gives the reading
where there is one. Every number in the reports can be recomputed from `cases.csv`:

```sh
python3 experiments/recompute_report.py
```

| Folder | Split | What it shows |
| --- | --- | --- |
| `baseline-validation` | validation | Single GPT-6 Luna under policy v4 at $1,000 and $4,000, 3 runs |
| `panel-test` | test | The four-physician panel prompt vs single Luna: +6.3 at $4,000 (p = 0.04) |
| `model-baselines` | test | GPT-6.1 Terra vs Luna: +8.3 at $1,000, +29.0 at $4,000 |
| `multi-agent-test` | test | Independent, Decentralized, Centralized teams × 3 or 5 agents: all within 3 points |
| `evidence-validation` | validation | Choosing the harness: only evidence + memory beat Luna at both budgets |
| `evidence-memory-test` | test | Final test of the chosen harness: +10.3 at $4,000 (p = 0.01), −0.7 at $1,000 |
