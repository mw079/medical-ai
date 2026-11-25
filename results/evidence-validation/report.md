# evidence-validation

Policy `reviewed-cpt-bedside-v4`. Runs per arm are in the Runs column.

| Mode | Split | Arm | Runs | Mean accuracy | Range over runs | Avg medical cost | Agent calls | Agent tokens (in / out) | Agent model cost | Δ vs baseline | p |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| budget_1000 | validation | baseline | 3 | 32.5% | 30–35% | $669 | 4.5 | 4,101 / 1,915 | $0.0014 | – | – |
| budget_1000 | validation | evidence_reconcile | 1 | 27.5% | 28–28% | $690 | 11.8 | 16,330 / 9,364 | $0.0066 | -5.0 | 0.56 |
| budget_1000 | validation | evidence_playbook | 1 | 32.5% | 32–32% | $767 | 11.2 | 16,723 / 10,779 | $0.0074 | +0.0 | 1.00 |
| budget_1000 | validation | evidence_memory | 1 | 37.5% | 38–38% | $748 | 12.2 | 31,476 / 12,072 | $0.0099 | +5.0 | 0.53 |
| budget_1000 | validation | panel_team | 1 | 32.5% | 32–32% | $715 | 40.0 | 51,785 / 28,106 | $0.0203 | +0.0 | 1.00 |
| budget_1000 | validation | panel_team_memory | 1 | 25.0% | 25–25% | $786 | 51.2 | 74,490 / 31,200 | $0.0241 | -7.5 | 0.33 |
| budget_4000 | validation | baseline | 3 | 55.0% | 52–57% | $1,599 | 4.3 | 4,068 / 1,827 | $0.0014 | – | – |
| budget_4000 | validation | evidence_reconcile | 1 | 47.5% | 48–48% | $2,185 | 16.0 | 25,502 / 12,706 | $0.0095 | -7.5 | 0.33 |
| budget_4000 | validation | evidence_playbook | 1 | 57.5% | 57–57% | $2,521 | 14.3 | 24,331 / 13,861 | $0.0099 | +2.5 | 0.83 |
| budget_4000 | validation | evidence_memory | 1 | 62.5% | 62–62% | $2,479 | 13.1 | 35,348 / 12,383 | $0.0106 | +7.5 | 0.32 |
| budget_4000 | validation | panel_team | 1 | 45.0% | 45–45% | $2,502 | 51.9 | 78,037 / 34,762 | $0.0269 | -10.0 | 0.08 |
| budget_4000 | validation | panel_team_memory | 1 | 52.5% | 52–52% | $2,582 | 57.7 | 91,350 / 36,393 | $0.0288 | -2.5 | 0.81 |
