# model-baselines

Policy `reviewed-cpt-bedside-v4`. Runs per arm are in the Runs column.

| Mode | Split | Arm | Runs | Mean accuracy | Range over runs | Avg medical cost | Agent calls | Agent tokens (in / out) | Agent model cost | Δ vs baseline | p |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| budget_1000 | same_source_test | baseline | 3 | 30.0% | 28–34% | $667 | 4.4 | 3,889 / 1,729 | $0.0013 | – | – |
| budget_1000 | same_source_test | terra | 3 | 32.7% | 26–40% | $779 | 4.6 | 4,777 / 1,046 | $0.0107 | +2.7 | 0.65 |
| budget_1000 | external_test | baseline | 3 | 44.0% | 42–46% | $602 | 4.1 | 3,621 / 1,539 | $0.0012 | – | – |
| budget_1000 | external_test | terra | 3 | 58.0% | 50–64% | $717 | 4.2 | 4,202 / 892 | $0.0093 | +14.0 | 0.00 |
| budget_1000 | pooled | terra | | | | | | | | +8.3 (95% CI +2.3 to +14.3) | 0.01 |
| budget_4000 | same_source_test | baseline | 3 | 39.3% | 32–46% | $1,634 | 4.6 | 4,409 / 2,006 | $0.0015 | – | – |
| budget_4000 | same_source_test | terra | 3 | 74.0% | 74–74% | $2,565 | 6.8 | 8,531 / 1,601 | $0.0183 | +34.7 | 0.00 |
| budget_4000 | external_test | baseline | 3 | 53.3% | 52–54% | $1,328 | 3.9 | 3,537 / 1,460 | $0.0011 | – | – |
| budget_4000 | external_test | terra | 3 | 76.7% | 66–84% | $2,046 | 5.3 | 6,249 / 1,156 | $0.0132 | +23.3 | 0.00 |
| budget_4000 | pooled | terra | | | | | | | | +29.0 (95% CI +20.0 to +37.7) | 0.00 |
