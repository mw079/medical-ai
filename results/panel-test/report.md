# panel-test

Policy `reviewed-cpt-bedside-v4`. Runs per arm are in the Runs column.

| Mode | Split | Arm | Runs | Mean accuracy | Range over runs | Avg medical cost | Agent calls | Agent tokens (in / out) | Agent model cost | Δ vs baseline | p |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| budget_1000 | same_source_test | baseline | 3 | 30.0% | 28–34% | $667 | 4.4 | 3,889 / 1,729 | $0.0013 | – | – |
| budget_1000 | same_source_test | panel | 3 | 26.7% | 24–30% | $631 | 4.5 | 6,892 / 2,657 | $0.0022 | -3.3 | 0.54 |
| budget_1000 | external_test | baseline | 3 | 44.0% | 42–46% | $602 | 4.1 | 3,621 / 1,539 | $0.0012 | – | – |
| budget_1000 | external_test | panel | 3 | 46.0% | 44–48% | $583 | 4.2 | 6,210 / 2,239 | $0.0019 | +2.0 | 0.75 |
| budget_1000 | pooled | panel | | | | | | | | -0.7 (95% CI -6.3 to +4.7) | 0.91 |
| budget_4000 | same_source_test | baseline | 3 | 39.3% | 32–46% | $1,634 | 4.6 | 4,409 / 2,006 | $0.0015 | – | – |
| budget_4000 | same_source_test | panel | 3 | 43.3% | 36–52% | $1,632 | 5.0 | 8,451 / 2,964 | $0.0025 | +4.0 | 0.42 |
| budget_4000 | external_test | baseline | 3 | 53.3% | 52–54% | $1,328 | 3.9 | 3,537 / 1,460 | $0.0011 | – | – |
| budget_4000 | external_test | panel | 3 | 62.0% | 60–64% | $1,338 | 4.2 | 6,587 / 2,214 | $0.0019 | +8.7 | 0.06 |
| budget_4000 | pooled | panel | | | | | | | | +6.3 (95% CI +0.7 to +12.0) | 0.04 |
