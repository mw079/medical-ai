# evidence-memory-test

Policy `reviewed-cpt-bedside-v4`. Runs per arm are in the Runs column.

| Mode | Split | Arm | Runs | Mean accuracy | Range over runs | Avg medical cost | Agent calls | Agent tokens (in / out) | Agent model cost | Δ vs baseline | p |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| budget_1000 | same_source_test | baseline | 3 | 30.0% | 28–34% | $667 | 4.4 | 3,889 / 1,729 | $0.0013 | – | – |
| budget_1000 | same_source_test | evidence_memory | 3 | 22.7% | 20–24% | $734 | 11.3 | 28,700 / 11,632 | $0.0092 | -7.3 | 0.23 |
| budget_1000 | external_test | baseline | 3 | 44.0% | 42–46% | $602 | 4.1 | 3,621 / 1,539 | $0.0012 | – | – |
| budget_1000 | external_test | evidence_memory | 3 | 50.0% | 48–52% | $708 | 11.5 | 28,760 / 10,832 | $0.0088 | +6.0 | 0.25 |
| budget_1000 | pooled | evidence_memory | | | | | | | | -0.7 (95% CI -7.7 to +6.3) | 0.93 |
| budget_4000 | same_source_test | baseline | 3 | 39.3% | 32–46% | $1,634 | 4.6 | 4,409 / 2,006 | $0.0015 | – | – |
| budget_4000 | same_source_test | evidence_memory | 3 | 44.7% | 42–48% | $2,575 | 14.8 | 40,740 / 15,323 | $0.0125 | +5.3 | 0.32 |
| budget_4000 | external_test | baseline | 3 | 53.3% | 52–54% | $1,328 | 3.9 | 3,537 / 1,460 | $0.0011 | – | – |
| budget_4000 | external_test | evidence_memory | 3 | 68.7% | 68–70% | $2,355 | 14.5 | 39,412 / 13,896 | $0.0117 | +15.3 | 0.01 |
| budget_4000 | pooled | evidence_memory | | | | | | | | +10.3 (95% CI +3.7 to +17.0) | 0.01 |
