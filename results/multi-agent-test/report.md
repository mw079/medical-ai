# multi-agent-test

Policy `reviewed-cpt-bedside-v4`. Runs per arm are in the Runs column.

| Mode | Split | Arm | Runs | Mean accuracy | Range over runs | Avg medical cost | Agent calls | Agent tokens (in / out) | Agent model cost | Δ vs baseline | p |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| budget_1000 | same_source_test | baseline | 3 | 30.0% | 28–34% | $667 | 4.4 | 3,889 / 1,729 | $0.0013 | – | – |
| budget_1000 | same_source_test | panel | 3 | 26.7% | 24–30% | $631 | 4.5 | 6,892 / 2,657 | $0.0022 | -3.3 | 0.54 |
| budget_1000 | same_source_test | independent_3 | 1 | 30.0% | 30–30% | $637 | 16.2 | 14,754 / 5,306 | $0.0041 | +0.0 | 1.00 |
| budget_1000 | same_source_test | independent_5 | 1 | 30.0% | 30–30% | $651 | 24.7 | 23,205 / 9,263 | $0.0068 | +0.0 | 1.00 |
| budget_1000 | same_source_test | decentralized_3 | 1 | 30.0% | 30–30% | $624 | 23.8 | 23,510 / 6,575 | $0.0054 | +0.0 | 1.00 |
| budget_1000 | same_source_test | decentralized_5 | 1 | 26.0% | 26–26% | $611 | 39.6 | 43,168 / 11,864 | $0.0094 | -4.0 | 0.52 |
| budget_1000 | same_source_test | centralized_3 | 1 | 30.0% | 30–30% | $699 | 20.5 | 22,023 / 11,453 | $0.0083 | +0.0 | 1.00 |
| budget_1000 | same_source_test | centralized_5 | 1 | 32.0% | 32–32% | $691 | 31.9 | 36,597 / 17,790 | $0.0132 | +2.0 | 0.76 |
| budget_1000 | external_test | baseline | 3 | 44.0% | 42–46% | $602 | 4.1 | 3,621 / 1,539 | $0.0012 | – | – |
| budget_1000 | external_test | panel | 3 | 46.0% | 44–48% | $583 | 4.2 | 6,210 / 2,239 | $0.0019 | +2.0 | 0.75 |
| budget_1000 | external_test | independent_3 | 1 | 48.0% | 48–48% | $580 | 16.0 | 14,645 / 4,968 | $0.0039 | +4.0 | 0.44 |
| budget_1000 | external_test | independent_5 | 1 | 38.0% | 38–38% | $637 | 25.0 | 22,704 / 8,405 | $0.0063 | -6.0 | 0.39 |
| budget_1000 | external_test | decentralized_3 | 1 | 50.0% | 50–50% | $570 | 23.4 | 23,847 / 5,920 | $0.0051 | +6.0 | 0.38 |
| budget_1000 | external_test | decentralized_5 | 1 | 48.0% | 48–48% | $626 | 38.9 | 41,950 / 10,304 | $0.0086 | +4.0 | 0.50 |
| budget_1000 | external_test | centralized_3 | 1 | 50.0% | 50–50% | $644 | 18.2 | 19,111 / 9,464 | $0.0069 | +6.0 | 0.39 |
| budget_1000 | external_test | centralized_5 | 1 | 44.0% | 44–44% | $630 | 26.2 | 28,112 / 13,305 | $0.0099 | +0.0 | 1.00 |
| budget_1000 | pooled | panel | | | | | | | | -0.7 (95% CI -6.3 to +4.7) | 0.91 |
| budget_1000 | pooled | independent_3 | | | | | | | | +2.0 (95% CI -4.0 to +8.0) | 0.60 |
| budget_1000 | pooled | independent_5 | | | | | | | | -3.0 (95% CI -10.3 to +4.3) | 0.48 |
| budget_1000 | pooled | decentralized_3 | | | | | | | | +3.0 (95% CI -4.0 to +9.7) | 0.45 |
| budget_1000 | pooled | decentralized_5 | | | | | | | | +0.0 (95% CI -6.7 to +6.7) | 1.00 |
| budget_1000 | pooled | centralized_3 | | | | | | | | +3.0 (95% CI -4.7 to +11.0) | 0.50 |
| budget_1000 | pooled | centralized_5 | | | | | | | | +1.0 (95% CI -5.3 to +7.7) | 0.85 |
| budget_4000 | same_source_test | baseline | 3 | 39.3% | 32–46% | $1,634 | 4.6 | 4,409 / 2,006 | $0.0015 | – | – |
| budget_4000 | same_source_test | panel | 3 | 43.3% | 36–52% | $1,632 | 5.0 | 8,451 / 2,964 | $0.0025 | +4.0 | 0.42 |
| budget_4000 | same_source_test | independent_3 | 1 | 34.0% | 34–34% | $1,614 | 16.3 | 15,930 / 5,860 | $0.0045 | -5.3 | 0.43 |
| budget_4000 | same_source_test | independent_5 | 1 | 38.0% | 38–38% | $1,555 | 25.1 | 24,839 / 9,455 | $0.0070 | -1.3 | 0.91 |
| budget_4000 | same_source_test | decentralized_3 | 1 | 36.0% | 36–36% | $1,766 | 26.0 | 28,413 / 7,580 | $0.0063 | -3.3 | 0.65 |
| budget_4000 | same_source_test | decentralized_5 | 1 | 34.0% | 34–34% | $1,532 | 42.6 | 51,759 / 12,940 | $0.0105 | -5.3 | 0.32 |
| budget_4000 | same_source_test | centralized_3 | 1 | 38.0% | 38–38% | $1,801 | 21.8 | 24,692 / 12,261 | $0.0090 | -1.3 | 0.90 |
| budget_4000 | same_source_test | centralized_5 | 1 | 44.0% | 44–44% | $1,883 | 34.9 | 41,404 / 20,207 | $0.0150 | +4.7 | 0.47 |
| budget_4000 | external_test | baseline | 3 | 53.3% | 52–54% | $1,328 | 3.9 | 3,537 / 1,460 | $0.0011 | – | – |
| budget_4000 | external_test | panel | 3 | 62.0% | 60–64% | $1,338 | 4.2 | 6,587 / 2,214 | $0.0019 | +8.7 | 0.06 |
| budget_4000 | external_test | independent_3 | 1 | 60.0% | 60–60% | $1,413 | 14.9 | 14,117 / 4,714 | $0.0037 | +6.7 | 0.33 |
| budget_4000 | external_test | independent_5 | 1 | 52.0% | 52–52% | $1,405 | 23.3 | 22,600 / 7,910 | $0.0060 | -1.3 | 0.91 |
| budget_4000 | external_test | decentralized_3 | 1 | 52.0% | 52–52% | $1,327 | 22.0 | 23,213 / 5,748 | $0.0050 | -1.3 | 0.91 |
| budget_4000 | external_test | decentralized_5 | 1 | 54.0% | 54–54% | $1,304 | 34.6 | 38,436 / 8,863 | $0.0075 | +0.7 | 1.00 |
| budget_4000 | external_test | centralized_3 | 1 | 50.0% | 50–50% | $1,600 | 19.6 | 21,295 / 9,995 | $0.0075 | -3.3 | 0.68 |
| budget_4000 | external_test | centralized_5 | 1 | 52.0% | 52–52% | $1,423 | 26.8 | 30,211 / 13,931 | $0.0105 | -1.3 | 0.91 |
| budget_4000 | pooled | panel | | | | | | | | +6.3 (95% CI +0.7 to +12.0) | 0.04 |
| budget_4000 | pooled | independent_3 | | | | | | | | +0.7 (95% CI -7.3 to +8.7) | 0.93 |
| budget_4000 | pooled | independent_5 | | | | | | | | -1.3 (95% CI -9.0 to +6.3) | 0.80 |
| budget_4000 | pooled | decentralized_3 | | | | | | | | -2.3 (95% CI -10.3 to +5.7) | 0.64 |
| budget_4000 | pooled | decentralized_5 | | | | | | | | -2.3 (95% CI -9.3 to +5.0) | 0.60 |
| budget_4000 | pooled | centralized_3 | | | | | | | | -2.3 (95% CI -10.3 to +5.7) | 0.63 |
| budget_4000 | pooled | centralized_5 | | | | | | | | +1.7 (95% CI -6.0 to +9.7) | 0.75 |
