# baseline-validation

Policy `reviewed-cpt-bedside-v4`. Runs per arm are in the Runs column.

| Mode | Split | Arm | Runs | Mean accuracy | Range over runs | Avg medical cost | Agent calls | Agent tokens (in / out) | Agent model cost | Δ vs baseline | p |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| budget_1000 | validation | baseline | 3 | 32.5% | 30–35% | $669 | 4.5 | 4,101 / 1,915 | $0.0014 | – | – |
| budget_4000 | validation | baseline | 3 | 55.0% | 52–57% | $1,599 | 4.3 | 4,068 / 1,827 | $0.0014 | – | – |
