# Validation: choosing the harness (GPT-6 Luna, 40 cases, one run per arm)

Each harness arm ran once against three runs of single GPT-6 Luna. The arm to test was chosen
here, before any test case was run: `evidence_memory`, the only arm above single Luna at both
budgets.

| Arm | $1,000 | $4,000 | Avg medical cost at $4,000 | Model cost per case at $4,000 |
| --- | ---: | ---: | ---: | ---: |
| Single Luna (3 runs) | 32.5% | 55.0% | $1,599 | 0.14¢ |
| Evidence + reconciliation | 27.5% (−5.0) | 47.5% (−7.5) | $2,185 | 0.95¢ |
| + playbook | 32.5% (+0.0) | 57.5% (+2.5) | $2,521 | 0.99¢ |
| **+ playbook and retrieved lessons** | **37.5% (+5.0)** | **62.5% (+7.5)** | $2,479 | 1.06¢ |
| Five-role panel team | 32.5% (+0.0) | 45.0% (−10.0) | $2,502 | 2.69¢ |
| Panel team + memory | 25.0% (−7.5) | 52.5% (−2.5) | $2,582 | 2.88¢ |

## Reading

- **Spending more on tests is not enough.** At $4,000 every harness arm spent about $2,200–2,600
  per case, against $1,599 for single Luna, but only the arm with retrieved lessons beat it.
- **Memory works better in one agent than in a team.** With memory the panel team went from −10.0
  to −2.5, still below single Luna, at nearly 3 times the model cost of `evidence_memory`. In the
  team only Dr. Experience reads the cards and passes a short hint; the single agent reads them
  directly.
- **Test confirmed the choice.** The chosen arm gained +10.3 points at $4,000 on the held-out
  cases (`results/evidence-memory-test/`).

## Limits

One run per arm on 40 cases: no difference here is significant (smallest p = 0.08), and three
identical runs of single Luna already differ by up to 5 points (30–35%, 52–57%). The planned
confirmation runs on validation were skipped.
