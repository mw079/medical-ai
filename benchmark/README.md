# Benchmark: 200 sequential-diagnosis cases

SDBench-format cases built from published case reports by `curation/`
(`python3 curation/export_benchmark.py` writes this folder).

| Split | Cases | Source | Use |
| --- | ---: | --- | --- |
| `development` | 60 | MedCaseReasoning | building; source of the memory cards |
| `validation` | 40 | MedCaseReasoning | choosing between designs |
| `same_source_test` | 50 | MedCaseReasoning | final test |
| `external_test` | 50 | MedR-Bench | final test |

**Files.** `<split>.json` holds the `cases`, each case's curation record (`audit`: findings and
diagnosis with their source quotes), its `blind_review`, and the shared `gatekeeper_cache` and
`question_evidence`, so every run sees the same gatekeeper answers. Runs never write here:
`evaluation/release.py` copies what they need to `data/benchmark/`. `cases.csv` lists each case's
source article and license screen. `prices.json.gz` holds 9,563 reviewed test prices, so reruns
price repeated orders identically.

## License and attribution

This folder is licensed under [CC BY-SA 4.0](LICENSE). The rest of the repository is MIT.

- **Source articles.** Every case is adapted from one case report in the PubMed Central
  open-access subset under a CC BY license. `cases.csv` links each article; its page names the
  authors and the exact CC BY license.
- **MedCaseReasoning** (Wu et al., 2025, [arXiv:2505.11733](https://arxiv.org/abs/2505.11733)),
  released under the MIT License on
  [Hugging Face](https://huggingface.co/datasets/zou-lab/MedCaseReasoning), supplied the
  `development`, `validation` and `same_source_test` reports and their diagnosis labels.
- **MedR-Bench** (Qiu et al., Nature Communications 2025,
  [doi:10.1038/s41467-025-64769-1](https://doi.org/10.1038/s41467-025-64769-1)), released under
  CC BY-SA on [GitHub](https://github.com/MAGIC-AI4Med/MedRBench), supplied the `external_test`
  reports and their diagnosis labels. Its ShareAlike term is why this folder is CC BY-SA.
- **Changes made.** Each case was rewritten from its source by language models (Grok 4.7 as
  writer and first reviewer, GPT-6 Luna as blind reviewer) into an opening, up to 8 findings with
  verbatim source quotes, and a diagnosis. The full record of each case's changes is in `audit`.

The cases are not medical advice and are not validated for clinical use.
