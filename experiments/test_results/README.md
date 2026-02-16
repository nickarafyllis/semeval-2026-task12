# Test Set Results (612 questions)

Results from Table 4 of the paper. Scoring: partial-credit (exact=1.0, subset=0.5, else=0.0).

| Configuration | Base | + Post-hoc | Directory |
|:---|:---:|:---:|:---|
| Claude Sonnet 4.5 Thinking | 0.904 | **0.952** | `claude_sonnet_4.5_thinking/` |
| GPT-5.2 | 0.912 | 0.950 | `gpt_5.2/` |
| Gemini 3 Flash Preview | 0.907 | 0.940 | `gemini_3_flash/` |
| SC: Sonnet 3× (θ=0.50) | 0.902 | 0.941 | `sc_sonnet_3x/` |
| SC: Gemini 5× (θ=0.50) | 0.902 | 0.927 | `sc_gemini_5x/` |
| Ensemble (Sonnet + GPT + Gemini) | 0.935 | 0.946 | `ensemble/` |

## Files per experiment

- `metadata.json` — Experiment configuration and metadata
- `results.json` — Base model predictions (612 questions)
- `results_verified_cascading.json` — Post-hoc consistency enforcement results (where applicable)
- `dashboard.html` — Interactive visualization (open in any browser)
- `submission.jsonl` / `submission.zip` — Codabench submission format
