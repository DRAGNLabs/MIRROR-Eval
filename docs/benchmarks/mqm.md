# MT Metric Reliability (MQM)

## About

The MQM benchmark evaluates a translation model by measuring how well automated
MT metrics track the quality of its outputs.

**Why this matters:** When you use BLEU, ChrF, or COMET to evaluate a translation
model, you're assuming those metrics correlate with human quality judgment. That
assumption breaks down in predictable ways — across language resource levels,
domain, and script type. This benchmark makes the reliability of that assumption
explicit for the specific model and languages you care about.

**What it does:**

1. Loads the model under test (`settings.model.model_checkpoint_path`) as a
   seq2seq translation pipeline.
2. Translates FLORES+ English source sentences into each configured target language.
3. Computes automated MT metrics (BLEU, ChrF, COMET) on those translations
   against FLORES+ reference translations.
4. Reports per-language corpus-level metric scores.
5. Annotates each metric score with empirical reliability estimates — Spearman ρ
   and pairwise accuracy vs. professional human judgment — derived from a
   meta-evaluation study across 11 languages, 371k+ segments, and three resource
   tiers (WMT MQM + WMT Direct Assessment annotations).

The reliability estimates answer: *"Given COMET scored this model at 0.82 for
German, how much should I trust that number?"* COMET has Spearman ρ ≈ 0.34 with
human MQM scores for high-resource languages, the highest of any metric evaluated.
BLEU has ρ ≈ 0.11 — still positive but weaker. Williams (1959) tests confirm
COMET's advantage is statistically significant at p ≈ 0 for all languages.

## Requirements

### Model interface

The model under test must satisfy two requirements:

1. **Loadable as a HuggingFace translation pipeline** — the benchmark calls:
   ```python
   pipeline("translation", model=model_checkpoint_path,
            src_lang="eng_Latn", tgt_lang="<nllb_code>")
   ```
   so the model must accept `src_lang` / `tgt_lang` keyword arguments.

2. **Uses NLLB-style language codes** — target languages are specified as
   4-letter script-tagged BCP-47 codes (e.g. `"deu_Latn"`, `"zho_Hans"`,
   `"arb_Arab"`). Models that use different tag formats (e.g. mBART-50's
   `"de_DE"`) are not directly compatible without a wrapper.

The simplest compatible model is `facebook/nllb-200-distilled-600M`. Any
model fine-tuned on top of NLLB-200 that preserves the HuggingFace
`transformers.pipeline` interface will also work.
- `sacrebleu` — for BLEU and ChrF: `pip install sacrebleu`
- `unbabel-comet` — for COMET: `pip install unbabel-comet`
- COMET model weights must be pre-downloaded on a login node before running
  on a compute node with `HF_HUB_OFFLINE=1`:
  ```python
  from comet import download_model
  download_model("Unbabel/wmt22-comet-da")
  ```

## Inputs

| Setting | Description | Default |
|---|---|---|
| `mqm.metrics` | Which metrics to compute (`"bleu"`, `"chrf"`, `"comet"`) | `["bleu", "chrf", "comet"]` |
| `mqm.languages` | Target language codes to evaluate | `["de", "zh", "es", "cs", "tr", "ha", "km"]` |
| `mqm.flores_split` | FLORES+ split to use as source/reference | `"devtest"` |
| `mqm.comet_model` | COMET model HuggingFace ID | `"Unbabel/wmt22-comet-da"` |
| `mqm.translation_batch_size` | Sentences per translation batch | `32` |
| `mqm.output_dir` | Directory to write `mqm_results.json` | `"./results/mqm"` |

Supported language codes: `de`, `zh`, `ru`, `he`, `es`, `cs`, `tr`, `uk`,
`ha`, `km`, `ps`, `sw`, `ht`, `lo`.

## Outputs

Results are written to `{output_dir}/mqm_results.json`:

```json
{
  "model": "facebook/nllb-200-distilled-600M",
  "languages_evaluated": ["de", "zh", "es", "cs", "tr", "ha", "km"],
  "flores_split": "devtest",
  "metrics_run": ["bleu", "chrf", "comet"],
  "metric_scores": {
    "de": {"bleu": 45.2, "chrf": 60.1, "comet": 0.821},
    "zh": {"bleu": 18.3, "chrf": 35.7, "comet": 0.762},
    ...
  },
  "metric_reliability": {
    "bleu": {
      "high":   {"spearman_r": 0.107, "pairwise_acc": 0.544},
      "medium": {"spearman_r": 0.206, "pairwise_acc": 0.569},
      "low":    {"spearman_r": 0.230, "pairwise_acc": 0.576}
    },
    "chrf": {
      "high":   {"spearman_r": 0.143, "pairwise_acc": 0.560},
      ...
    },
    "comet": {
      "high":   {"spearman_r": 0.343, "pairwise_acc": 0.639},
      ...
    }
  },
  "recommendation": "COMET is the most reliable automated signal..."
}
```

### Field descriptions

| Field | Description |
|---|---|
| `metric_scores` | Corpus-level metric averages for the model's translations. BLEU and ChrF are on a 0–100 scale; COMET is roughly 0–1. |
| `metric_reliability.{metric}.{tier}.spearman_r` | Spearman ρ between this metric and human quality judgment, averaged over languages in this resource tier. Higher is better. |
| `metric_reliability.{metric}.{tier}.pairwise_acc` | Fraction of concordant translation pairs when ranked by this metric vs. human judgment. 0.5 = random; 1.0 = perfect agreement. |
| `recommendation` | Plain-English summary of which metric to trust most for this language set. |

Resource tiers in `metric_reliability`: `high` (de, zh, ru, he), `medium` (es,
cs, tr, uk), `low` (ha, km, ps, sw, ht, lo). Only tiers present in the
configured languages are included.
