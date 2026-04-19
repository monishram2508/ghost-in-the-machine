# Ghost in the Machine: AI Text Detection

**Author:** Monishram Selvaraj

This project studies whether machine-generated writing leaves detectable stylistic traces, and how robust those traces are when generation is adapted to imitate human style.

## What this project does

The workflow builds an end-to-end authorship detection pipeline:

1. Prepare human text from books
2. Extract topic prompts
3. Generate AI text (generic + stylized)
4. Build balanced datasets
5. Extract linguistic/statistical fingerprints
6. Train multiple detector tiers
7. Interpret model behavior (SHAP)
8. Attempt adversarial evasion with a genetic algorithm

## Core conclusions from the project

- **Statistical fingerprints are real and useful.** A classical feature-based classifier (Tier A) reached about **93.18% accuracy** on the prepared split.
- **Generalization is fragile when leakage is controlled.** The embedding NN setup (Tier B, topic-aware split) dropped to roughly **50.9% test accuracy**, showing that apparent performance can collapse when topic overlap is reduced.
- **Models can rely on style cues more than true authorship intent.** Across experiments, genre/topic/style artifacts can dominate predictions.
- **Interpretability is essential.** SHAP analysis is included to inspect token-level signals driving detector outputs.
- **Detectors can be pressured adversarially.** The “Super-Imposter” stage uses iterative mutation (Gemini + detector feedback) to push AI text toward higher “human” confidence, illustrating an evasion risk.

## Repository layout

- `notebooks/overview.ipynb` — main end-to-end notebook for all tasks
- `src/` — script version of each task stage
- `models/distilbert_lora_tuned/` — saved LoRA adapter artifacts
- `precog_report.pdf` — project report
- `requirements.txt` — Python dependencies

## Reproducibility

### 1) Environment

```bash
pip install -r requirements.txt
```

### 2) API key (for generation/evasion stages)

Set Gemini credentials in your environment (or `.env` where applicable):

```bash
export GEMINI_API_KEY="<your_key>"
```

### 3) Data dependencies

Some large assets are not versioned in Git (for example embeddings and generated corpora).

- Download GloVe file used by Tier B from:  
  https://nlp.stanford.edu/data/wordvecs/glove.2024.dolma.300d.zip
- Place the embedding text file at:  
  `data/dolma_300_2024_1.2M.100_combined.txt`

### 4) Run

Recommended path:

```bash
jupyter notebook notebooks/overview.ipynb
```

Script path (stage-by-stage) is available in `src/` using numbered files (`01_...` to `41_...`).

## Notes and limitations

- Reported metrics are split- and setup-dependent; they should be interpreted as experimental findings, not universal detector performance.
- Generated datasets, API behavior, and model checkpoints can change over time, so exact reproducibility may vary.
- The included LoRA model card is still template-level and can be expanded with full training/evaluation metadata.

## Final takeaway

This project suggests that AI text detection can work well under favorable conditions, but robustness drops when confounds are controlled and when adversarial rewriting is introduced. Reliable deployment needs careful split design, interpretability checks, and explicit evaluation against adaptive attackers.
