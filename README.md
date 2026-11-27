# emotion-classifier

Six-class emotion classification of tweets on [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)
(sadness, joy, love, anger, fear, surprise). Three approaches are compared on the same splits:

| Family | Experiments | Models |
|---|---|---|
| Baselines | E0 | majority class, stratified random |
| Classical | E1, E2 | BoW / TF-IDF × Naive Bayes / logistic regression / linear SVM (+ stop-word and class-weight ablations) |
| Fine-tuned transformers | E3 | `distilbert-base-uncased`, `roberta-base` (3 seeds each) |
| Zero-shot | E4, E5 | `facebook/bart-large-mnli` (NLI), `Qwen/Qwen3-4B-Instruct-2507` (prompted) |
| Learning curve | E6 | linear SVM and DistilBERT on 10 / 25 / 50 / 100 % of train |

Code for a CMP6231 NLP coursework report. Every experiment is driven by a YAML file in `configs/`.

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run pytest            # fast tests, no network, a few seconds
```

Optional checks: `uv run pytest -m slow` (trains a tiny model) and `uv run pytest -m network` (downloads the real dataset).

The first run downloads the dataset (~16 MB) and, for E3–E5, the model weights (RoBERTa and BART ≈ 0.5–1.6 GB, Qwen3-4B ≈ 8 GB) into the Hugging Face cache.

## Hardware used

Apple M5 Pro, 24 GB unified memory, macOS, device `mps` (picked automatically by `device.pick_device()`: CUDA, else MPS, else CPU).
Versions: Python 3.11, torch 2.15, transformers 5.23, datasets 5.2, scikit-learn 1.9 (pinned in `uv.lock`).

Indicative wall time on this machine: the classical grid takes about a minute, DistilBERT about 4–5 min per seed, RoBERTa-base about 9 min per seed,
BART-MNLI about 1.5 min for the test set, Qwen3-4B about 2 min for the test set.

## Reproducing the experiments

Run from the repository root. Each run writes a `RunResult` JSON to `results/` and its predictions to `results/preds/`;
checkpoints go to `outputs/` (git-ignored). Result files are never overwritten, so a re-run adds new files with a new timestamp
(the report uses the latest run per seed).

```
uv run emotion run configs/baselines/majority.yaml           # E0 (also stratified.yaml)
uv run emotion grid configs/classical                        # E1: 12 BoW/TF-IDF × NB/LogReg/SVM × 1-gram/1-2-gram
uv run emotion grid configs/ablations                        # E2: stop words, class weights (linear SVM)
uv run emotion grid configs/transformer                      # E3: DistilBERT and RoBERTa, seeds 13, 42, 7
uv run emotion run configs/zeroshot/bart_mnli.yaml           # E4: NLI zero-shot (test split only)
uv run emotion run configs/llm/qwen3_4b_instruct.yaml        # E5: instruct-LLM zero-shot (test split only)
uv run emotion grid configs/learning_curve                   # E6: 10/25/50/100 % of train
```

Single configs also work, e.g. `uv run emotion run configs/classical/tfidf_logreg_12.yaml`.

Then regenerate every table and figure from `results/` (nothing is re-trained):

```
uv run emotion report                                        # results/summary.{csv,md} + figures/*.png
uv run emotion report --select Transformer=roberta_base      # show a different model in the figures
```

`notebooks/02_results.ipynb` does the same and adds the most common confusions with example tweets and the
top-weighted n-grams of the best linear model. `notebooks/01_eda.ipynb` is the dataset exploration.

## Evaluation protocol

- Hyperparameters and models are chosen on the **validation** split only. The test split is used for reporting.
- Zero-shot models (E4, E5) have nothing to tune and are evaluated on test only; their prompt and label template are fixed in the config.
- The figures show, per family, the model with the best validation macro-F1 (override with `--select`).
- Primary metric: macro-F1 (the classes are imbalanced). Accuracy and weighted-F1 are also reported.
- Cost: training seconds and inference seconds per 1,000 examples, measured on the hardware above.
- E5 counts an answer that is not exactly one label as wrong (prediction `-1`) and reports it as the invalid rate.

## Reproducibility notes

- Seeds are fixed in the configs (classical and zero-shot: 42; transformers: 13, 42, 7).
- Classical models are deterministic. **Training on MPS is not bit-reproducible**: re-running the same seed can move
  macro-F1 by roughly ±0.007, so re-running E3 will give close but not identical numbers. Greedy LLM decoding is
  affected in the same way.
- E6 uses a single seed (42) and a fixed 4-epoch budget for DistilBERT at every fraction, which understates it at 10 %.

## Layout

```
configs/      experiment YAML (one folder per family)
src/emotion_classifier/   data, features, classical, transformer, zeroshot, metrics, results, report, plots, analysis, cli
tests/        pytest suite (fast by default)
notebooks/    01_eda, 02_results
results/      RunResult JSON, predictions CSV, summary tables (committed)
figures/      generated PNGs (committed)
outputs/      checkpoints (git-ignored)
```

## Dataset

`dair-ai/emotion` (the "split" configuration: 16,000 train / 2,000 validation / 2,000 test), built from English Twitter messages
labelled by distant supervision from emotion hashtags (the hashtags are removed from the text).
The dataset card gives the licence as "other" and states that it **should be used for educational and research purposes only**.

> Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). CARER: Contextualized Affect Representations for Emotion Recognition.
> *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, Brussels, Belgium. ACL. https://www.aclweb.org/anthology/D18-1404

## Models

- DistilBERT: Sanh et al. (2019), `distilbert-base-uncased`
- RoBERTa: Liu et al. (2019), `roberta-base`
- BART-large-MNLI: Lewis et al. (2020), `facebook/bart-large-mnli`, used through the Hugging Face zero-shot-classification pipeline (Yin et al., 2019)
- Qwen3-4B-Instruct-2507: Qwen Team, `Qwen/Qwen3-4B-Instruct-2507`; check the model card for its licence

## Tutorials and other references

<!-- TODO (Naresh): list any tutorials, blog posts or course materials you followed, with links. None are recorded in the repo. -->
Libraries: scikit-learn, Hugging Face `transformers` / `datasets`, PyTorch, Typer, Matplotlib.
