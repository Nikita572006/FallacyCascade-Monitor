# Early Warning System for LLM Hallucination Cascades

This project implements an explicit, laptop-friendly cascade-risk pipeline:

- spaCy extracts sentence-level claim hooks from dependency parses.
- NetworkX represents chronological factual carryover as a directed acyclic graph.
- `all-MiniLM-L6-v2` supplies normalized semantic similarity features.
- XGBoost learns compact risk features from official HaluEval responses.
- BFS-equivalent descendant traversal identifies downstream claims that inherit an alerted premise.

The implementation is in `hallucination_cascade_detector.py`.

## Important Scientific Boundary

HaluEval's task-specific subsets contain paired right and hallucinated outputs but do not identify sentence-level cascade onset. The current official `general_data.json` contains human labels in `hallucination` and includes `hallucination_spans`; this implementation uses those spans when they can be located in the response.

Accordingly, this detector reports **partially localized, otherwise weakly supervised sentence risk scores** and propagating structural exposure. It cannot prove that a sentence is factually false from graph form or embedding similarity alone. For meaningful deployment, supply source evidence during analysis and evaluate on a dedicated cascade-onset set.

## Corrections Made to the Supplied Blueprint

- Data loads directly from the official `RUCAIBox/HaluEval` release and handles each subset's actual fields, including QA's answer pair and general-set `hallucination`/`hallucination_spans`.
- GPU use is auto-detected for sentence embeddings; it does not crash on a CPU-only install. XGBoost remains on CPU by default because the feature matrix is small.
- The graph contains only observed premise-reuse edges. It does not insert unconditional serial edges that would automatically contaminate every later sentence.
- Training uses evidence-alignment features for grounded HaluEval tasks and holds whole source records out of validation, preventing each right/hallucinated pair from leaking across the split.
- Capped development runs draw a reproducible seeded sample rather than using a possibly ordered dataset prefix.
- Per-sentence samples inherit response labels with equal response weighting, and model output is named a risk score rather than a factuality probability.

## Installation on Windows

Python is not currently available on this machine's command path. Install a current 64-bit CPython release first (Python 3.11 or 3.12 is a conservative choice), then open PowerShell in this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For RTX 4050 acceleration, install the CUDA-enabled PyTorch wheel recommended by the current [PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements. With a CUDA-enabled PyTorch build, `--device auto` uses the GPU for MiniLM embeddings. CPU execution remains supported.

## Train

The default command selects the span-annotated general set plus grounded dialogue and summarization data, caps each at 1,000 source records, and stores a model under `artifacts/cascade_detector`:

```powershell
python .\hallucination_cascade_detector.py train `
  --tasks general dialogue summarization `
  --limit-per-task 1000 `
  --output-dir .\artifacts\cascade_detector `
  --report-json .\artifacts\training_report.json
```

For a quick smoke run, reduce `--limit-per-task` to `100`. For a larger experiment, increase it toward the 10,000 records available in each task-specific subset.

## Analyze a Generation

Analysis is most useful when source evidence is provided:

```powershell
python .\hallucination_cascade_detector.py analyze `
  --model-dir .\artifacts\cascade_detector `
  --text-file .\generation.txt `
  --evidence-file .\retrieved_evidence.txt `
  --alert-threshold 0.65 `
  --output-json .\artifacts\analysis.json
```

Run the embedded illustration after training:

```powershell
python .\hallucination_cascade_detector.py demo `
  --model-dir .\artifacts\cascade_detector
```

## Jupyter Notebook Use

When the source is executed in a notebook cell, Jupyter supplies a kernel connection argument such as `-f ...\kernel-....json`. The script now ignores that injected argument and does not try to begin training until a command is supplied.

Run training as a notebook shell command:

```python
%run ./hallucination_cascade_detector.py train --tasks general dialogue summarization --limit-per-task 100 --output-dir ./artifacts/cascade_detector
```

Or call the analysis API from a cell after training:

```python
from hallucination_cascade_detector import HallucinationCascadeDetector

detector = HallucinationCascadeDetector.load("artifacts/cascade_detector")
result = detector.analyze(
    "The first claim establishes a number. The next claim relies on that number.",
    evidence="Source material supporting or disputing the generated claims.",
)
result
```

## Output

Analysis JSON contains:

- `nodes`: each extracted sentence, its risk score, and alert/cascade flags.
- `edges`: explicit lexical, entity, or numeric dependencies and their overlapping hooks.
- `trigger_indices`: nodes whose learned risk crosses the alert threshold.
- `cascade_indices`: trigger nodes plus dependent descendants in the DAG.
- `cascade_ratio`: the fraction of sentences exposed to an alerted dependency.

## Research Extension Path

For a publishable factuality claim, add sentence- or span-level annotations and an evidence-aware entailment/contradiction component. MiniLM cosine similarity identifies support proximity and topic drift, but similarity alone does not detect a plausible contradiction such as an incorrect number.

## Sources

- Official HaluEval repository and schema: <https://github.com/RUCAIBox/HaluEval>
- HaluEval paper: <https://arxiv.org/abs/2305.11747>
- Sentence Transformers API: <https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html>
- XGBoost GPU configuration: <https://xgboost.readthedocs.io/en/stable/gpu/index.html>
