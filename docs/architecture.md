# Architecture Overview

The system represents generated text as a sentence-level dependency graph.

1. spaCy extracts sentence claims and lexical hooks.
2. NetworkX builds a directed acyclic graph from reused entities, numbers, and claim terms.
3. Sentence Transformers generate semantic similarity features.
4. XGBoost predicts weakly supervised sentence-level risk.
5. Breadth-first traversal marks downstream claims exposed to a risky premise.

The detector is an early-warning system, not a factuality oracle.
