"""Early-warning detector for structurally propagating hallucination risk.

This module builds sentence-level dependency DAGs directly from spaCy parses,
derives structural and semantic features, and trains an XGBoost risk model on
HaluEval. General-set hallucination spans support partial localization, while
task-specific examples are response-labeled: node predictions are risk scores,
not factuality proofs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import spacy
import torch
import xgboost as xgb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


LOGGER = logging.getLogger("cascade_detector")

HALUEVAL_URLS = {
    "qa": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
    "dialogue": (
        "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/"
        "dialogue_data.json"
    ),
    "summarization": (
        "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/"
        "summarization_data.json"
    ),
    "general": (
        "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/"
        "general_data.json"
    ),
}

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "dative", "pobj"}
NOMINAL_MODIFIER_DEPS = {
    "amod",
    "appos",
    "compound",
    "flat",
    "nmod",
    "nummod",
    "poss",
    "quantmod",
}
NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")

FEATURE_NAMES = (
    "position_norm",
    "token_count_log",
    "subject_term_count_log",
    "object_term_count_log",
    "entity_term_count_log",
    "number_count_log",
    "new_number_ratio",
    "reused_number_ratio",
    "in_degree_norm",
    "out_degree_norm",
    "dependency_depth_norm",
    "parent_overlap_norm",
    "parent_similarity_max",
    "previous_similarity",
    "history_similarity_max",
    "evidence_similarity_max",
    "has_evidence",
)


@dataclass(frozen=True, slots=True)
class FactNode:
    """One parsed candidate claim corresponding to one response sentence."""

    index: int
    text: str
    start_char: int
    end_char: int
    relation: str
    subject_terms: tuple[str, ...]
    object_terms: tuple[str, ...]
    entity_terms: tuple[str, ...]
    content_terms: tuple[str, ...]
    numbers: tuple[str, ...]
    token_count: int


@dataclass(frozen=True, slots=True)
class DocumentSample:
    """A response-level HaluEval supervision record."""

    text: str
    label: int
    evidence: str | None
    group: str
    task: str
    hallucination_spans: tuple[str, ...] = ()


@dataclass(slots=True)
class FeaturizedBatch:
    features: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    spans: list[tuple[int, int]]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _as_label(value: Any) -> int:
    return int(str(value).strip().casefold() in {"yes", "true", "1", "hallucinated"})


def _safe_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    if len(set(int(value) for value in y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(
    y_true: Sequence[int], y_score: Sequence[float]
) -> float | None:
    if len(set(int(value) for value in y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


class HallucinationCascadeDetector:
    """Constructs claim DAGs and identifies high-risk dependent claims."""

    def __init__(
        self,
        *,
        spacy_model: str = "en_core_web_sm",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        xgb_device: str = "cpu",
        embedding_batch_size: int = 64,
        max_lookback: int = 12,
        max_evidence_sentences: int = 48,
        random_state: int = 42,
    ) -> None:
        self.spacy_model = spacy_model
        self.embedding_model_name = embedding_model
        self.device_request = device
        self.device = self._resolve_device(device)
        self.xgb_device = xgb_device
        self.embedding_batch_size = embedding_batch_size
        self.max_lookback = max_lookback
        self.max_evidence_sentences = max_evidence_sentences
        self.random_state = random_state

        LOGGER.info("Loading spaCy parser '%s' on CPU.", spacy_model)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{spacy_model}' is unavailable. Install it with: "
                f"python -m spacy download {spacy_model}"
            ) from exc

        self.evidence_segmenter = spacy.blank("en")
        self.evidence_segmenter.add_pipe("sentencizer")

        LOGGER.info(
            "Loading sentence transformer '%s' on %s.",
            embedding_model,
            self.device,
        )
        self.vector_model = SentenceTransformer(embedding_model, device=self.device)
        self.classifier: xgb.XGBClassifier | None = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        normalized = requested.casefold()
        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch does not report an available CUDA "
                "device. Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        return requested

    @staticmethod
    def _normalize_token(token: Any) -> str | None:
        if token.is_space or token.is_punct:
            return None
        if token.is_stop and not token.like_num:
            return None
        lemma = (token.lemma_ or token.text).strip().casefold()
        if not lemma or lemma == "-pron-":
            lemma = token.text.strip().casefold()
        if len(lemma) == 1 and not token.like_num:
            return None
        return lemma

    def _terms(self, tokens: Iterable[Any]) -> set[str]:
        terms: set[str] = set()
        for token in tokens:
            term = self._normalize_token(token)
            if term:
                terms.add(term)
        return terms

    def _nominal_terms(self, head: Any) -> set[str]:
        collected = [head]
        frontier = [head]
        while frontier:
            current = frontier.pop()
            for child in current.children:
                if child.dep_ in NOMINAL_MODIFIER_DEPS:
                    collected.append(child)
                    frontier.append(child)
        return self._terms(collected)

    def _extract_fact_nodes_from_doc(self, doc: Any) -> list[FactNode]:
        facts: list[FactNode] = []
        for index, sentence in enumerate(doc.sents):
            sentence_text = sentence.text.strip()
            if not sentence_text:
                continue

            roots = [token for token in sentence if token.dep_ == "ROOT"]
            root = roots[0] if roots else None
            subject_heads = [token for token in sentence if token.dep_ in SUBJECT_DEPS]
            object_heads = [token for token in sentence if token.dep_ in OBJECT_DEPS]

            subject_terms: set[str] = set()
            for token in subject_heads:
                subject_terms.update(self._nominal_terms(token))

            object_terms: set[str] = set()
            for token in object_heads:
                object_terms.update(self._nominal_terms(token))

            content_terms = self._terms(sentence)
            noun_terms = self._terms(
                token for token in sentence if token.pos_ in {"NOUN", "PROPN", "NUM"}
            )
            if not subject_terms:
                subject_terms.update(noun_terms)
            if not object_terms:
                object_terms.update(content_terms.difference(subject_terms))

            entity_terms: set[str] = set()
            for entity in doc.ents:
                if entity.start >= sentence.start and entity.end <= sentence.end:
                    entity_terms.update(self._terms(entity))

            numbers = tuple(
                sorted({match.group(0).replace(",", "") for match in NUMBER_RE.finditer(sentence_text)})
            )
            facts.append(
                FactNode(
                    index=len(facts),
                    text=sentence_text,
                    start_char=sentence.start_char,
                    end_char=sentence.end_char,
                    relation=(root.lemma_.casefold() if root else ""),
                    subject_terms=tuple(sorted(subject_terms)),
                    object_terms=tuple(sorted(object_terms)),
                    entity_terms=tuple(sorted(entity_terms)),
                    content_terms=tuple(sorted(content_terms)),
                    numbers=numbers,
                    token_count=sum(not token.is_space for token in sentence),
                )
            )
        return facts

    def extract_fact_nodes(self, text: str) -> list[FactNode]:
        """Parse response text into sentence-level claim candidates."""

        return self._extract_fact_nodes_from_doc(self.nlp(text))

    def build_dependency_graph(self, facts: Sequence[FactNode]) -> nx.DiGraph:
        """Build a chronological DAG using explicit lexical/entity carryover.

        Edges are not added merely because two sentences are adjacent. This keeps
        cascade propagation tied to observed premise reuse instead of marking the
        rest of every paragraph contaminated after any alert.
        """

        graph = nx.DiGraph()
        for fact in facts:
            graph.add_node(fact.index, fact=fact, text=fact.text)

        for target_index, target in enumerate(facts):
            start = max(0, target_index - self.max_lookback)
            target_hooks = (
                set(target.subject_terms)
                | set(target.entity_terms)
                | set(target.content_terms)
                | set(target.numbers)
            )
            for source_index in range(start, target_index):
                source = facts[source_index]
                source_hooks = (
                    set(source.object_terms)
                    | set(source.entity_terms)
                    | set(source.numbers)
                )
                overlap = sorted(source_hooks.intersection(target_hooks))
                if overlap:
                    graph.add_edge(
                        source.index,
                        target.index,
                        kind="lexical_dependency",
                        overlap=overlap,
                        weight=float(len(overlap)),
                    )
        return graph

    def _segment_evidence(self, evidence: str | None) -> list[str]:
        if not evidence:
            return []
        doc = self.evidence_segmenter(evidence)
        segments = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]
        return segments[: self.max_evidence_sentences]

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dimension = self.vector_model.get_sentence_embedding_dimension()
            return np.empty((0, int(dimension or 0)), dtype=np.float32)
        vectors = self.vector_model.encode(
            list(texts),
            batch_size=self.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def _dependency_depths(graph: nx.DiGraph) -> dict[int, int]:
        depths: dict[int, int] = {}
        for node in nx.topological_sort(graph):
            parents = list(graph.predecessors(node))
            depths[node] = 0 if not parents else 1 + max(depths[parent] for parent in parents)
        return depths

    def compute_graph_features(
        self,
        graph: nx.DiGraph,
        sentence_embeddings: np.ndarray,
        evidence_embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute numeric risk-model input for each node in a claim DAG."""

        node_indices = list(sorted(graph.nodes()))
        if not node_indices:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        if sentence_embeddings.shape[0] != len(node_indices):
            raise ValueError("One sentence embedding is required for every graph node.")

        position_by_node = {node: row for row, node in enumerate(node_indices)}
        node_scale = float(max(len(node_indices) - 1, 1))
        out_degree = nx.out_degree_centrality(graph)
        in_degree = nx.in_degree_centrality(graph)
        depths = self._dependency_depths(graph)
        has_evidence = bool(evidence_embeddings is not None and evidence_embeddings.size)
        seen_numbers: set[str] = set()
        rows: list[list[float]] = []

        for position, node in enumerate(node_indices):
            fact: FactNode = graph.nodes[node]["fact"]
            parents = list(graph.predecessors(node))
            embedding = sentence_embeddings[position]

            parent_similarities = [
                float(embedding @ sentence_embeddings[position_by_node[parent]])
                for parent in parents
            ]
            parent_similarity = max(parent_similarities, default=0.0)
            previous_similarity = (
                float(embedding @ sentence_embeddings[position - 1])
                if position > 0
                else 1.0
            )
            history_similarity = (
                float(np.max(sentence_embeddings[:position] @ embedding))
                if position > 0
                else 1.0
            )
            evidence_similarity = (
                float(np.max(evidence_embeddings @ embedding)) if has_evidence else 0.0
            )

            numbers = set(fact.numbers)
            number_denominator = float(max(len(numbers), 1))
            new_number_ratio = len(numbers.difference(seen_numbers)) / number_denominator
            reused_number_ratio = len(numbers.intersection(seen_numbers)) / number_denominator
            seen_numbers.update(numbers)

            overlap_count = sum(
                len(graph.edges[parent, node].get("overlap", [])) for parent in parents
            )
            rows.append(
                [
                    position / node_scale,
                    math.log1p(fact.token_count),
                    math.log1p(len(fact.subject_terms)),
                    math.log1p(len(fact.object_terms)),
                    math.log1p(len(fact.entity_terms)),
                    math.log1p(len(fact.numbers)),
                    new_number_ratio,
                    reused_number_ratio,
                    in_degree[node],
                    out_degree[node],
                    depths[node] / node_scale,
                    overlap_count / float(max(fact.token_count, 1)),
                    parent_similarity,
                    previous_similarity,
                    history_similarity,
                    evidence_similarity,
                    float(has_evidence),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def _graph_and_features(
        self, text: str, evidence: str | None
    ) -> tuple[nx.DiGraph, np.ndarray]:
        facts = self.extract_fact_nodes(text)
        graph = self.build_dependency_graph(facts)
        if not facts:
            return graph, np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

        evidence_sentences = self._segment_evidence(evidence)
        all_embeddings = self._encode([fact.text for fact in facts] + evidence_sentences)
        sentence_embeddings = all_embeddings[: len(facts)]
        evidence_embeddings = all_embeddings[len(facts) :] if evidence_sentences else None
        return graph, self.compute_graph_features(
            graph, sentence_embeddings, evidence_embeddings
        )

    @staticmethod
    def _node_training_labels(
        sample: DocumentSample, facts: Sequence[FactNode]
    ) -> list[int]:
        """Localize general-set spans; otherwise inherit the response label."""

        if not sample.label or not sample.hallucination_spans:
            return [sample.label] * len(facts)

        intervals: list[tuple[int, int]] = []
        for span in sample.hallucination_spans:
            start = sample.text.find(span)
            while start >= 0:
                intervals.append((start, start + len(span)))
                start = sample.text.find(span, start + 1)

        if not intervals:
            return [sample.label] * len(facts)

        localized = [
            int(
                any(
                    max(fact.start_char, start) < min(fact.end_char, end)
                    for start, end in intervals
                )
            )
            for fact in facts
        ]
        return localized if any(localized) else [sample.label] * len(facts)

    def _featurize_samples(
        self, samples: Sequence[DocumentSample], chunk_size: int = 96
    ) -> FeaturizedBatch:
        feature_chunks: list[np.ndarray] = []
        labels: list[int] = []
        weights: list[float] = []
        spans: list[tuple[int, int]] = []
        cursor = 0

        for offset in range(0, len(samples), chunk_size):
            sample_chunk = samples[offset : offset + chunk_size]
            response_docs = list(
                self.nlp.pipe((sample.text for sample in sample_chunk), batch_size=chunk_size)
            )
            facts_per_sample = [
                self._extract_fact_nodes_from_doc(doc) for doc in response_docs
            ]
            graphs = [self.build_dependency_graph(facts) for facts in facts_per_sample]
            sentence_texts = [
                fact.text for facts in facts_per_sample for fact in facts
            ]

            evidence_per_sample = [
                self._segment_evidence(sample.evidence) for sample in sample_chunk
            ]
            evidence_texts = [
                text for evidence_segments in evidence_per_sample for text in evidence_segments
            ]
            embeddings = self._encode(sentence_texts + evidence_texts)

            sentence_cursor = 0
            evidence_cursor = len(sentence_texts)
            for sample, facts, graph, evidence_segments in zip(
                sample_chunk, facts_per_sample, graphs, evidence_per_sample
            ):
                sentence_count = len(facts)
                evidence_count = len(evidence_segments)
                if not sentence_count:
                    spans.append((cursor, cursor))
                    evidence_cursor += evidence_count
                    continue

                sentence_vectors = embeddings[
                    sentence_cursor : sentence_cursor + sentence_count
                ]
                evidence_vectors = (
                    embeddings[evidence_cursor : evidence_cursor + evidence_count]
                    if evidence_count
                    else None
                )
                features = self.compute_graph_features(
                    graph, sentence_vectors, evidence_vectors
                )
                feature_chunks.append(features)
                labels.extend(self._node_training_labels(sample, facts))
                weights.extend([1.0 / sentence_count] * sentence_count)
                spans.append((cursor, cursor + sentence_count))
                cursor += sentence_count
                sentence_cursor += sentence_count
                evidence_cursor += evidence_count

        matrix = (
            np.vstack(feature_chunks)
            if feature_chunks
            else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        )
        return FeaturizedBatch(
            features=matrix,
            labels=np.asarray(labels, dtype=np.int32),
            weights=np.asarray(weights, dtype=np.float32),
            spans=spans,
        )

    @staticmethod
    def _balance_weights(labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
        adjusted = weights.copy()
        positive_weight = float(adjusted[labels == 1].sum())
        negative_weight = float(adjusted[labels == 0].sum())
        if positive_weight and negative_weight:
            adjusted[labels == 1] *= negative_weight / positive_weight
        return adjusted

    def _new_classifier(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=self.xgb_device,
            max_depth=5,
            min_child_weight=2.0,
            n_estimators=400,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=self.random_state,
            early_stopping_rounds=30,
        )

    def train(
        self,
        samples: Sequence[DocumentSample],
        *,
        validation_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train from response-labeled samples with group-isolated validation."""

        if len(samples) < 10:
            raise ValueError("At least 10 response samples are required for training.")
        labels = {sample.label for sample in samples}
        if labels != {0, 1}:
            raise ValueError("Training requires both clean and hallucinated samples.")

        groups = np.asarray([sample.group for sample in samples])
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=self.random_state,
        )
        train_index, validation_index = next(
            splitter.split(np.zeros(len(samples)), groups=groups)
        )
        training_samples = [samples[index] for index in train_index]
        validation_samples = [samples[index] for index in validation_index]

        LOGGER.info("Featurizing %d training responses.", len(training_samples))
        training = self._featurize_samples(training_samples)
        LOGGER.info("Featurizing %d validation responses.", len(validation_samples))
        validation = self._featurize_samples(validation_samples)
        if not training.features.size or not validation.features.size:
            raise RuntimeError("Training or validation data produced no sentence features.")

        self.classifier = self._new_classifier()
        training_weights = self._balance_weights(training.labels, training.weights)
        validation_weights = self._balance_weights(validation.labels, validation.weights)
        self.classifier.fit(
            training.features,
            training.labels,
            sample_weight=training_weights,
            eval_set=[(validation.features, validation.labels)],
            sample_weight_eval_set=[validation_weights],
            verbose=False,
        )

        node_scores = self.classifier.predict_proba(validation.features)[:, 1]
        document_labels: list[int] = []
        document_scores: list[float] = []
        for sample, (start, end) in zip(validation_samples, validation.spans):
            if end > start:
                document_labels.append(sample.label)
                document_scores.append(float(np.max(node_scores[start:end])))

        return {
            "training_responses": len(training_samples),
            "validation_responses": len(validation_samples),
            "training_nodes": int(training.features.shape[0]),
            "validation_nodes": int(validation.features.shape[0]),
            "validation_node_weak_auc": _safe_auc(validation.labels, node_scores),
            "validation_node_weak_average_precision": _safe_average_precision(
                validation.labels, node_scores
            ),
            "validation_document_max_auc": _safe_auc(
                document_labels, document_scores
            ),
            "validation_document_max_average_precision": _safe_average_precision(
                document_labels, document_scores
            ),
            "span_annotated_responses": sum(
                bool(sample.hallucination_spans) for sample in samples
            ),
            "feature_names": list(FEATURE_NAMES),
            "supervision_note": (
                "For HaluEval general records, locatable supplied hallucination "
                "spans provide localized labels. Task-specific records and "
                "unlocatable spans use weak response-inherited node labels."
            ),
        }

    def analyze(
        self,
        text: str,
        *,
        evidence: str | None = None,
        alert_threshold: float = 0.65,
    ) -> dict[str, Any]:
        """Score sentence claims and propagate alerts along dependency edges."""

        if self.classifier is None:
            raise RuntimeError("Load or train a classifier before analysis.")
        if not 0.0 <= alert_threshold <= 1.0:
            raise ValueError("alert_threshold must be between 0 and 1.")

        graph, features = self._graph_and_features(text, evidence)
        if not graph.nodes:
            return {
                "nodes": [],
                "edges": [],
                "trigger_indices": [],
                "cascade_indices": [],
                "cascade_ratio": 0.0,
                "message": "No sentence claims were extracted.",
            }

        risk_scores = self.classifier.predict_proba(features)[:, 1]
        trigger_indices = [
            int(index)
            for index, score in enumerate(risk_scores)
            if score >= alert_threshold
        ]
        cascade_indices: set[int] = set(trigger_indices)
        for trigger in trigger_indices:
            cascade_indices.update(nx.bfs_tree(graph, source=trigger).nodes())

        nodes = []
        for node, score in zip(sorted(graph.nodes()), risk_scores):
            fact: FactNode = graph.nodes[node]["fact"]
            nodes.append(
                {
                    "index": node,
                    "text": fact.text,
                    "risk_score": float(score),
                    "trigger": node in trigger_indices,
                    "cascade_affected": node in cascade_indices,
                    "fact": asdict(fact),
                }
            )
        edges = [
            {
                "source": int(source),
                "target": int(target),
                "kind": data["kind"],
                "overlap": data["overlap"],
                "weight": data["weight"],
            }
            for source, target, data in graph.edges(data=True)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "trigger_indices": trigger_indices,
            "cascade_indices": sorted(cascade_indices),
            "cascade_ratio": len(cascade_indices) / float(graph.number_of_nodes()),
            "alert_threshold": alert_threshold,
            "interpretation": (
                "Scores indicate learned structural/semantic risk under partially "
                "localized and otherwise weak response-level supervision; they do "
                "not independently establish whether a factual claim is true."
            ),
        }

    def save(self, directory: str | Path, training_report: dict[str, Any]) -> None:
        if self.classifier is None:
            raise RuntimeError("There is no trained classifier to save.")
        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(str(output_directory / "cascade_xgb.ubj"))
        metadata = {
            "spacy_model": self.spacy_model,
            "embedding_model": self.embedding_model_name,
            "embedding_device_request": self.device_request,
            "xgb_device": self.xgb_device,
            "embedding_batch_size": self.embedding_batch_size,
            "max_lookback": self.max_lookback,
            "max_evidence_sentences": self.max_evidence_sentences,
            "random_state": self.random_state,
            "feature_names": list(FEATURE_NAMES),
            "training_report": training_report,
        }
        (output_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        device: str | None = None,
        xgb_device: str | None = None,
    ) -> "HallucinationCascadeDetector":
        input_directory = Path(directory)
        metadata = json.loads(
            (input_directory / "metadata.json").read_text(encoding="utf-8")
        )
        detector = cls(
            spacy_model=metadata["spacy_model"],
            embedding_model=metadata["embedding_model"],
            device=device or metadata.get("embedding_device_request", "auto"),
            xgb_device=xgb_device or metadata.get("xgb_device", "cpu"),
            embedding_batch_size=metadata.get("embedding_batch_size", 64),
            max_lookback=metadata.get("max_lookback", 12),
            max_evidence_sentences=metadata.get("max_evidence_sentences", 48),
            random_state=metadata.get("random_state", 42),
        )
        detector.classifier = xgb.XGBClassifier()
        detector.classifier.load_model(str(input_directory / "cascade_xgb.ubj"))
        detector.classifier.set_params(device=detector.xgb_device)
        return detector


def load_halueval_samples(
    tasks: Sequence[str],
    *,
    limit_per_task: int | None = None,
    cache_dir: str | None = None,
    random_state: int = 42,
) -> list[DocumentSample]:
    """Load samples directly from the official RUCAIBox/HaluEval release."""

    output: list[DocumentSample] = []
    for task in tasks:
        LOGGER.info("Downloading/loading official HaluEval '%s' records.", task)
        dataset = load_dataset(
            "json",
            data_files={"train": HALUEVAL_URLS[task]},
            split="train",
            cache_dir=cache_dir,
        )
        if limit_per_task is not None and limit_per_task < len(dataset):
            dataset = dataset.shuffle(seed=random_state).select(range(limit_per_task))

        for row_index, row in enumerate(dataset):
            group = f"{task}:{row_index}"
            if task == "general":
                response = _clean_text(row.get("chatgpt_response"))
                if response:
                    spans = tuple(
                        span
                        for span in (
                            _clean_text(value)
                            for value in (row.get("hallucination_spans") or [])
                        )
                        if span
                    )
                    output.append(
                        DocumentSample(
                            text=response,
                            label=_as_label(
                                row.get("hallucination_label")
                                or row.get("hallucination")
                            ),
                            evidence=None,
                            group=group,
                            task=task,
                            hallucination_spans=spans,
                        )
                    )
                continue

            if task == "qa":
                evidence = _clean_text(row.get("knowledge"))
                right = _clean_text(row.get("right_answer"))
                hallucinated = _clean_text(row.get("hallucinated_answer"))
            elif task == "dialogue":
                evidence = "\n".join(
                    part
                    for part in (
                        _clean_text(row.get("knowledge")),
                        _clean_text(row.get("dialogue_history")),
                    )
                    if part
                )
                right = _clean_text(row.get("right_response"))
                hallucinated = _clean_text(row.get("hallucinated_response"))
            else:
                evidence = _clean_text(row.get("document"))
                right = _clean_text(row.get("right_summary"))
                hallucinated = _clean_text(row.get("hallucinated_summary"))

            if right:
                output.append(
                    DocumentSample(right, 0, evidence or None, group, task)
                )
            if hallucinated:
                output.append(
                    DocumentSample(hallucinated, 1, evidence or None, group, task)
                )
    return output


def _runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Embedding device: auto, cpu, cuda, or cuda:N (default: auto).",
    )
    parser.add_argument(
        "--xgb-device",
        default="cpu",
        help="XGBoost device; CPU is efficient for this small feature matrix.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--max-lookback", type=int, default=12)
    parser.add_argument("--max-evidence-sentences", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)


def _read_optional(text: str | None, file_path: Path | None) -> str | None:
    if file_path is not None:
        return file_path.read_text(encoding="utf-8")
    return text


def _write_or_print_json(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2)
    if output is None:
        print(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {output}")


def _create_detector(args: argparse.Namespace) -> HallucinationCascadeDetector:
    return HallucinationCascadeDetector(
        spacy_model=args.spacy_model,
        embedding_model=args.embedding_model,
        device=args.device,
        xgb_device=args.xgb_device,
        embedding_batch_size=args.embedding_batch_size,
        max_lookback=args.max_lookback,
        max_evidence_sentences=args.max_evidence_sentences,
        random_state=args.seed,
    )


def _command_train(args: argparse.Namespace) -> None:
    detector = _create_detector(args)
    samples = load_halueval_samples(
        args.tasks,
        limit_per_task=args.limit_per_task,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        random_state=args.seed,
    )
    LOGGER.info("Loaded %d response samples.", len(samples))
    report = detector.train(samples, validation_size=args.validation_size)
    detector.save(args.output_dir, report)
    _write_or_print_json(report, args.report_json)


def _command_analyze(args: argparse.Namespace) -> None:
    text = _read_optional(args.text, args.text_file)
    evidence = _read_optional(args.evidence, args.evidence_file)
    if not text:
        raise ValueError("Analysis text is empty.")
    detector = HallucinationCascadeDetector.load(
        args.model_dir, device=args.device, xgb_device=args.xgb_device
    )
    result = detector.analyze(
        text, evidence=evidence, alert_threshold=args.alert_threshold
    )
    _write_or_print_json(result, args.output_json)


def _command_demo(args: argparse.Namespace) -> None:
    detector = HallucinationCascadeDetector.load(
        args.model_dir, device=args.device, xgb_device=args.xgb_device
    )
    text = (
        "The 2020 United States presidential election recorded 250 million voters. "
        "This 250 million voter total represented a major increase over 2016. "
        "Because of that total, state processing systems were saturated for weeks."
    )
    result = detector.analyze(text, alert_threshold=args.alert_threshold)
    _write_or_print_json(result, args.output_json)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and run an explicit graph-based early-warning model for "
            "hallucination cascade risk."
        )
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train on official HaluEval data.")
    _runtime_arguments(train_parser)
    train_parser.add_argument(
        "--tasks",
        nargs="+",
        choices=tuple(HALUEVAL_URLS),
        default=["general", "dialogue", "summarization"],
        help="HaluEval subsets to use (default: general dialogue summarization).",
    )
    train_parser.add_argument(
        "--limit-per-task",
        type=int,
        default=1000,
        help="Maximum source records per selected task; each paired record makes two responses.",
    )
    train_parser.add_argument("--validation-size", type=float, default=0.2)
    train_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/cascade_detector")
    )
    train_parser.add_argument("--cache-dir", type=Path)
    train_parser.add_argument("--report-json", type=Path)
    train_parser.set_defaults(handler=_command_train)

    analyze_parser = commands.add_parser("analyze", help="Analyze a generated response.")
    analyze_parser.add_argument("--model-dir", type=Path, required=True)
    analyze_parser.add_argument("--device", default="auto")
    analyze_parser.add_argument("--xgb-device", default="cpu")
    text_source = analyze_parser.add_mutually_exclusive_group(required=True)
    text_source.add_argument("--text")
    text_source.add_argument("--text-file", type=Path)
    evidence_source = analyze_parser.add_mutually_exclusive_group()
    evidence_source.add_argument("--evidence")
    evidence_source.add_argument("--evidence-file", type=Path)
    analyze_parser.add_argument("--alert-threshold", type=float, default=0.65)
    analyze_parser.add_argument("--output-json", type=Path)
    analyze_parser.set_defaults(handler=_command_analyze)

    demo_parser = commands.add_parser("demo", help="Run the built-in cascade example.")
    demo_parser.add_argument("--model-dir", type=Path, required=True)
    demo_parser.add_argument("--device", default="auto")
    demo_parser.add_argument("--xgb-device", default="cpu")
    demo_parser.add_argument("--alert-threshold", type=float, default=0.65)
    demo_parser.add_argument("--output-json", type=Path)
    demo_parser.set_defaults(handler=_command_demo)
    return parser


def _is_jupyter_connection_file(value: str) -> bool:
    normalized = value.replace("\\", "/").casefold()
    return normalized.endswith(".json") and (
        "/jupyter/runtime/kernel-" in normalized
        or "/jupyter/runtime/jpserver-" in normalized
    )


def _remove_jupyter_arguments(argv: Sequence[str]) -> tuple[list[str], bool]:
    """Remove only IPython kernel transport arguments from script execution."""

    cleaned: list[str] = []
    removed = False
    index = 0
    while index < len(argv):
        argument = argv[index]
        if (
            argument in {"-f", "--file"}
            and index + 1 < len(argv)
            and _is_jupyter_connection_file(argv[index + 1])
        ):
            removed = True
            index += 2
            continue
        if argument.startswith("--file=") and _is_jupyter_connection_file(
            argument.split("=", 1)[1]
        ):
            removed = True
            index += 1
            continue
        if argument.startswith("-f") and _is_jupyter_connection_file(argument[2:]):
            removed = True
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return cleaned, removed


def main(argv: Sequence[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments, removed_kernel_arguments = _remove_jupyter_arguments(raw_arguments)
    if removed_kernel_arguments and not arguments:
        print(
            "Notebook kernel detected. Import the detector for programmatic use, "
            "or run `%run ./hallucination_cascade_detector.py train ...`."
        )
        return

    parser = build_argument_parser()
    args = parser.parse_args(arguments)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    args.handler(args)


if __name__ == "__main__":
    main()
