import unittest

from hallucination_cascade_detector import DocumentSample, FactNode, FeaturizedBatch
import numpy as np


class TestDataModels(unittest.TestCase):
    def test_fact_node_stores_sentence_claim_fields(self):
        node = FactNode(
            index=0,
            text="The report listed 250 voters.",
            start_char=0,
            end_char=30,
            relation="list",
            subject_terms=("report",),
            object_terms=("voter",),
            entity_terms=(),
            content_terms=("report", "voter", "250"),
            numbers=("250",),
            token_count=6,
        )

        self.assertEqual(node.index, 0)
        self.assertEqual(node.numbers, ("250",))
        self.assertIn("report", node.subject_terms)

    def test_document_sample_accepts_optional_spans(self):
        sample = DocumentSample(
            text="Claim with an unsupported span.",
            label=1,
            evidence=None,
            group="general:1",
            task="general",
            hallucination_spans=("unsupported span",),
        )

        self.assertEqual(sample.label, 1)
        self.assertEqual(sample.hallucination_spans, ("unsupported span",))

    def test_featurized_batch_shapes_are_preserved(self):
        batch = FeaturizedBatch(
            features=np.zeros((2, 3), dtype=np.float32),
            labels=np.array([0, 1]),
            weights=np.array([0.5, 0.5]),
            spans=[(0, 1), (1, 2)],
        )

        self.assertEqual(batch.features.shape, (2, 3))
        self.assertEqual(batch.labels.tolist(), [0, 1])
        self.assertEqual(batch.spans, [(0, 1), (1, 2)])


if __name__ == "__main__":
    unittest.main()
