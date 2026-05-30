import unittest

from hallucination_cascade_detector import FactNode, HallucinationCascadeDetector


def make_fact(index, text, subjects=(), objects=(), entities=(), terms=(), numbers=()):
    return FactNode(
        index=index,
        text=text,
        start_char=0,
        end_char=len(text),
        relation="",
        subject_terms=subjects,
        object_terms=objects,
        entity_terms=entities,
        content_terms=terms,
        numbers=numbers,
        token_count=len(text.split()),
    )


class TestGraphEdgeMetadata(unittest.TestCase):
    def test_edge_stores_kind_overlap_and_weight(self):
        detector = object.__new__(HallucinationCascadeDetector)
        detector.max_lookback = 12

        facts = [
            make_fact(0, "The claim introduced 250 voters.", objects=("voters",), numbers=("250",)),
            make_fact(1, "The 250 voter figure was reused.", subjects=("voter",), terms=("250", "voter"), numbers=("250",)),
        ]

        graph = detector.build_dependency_graph(facts)

        edge = graph.edges[0, 1]
        self.assertEqual(edge["kind"], "lexical_dependency")
        self.assertIn("250", edge["overlap"])
        self.assertGreaterEqual(edge["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
