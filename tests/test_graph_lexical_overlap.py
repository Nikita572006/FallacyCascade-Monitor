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


class TestGraphLexicalOverlap(unittest.TestCase):
    def test_creates_edge_when_prior_object_reappears_as_subject_hook(self):
        detector = object.__new__(HallucinationCascadeDetector)
        detector.max_lookback = 12

        facts = [
            make_fact(0, "The model reported a turnout estimate.", objects=("turnout",)),
            make_fact(1, "The turnout estimate shaped the conclusion.", subjects=("turnout",)),
        ]

        graph = detector.build_dependency_graph(facts)

        self.assertTrue(graph.has_edge(0, 1))
        self.assertIn("turnout", graph.edges[0, 1]["overlap"])


if __name__ == "__main__":
    unittest.main()
