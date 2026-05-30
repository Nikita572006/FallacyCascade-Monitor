import unittest

from hallucination_cascade_detector import FactNode, HallucinationCascadeDetector


def make_fact(index, term):
    text = f"Sentence {index} about {term}."
    return FactNode(
        index=index,
        text=text,
        start_char=0,
        end_char=len(text),
        relation="",
        subject_terms=(term,),
        object_terms=(term,),
        entity_terms=(),
        content_terms=(term,),
        numbers=(),
        token_count=len(text.split()),
    )


class TestGraphLookback(unittest.TestCase):
    def test_lookback_window_limits_long_range_edges(self):
        detector = object.__new__(HallucinationCascadeDetector)
        detector.max_lookback = 1

        facts = [
            make_fact(0, "budget"),
            make_fact(1, "unrelated"),
            make_fact(2, "budget"),
        ]

        graph = detector.build_dependency_graph(facts)

        self.assertFalse(graph.has_edge(0, 2))


if __name__ == "__main__":
    unittest.main()
