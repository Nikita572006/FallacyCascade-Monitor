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


class TestGraphNodes(unittest.TestCase):
    def test_adds_sentence_nodes_with_fact_metadata(self):
        detector = object.__new__(HallucinationCascadeDetector)
        detector.max_lookback = 12

        facts = [
            make_fact(0, "The report listed 250 voters.", objects=("voters",), numbers=("250",)),
            make_fact(1, "The voter total was repeated.", subjects=("voter",), terms=("voter",)),
        ]

        graph = detector.build_dependency_graph(facts)

        self.assertEqual(graph.number_of_nodes(), 2)
        self.assertEqual(graph.nodes[0]["text"], "The report listed 250 voters.")
        self.assertEqual(graph.nodes[0]["fact"], facts[0])


if __name__ == "__main__":
    unittest.main()
