import unittest
import networkx as nx

from hallucination_cascade_detector import HallucinationCascadeDetector


class TestDependencyDepths(unittest.TestCase):
    def test_dependency_depths_follow_topological_order(self):
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (0, 3)])

        depths = HallucinationCascadeDetector._dependency_depths(graph)

        self.assertEqual(depths[0], 0)
        self.assertEqual(depths[1], 1)
        self.assertEqual(depths[2], 2)
        self.assertEqual(depths[3], 1)


if __name__ == "__main__":
    unittest.main()
