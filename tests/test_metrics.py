import unittest

from hallucination_cascade_detector import _safe_auc, _safe_average_precision


class TestMetricHelpers(unittest.TestCase):
    def test_safe_auc_returns_none_for_single_class(self):
        self.assertIsNone(_safe_auc([1, 1, 1], [0.8, 0.7, 0.9]))

    def test_safe_average_precision_returns_none_for_single_class(self):
        self.assertIsNone(_safe_average_precision([0, 0, 0], [0.1, 0.2, 0.3]))

    def test_safe_auc_computes_for_two_classes(self):
        value = _safe_auc([0, 1], [0.2, 0.8])
        self.assertEqual(value, 1.0)

    def test_safe_average_precision_computes_for_two_classes(self):
        value = _safe_average_precision([0, 1], [0.2, 0.8])
        self.assertEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
