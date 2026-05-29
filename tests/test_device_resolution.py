import unittest
from unittest.mock import patch

from hallucination_cascade_detector import HallucinationCascadeDetector


class TestDeviceResolution(unittest.TestCase):
    @patch("hallucination_cascade_detector.torch.cuda.is_available", return_value=False)
    def test_auto_uses_cpu_when_cuda_is_unavailable(self, _mock_cuda):
        self.assertEqual(HallucinationCascadeDetector._resolve_device("auto"), "cpu")

    @patch("hallucination_cascade_detector.torch.cuda.is_available", return_value=True)
    def test_auto_uses_cuda_when_available(self, _mock_cuda):
        self.assertEqual(HallucinationCascadeDetector._resolve_device("auto"), "cuda")

    @patch("hallucination_cascade_detector.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_raises_when_unavailable(self, _mock_cuda):
        with self.assertRaises(RuntimeError):
            HallucinationCascadeDetector._resolve_device("cuda")

    def test_explicit_cpu_is_preserved(self):
        self.assertEqual(HallucinationCascadeDetector._resolve_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
