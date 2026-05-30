import unittest

from hallucination_cascade_detector import NUMBER_RE


class TestNumberExtraction(unittest.TestCase):
    def test_extracts_plain_number(self):
        values = [m.group(0) for m in NUMBER_RE.finditer("There were 250 voters.")]
        self.assertEqual(values, ["250"])

    def test_extracts_comma_number(self):
        values = [m.group(0).replace(",", "") for m in NUMBER_RE.finditer("There were 250,000 voters.")]
        self.assertEqual(values, ["250000"])

    def test_extracts_percentage(self):
        values = [m.group(0) for m in NUMBER_RE.finditer("Turnout rose by 20.5%.")]
        self.assertEqual(values, ["20.5%"])


if __name__ == "__main__":
    unittest.main()
