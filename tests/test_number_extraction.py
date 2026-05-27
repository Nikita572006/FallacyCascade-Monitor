import unittest

from hallucination_cascade_detector import NUMBER_RE


class TestNumberExtraction(unittest.TestCase):
    def test_extracts_comma_separated_number(self):
        text = "The estimate increased to 250,000 voters."
        numbers = [match.group(0).replace(",", "") for match in NUMBER_RE.finditer(text)]
        self.assertEqual(numbers, ["250000"])

    def test_extracts_percentage(self):
        text = "Turnout rose by 20.5% in the report."
        numbers = [match.group(0).replace(",", "") for match in NUMBER_RE.finditer(text)]
        self.assertEqual(numbers, ["20.5%"])

    def test_ignores_text_without_numeric_claims(self):
        text = "The system reported an increase in turnout."
        numbers = [match.group(0) for match in NUMBER_RE.finditer(text)]
        self.assertEqual(numbers, [])


if __name__ == "__main__":
    unittest.main()
