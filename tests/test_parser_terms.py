import unittest
import spacy

from hallucination_cascade_detector import HallucinationCascadeDetector


class TestTermExtraction(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("en")
        self.detector = object.__new__(HallucinationCascadeDetector)

    def test_extracts_non_stopword_terms(self):
        doc = self.nlp("report listed 250 voters")
        terms = self.detector._terms(doc)
        self.assertIn("report", terms)
        self.assertIn("250", terms)
        self.assertIn("voters", terms)

    def test_skips_punctuation(self):
        doc = self.nlp("report, voters.")
        terms = self.detector._terms(doc)
        self.assertNotIn(",", terms)
        self.assertNotIn(".", terms)


if __name__ == "__main__":
    unittest.main()
