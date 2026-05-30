import unittest
import spacy

from hallucination_cascade_detector import HallucinationCascadeDetector


class TestTokenNormalization(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.blank("en")

    def test_removes_stopword(self):
        token = self.nlp("the")[0]
        self.assertIsNone(HallucinationCascadeDetector._normalize_token(token))

    def test_keeps_numeric_token(self):
        token = self.nlp("250")[0]
        self.assertEqual(HallucinationCascadeDetector._normalize_token(token), "250")

    def test_removes_single_character_non_number(self):
        token = self.nlp("x")[0]
        self.assertIsNone(HallucinationCascadeDetector._normalize_token(token))


if __name__ == "__main__":
    unittest.main()
