import unittest

from hallucination_cascade_detector import _as_label, _clean_text


class TestUtilityFunctions(unittest.TestCase):
    def test_clean_text_strips_whitespace(self):
        self.assertEqual(_clean_text("  hello world  "), "hello world")

    def test_clean_text_converts_none_to_empty_string(self):
        self.assertEqual(_clean_text(None), "")

    def test_as_label_marks_yes_as_positive(self):
        self.assertEqual(_as_label("yes"), 1)

    def test_as_label_marks_false_as_negative(self):
        self.assertEqual(_as_label("false"), 0)


if __name__ == "__main__":
    unittest.main()
