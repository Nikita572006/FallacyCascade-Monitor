import unittest

from hallucination_cascade_detector import OBJECT_DEPS, SUBJECT_DEPS, NOMINAL_MODIFIER_DEPS


class TestDependencyConstants(unittest.TestCase):
    def test_subject_dependencies_include_active_and_passive_subjects(self):
        self.assertIn("nsubj", SUBJECT_DEPS)
        self.assertIn("nsubjpass", SUBJECT_DEPS)

    def test_object_dependencies_include_direct_and_prepositional_objects(self):
        self.assertIn("dobj", OBJECT_DEPS)
        self.assertIn("pobj", OBJECT_DEPS)

    def test_nominal_modifiers_include_compound_and_numeric_modifiers(self):
        self.assertIn("compound", NOMINAL_MODIFIER_DEPS)
        self.assertIn("nummod", NOMINAL_MODIFIER_DEPS)


if __name__ == "__main__":
    unittest.main()
