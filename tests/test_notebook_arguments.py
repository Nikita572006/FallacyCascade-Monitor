import unittest

from hallucination_cascade_detector import _remove_jupyter_arguments


class TestJupyterArgumentFiltering(unittest.TestCase):
    def test_removes_injected_kernel_connection_file(self):
        kernel_file = r"C:\Users\student\AppData\Roaming\jupyter\runtime\kernel-123.json"

        arguments, removed = _remove_jupyter_arguments(["-f", kernel_file])

        self.assertEqual(arguments, [])
        self.assertTrue(removed)

    def test_preserves_explicit_training_command(self):
        kernel_file = r"C:\Users\student\AppData\Roaming\jupyter\runtime\kernel-123.json"

        arguments, removed = _remove_jupyter_arguments(
            ["train", "--limit-per-task", "100", "-f", kernel_file]
        )

        self.assertEqual(arguments, ["train", "--limit-per-task", "100"])
        self.assertTrue(removed)

    def test_leaves_normal_cli_arguments_unchanged(self):
        original = ["demo", "--model-dir", "./artifacts/cascade_detector"]

        arguments, removed = _remove_jupyter_arguments(original)

        self.assertEqual(arguments, original)
        self.assertFalse(removed)


if __name__ == "__main__":
    unittest.main()
