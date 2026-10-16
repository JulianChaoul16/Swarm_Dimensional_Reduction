"""CSV input contract checks; run with python -m unittest -v."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dataset import load_csv


class DatasetTests(unittest.TestCase):
    def load(self, contents, target="label", delimiter=","):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text(contents, encoding="utf-8")
            return load_csv(path, target, delimiter)

    def test_target_in_middle_and_feature_order(self):
        X, y, names = self.load("\ufefffirst,label,last\n1,a,2\n3,b,4\n5,a,6\n7,b,8\n")
        self.assertEqual(names, ["first", "last"])
        self.assertEqual(X.tolist(), [[1, 2], [3, 4], [5, 6], [7, 8]])
        self.assertEqual(y.tolist(), ["a", "b", "a", "b"])

    def test_custom_separator_and_blank_lines(self):
        X, y, names = self.load("value;label\n1;a\n\n2;b\n3;a\n4;b\n", delimiter=";")
        self.assertEqual(X.shape, (4, 1))
        self.assertEqual(names, ["value"])

    def test_invalid_inputs_have_actionable_errors(self):
        cases = [
            ("", "empty"),
            ("x,other\n1,a\n", "Target column"),
            ("x,x,label\n1,2,a\n", "unique"),
            ("label\na\n", "at least one feature"),
            ("x,label\n", "no data rows"),
            ("x,label\n1,a,extra\n", "expected 2 values"),
            ("x,label\ntext,a\n", "numeric"),
            ("x,label\nnan,a\n", "finite"),
            ("x,label\n1,\n", "target value is missing"),
            ("x,label\n1,a\n2,a\n", "at least two target classes"),
            ("x,label\n1,a\n2,b\n3,b\n", "at least two rows"),
        ]
        for contents, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.load(contents)


if __name__ == "__main__":
    unittest.main()
