import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_eval_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_eval_data", SCRIPT)
data_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data_tools)


class DataCopyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.glb"
        self.source.write_bytes(b"scene fixture")
        self.expected = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.target = self.root / "output" / "scene.glb"

    def test_copy_preserves_source_and_is_idempotent(self):
        data_tools.copy_verified(self.source, self.target, self.expected)
        data_tools.copy_verified(self.source, self.target, self.expected)
        self.assertEqual(self.source.read_bytes(), self.target.read_bytes())
        self.assertFalse(self.source.samefile(self.target))
        self.assertFalse(self.target.is_symlink())

    def test_conflicting_destination_is_not_overwritten(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"existing scene")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            data_tools.copy_verified(self.source, self.target, self.expected)
        self.assertEqual(self.target.read_bytes(), b"existing scene")
        self.assertEqual(self.source.read_bytes(), b"scene fixture")

    def test_bad_source_leaves_no_partial_destination(self):
        self.source.write_bytes(b"corrupt scene")
        with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
            data_tools.copy_verified(self.source, self.target, self.expected)
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.iterdir()), [])

    def test_paths_cannot_escape_data_root(self):
        for relative in ("../escape", "/absolute", "C:/absolute", "a/../../escape"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                data_tools.local_path(self.root, relative)
        outside = self.root.parent
        (self.root / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escapes"):
            data_tools.local_path(self.root, "linked/escape")


if __name__ == "__main__":
    unittest.main()
