from __future__ import annotations

import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import pegs_utils  # noqa: E402


class PegsUtilsBundleTest(unittest.TestCase):
    def test_get_tar_write_mode_for_bundle_path(self) -> None:
        self.assertEqual(
            pegs_utils.get_tar_write_mode_for_bundle_path("handoff.tar.gz"),
            "w:gz",
        )
        self.assertEqual(
            pegs_utils.get_tar_write_mode_for_bundle_path("handoff.tgz"),
            "w:gz",
        )
        self.assertEqual(
            pegs_utils.get_tar_write_mode_for_bundle_path("handoff.tar"),
            "w",
        )
        with self.assertRaisesRegex(ValueError, "must end with .tar, .tar.gz, or .tgz"):
            pegs_utils.get_tar_write_mode_for_bundle_path("handoff.zip")

    def test_load_bundle_manifest_and_defaults_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "X.tsv.gz").write_text("SET_A\tGENE1\n", encoding="utf-8")
            manifest = {
                "schema": pegs_utils.EAGGL_BUNDLE_SCHEMA,
                "default_inputs": {"X_in": "X.tsv.gz"},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            bundle_path = root / "handoff.tar.gz"
            with tarfile.open(bundle_path, "w:gz") as tar_fh:
                tar_fh.add(root / "manifest.json", arcname="manifest.json")
                tar_fh.add(root / "X.tsv.gz", arcname="X.tsv.gz")

            extract_dir, loaded_manifest = pegs_utils.load_bundle_manifest(
                str(bundle_path),
                pegs_utils.EAGGL_BUNDLE_SCHEMA,
                bundle_flag_name="--eaggl-bundle-in",
            )
            try:
                self.assertEqual(loaded_manifest["schema"], pegs_utils.EAGGL_BUNDLE_SCHEMA)
                resolved = pegs_utils.resolve_bundle_default_inputs(
                    loaded_manifest["default_inputs"],
                    extract_dir,
                    pegs_utils.EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
                    bundle_flag_name="--eaggl-bundle-in",
                )
                self.assertIn("X_in", resolved)
                self.assertTrue(Path(resolved["X_in"]).exists())
            finally:
                shutil.rmtree(extract_dir, ignore_errors=True)

    def test_resolve_bundle_default_inputs_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, "Refusing to resolve path outside"):
                pegs_utils.resolve_bundle_default_inputs(
                    {"X_in": "../outside.tsv"},
                    str(root),
                    pegs_utils.EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
                    bundle_flag_name="--eaggl-bundle-in",
                )

    def test_collect_file_metadata_includes_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "value.txt"
            path.write_text("abc\n", encoding="utf-8")
            meta = pegs_utils.collect_file_metadata(str(path))
            self.assertEqual(meta["size_bytes"], path.stat().st_size)
            self.assertRegex(meta["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
