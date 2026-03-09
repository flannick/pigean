from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class StitchedArtifactsTest(unittest.TestCase):
    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    def _build(self, out_dir: Path) -> subprocess.CompletedProcess[str]:
        repo_root = self._repo_root()
        return subprocess.run(
            [sys.executable, "scripts/build_stitched_artifacts.py", "--out-dir", str(out_dir)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    def _sha256(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_stitched_artifacts_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            out_dir_1 = Path(tmp1)
            out_dir_2 = Path(tmp2)
            proc1 = self._build(out_dir_1)
            self.assertEqual(proc1.returncode, 0, msg=(proc1.stderr or "") + (proc1.stdout or ""))
            proc2 = self._build(out_dir_2)
            self.assertEqual(proc2.returncode, 0, msg=(proc2.stderr or "") + (proc2.stdout or ""))

            for artifact_name in ("pigean_stitched.py", "eaggl_stitched.py"):
                artifact_1 = out_dir_1 / artifact_name
                artifact_2 = out_dir_2 / artifact_name
                self.assertTrue(artifact_1.exists(), msg=str(artifact_1))
                self.assertTrue(artifact_2.exists(), msg=str(artifact_2))
                self.assertEqual(self._sha256(artifact_1), self._sha256(artifact_2))

            pigean_text = (out_dir_1 / "pigean_stitched.py").read_text(encoding="utf-8")
            eaggl_text = (out_dir_1 / "eaggl_stitched.py").read_text(encoding="utf-8")
            self.assertIn("# ===== BEGIN src/pigean/__main__.py =====", pigean_text)
            self.assertIn("# ===== BEGIN src/pigean/dispatch.py =====", pigean_text)
            self.assertIn("# ===== BEGIN src/pigean/pipeline.py =====", pigean_text)
            self.assertIn("# ===== BEGIN src/pegs_shared/types.py =====", pigean_text)
            self.assertNotIn("# ===== BEGIN src/pigean_dispatch.py =====", pigean_text)
            self.assertNotIn("# ===== BEGIN src/pigean_pipeline.py =====", pigean_text)
            self.assertNotIn("# ===== BEGIN src/pigean_huge.py =====", pigean_text)
            self.assertNotIn("# ===== BEGIN src/pigean_outputs.py =====", pigean_text)
            self.assertNotIn("# ===== BEGIN src/pigean_phewas.py =====", pigean_text)
            self.assertIn("# ===== BEGIN src/eaggl/__main__.py =====", eaggl_text)
            self.assertIn("# ===== BEGIN src/pegs_shared/types.py =====", eaggl_text)

    def test_stitched_artifacts_run_help_without_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            build_proc = self._build(out_dir)
            self.assertEqual(build_proc.returncode, 0, msg=(build_proc.stderr or "") + (build_proc.stdout or ""))

            pigean_proc = subprocess.run(
                [sys.executable, str(out_dir / "pigean_stitched.py"), "gibbs", "--help"],
                cwd=self._repo_root(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(pigean_proc.returncode, 0, msg=(pigean_proc.stderr or "") + (pigean_proc.stdout or ""))
            self.assertIn("Usage: python -m pigean", pigean_proc.stdout)

            eaggl_proc = subprocess.run(
                [sys.executable, str(out_dir / "eaggl_stitched.py"), "factor", "--help"],
                cwd=self._repo_root(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(eaggl_proc.returncode, 0, msg=(eaggl_proc.stderr or "") + (eaggl_proc.stdout or ""))
            self.assertIn("Usage: python -m eaggl", eaggl_proc.stdout)
