from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIGEAN_LEGACY = REPO_ROOT / "src" / "pigean_legacy_main.py"
EAGGL_LEGACY = REPO_ROOT / "src" / "eaggl" / "legacy_main.py"

MAX_PIGEAN_LEGACY_LINES = 753
MAX_EAGGL_LEGACY_LINES = 5693

ALLOWED_PIGEAN_LEGACY_IMPORTERS = {
    "tests/test_gibbs_hyper_mutation_unittest.py",
    "tests/test_phewas_stage_reuse_unittest.py",
}
ALLOWED_PIGEAN_LEGACY_DYNAMIC_IMPORTERS = {
    "src/pigean/main_support.py",
    "src/pigean/state.py",
}

ALLOWED_EAGGL_LEGACY_IMPORTERS = {
    "tests/eaggl/test_factor_stage_unittest.py",
    "tests/eaggl/test_phewas_stage_reuse_unittest.py",
}


class LegacyRetirementGuardrailsTest(unittest.TestCase):
    def test_legacy_file_line_counts_do_not_grow(self) -> None:
        pigean_lines = len(PIGEAN_LEGACY.read_text(encoding="utf-8").splitlines())
        eaggl_lines = len(EAGGL_LEGACY.read_text(encoding="utf-8").splitlines())
        self.assertLessEqual(
            pigean_lines,
            MAX_PIGEAN_LEGACY_LINES,
            msg="src/pigean_legacy_main.py grew from baseline %d to %d lines" % (
                MAX_PIGEAN_LEGACY_LINES,
                pigean_lines,
            ),
        )
        self.assertLessEqual(
            eaggl_lines,
            MAX_EAGGL_LEGACY_LINES,
            msg="src/eaggl/legacy_main.py grew from baseline %d to %d lines" % (
                MAX_EAGGL_LEGACY_LINES,
                eaggl_lines,
            ),
        )

    def test_only_allowed_python_files_import_legacy_runtime_modules(self) -> None:
        self.assertEqual(
            self._collect_import_sites("pigean_legacy_main"),
            sorted(ALLOWED_PIGEAN_LEGACY_IMPORTERS),
        )
        self.assertEqual(
            self._collect_import_sites("eaggl.legacy_main"),
            sorted(ALLOWED_EAGGL_LEGACY_IMPORTERS),
        )

    def test_only_allowed_source_files_use_dynamic_legacy_imports(self) -> None:
        self.assertEqual(
            self._collect_dynamic_import_sites("pigean_legacy_main"),
            sorted(ALLOWED_PIGEAN_LEGACY_DYNAMIC_IMPORTERS),
        )
        self.assertEqual(self._collect_dynamic_import_sites("eaggl.legacy_main"), [])

    def test_normal_entrypoints_do_not_import_pigean_legacy_main_directly(self) -> None:
        package_main = (REPO_ROOT / "src" / "pigean" / "__main__.py").read_text(encoding="utf-8")
        package_init = (REPO_ROOT / "src" / "pigean" / "__init__.py").read_text(encoding="utf-8")
        self.assertNotIn("pigean_legacy_main", package_main)
        self.assertNotIn("pigean_legacy_main", package_init)
        self.assertNotIn("from .legacy_main import main", package_main)

    def test_module_object_dispatch_counts_do_not_grow(self) -> None:
        report = self._run_legacy_symbol_report()
        counts = report["module_object_dispatch_counts"]
        self.assertEqual(counts["dispatch.run_main_pipeline(_legacy_main"], 0)
        self.assertEqual(counts["dispatch.run_main_pipeline(sys.modules[__name__]"], 0)
        self.assertLessEqual(counts["_eaggl_dispatch.run_main_pipeline(_build_main_domain()"], 1)

    def test_legacy_symbol_report_script_runs(self) -> None:
        report = self._run_legacy_symbol_report()
        self.assertIn("line_counts", report)
        self.assertIn("legacy_import_sites", report)
        self.assertIn("legacy_dynamic_import_sites", report)
        self.assertIn("legacy_import_counts", report)
        self.assertIn("module_object_dispatch_counts", report)
        self.assertIn("pigean_legacy_top_level_defs", report)
        self.assertIn("eaggl_legacy_top_level_defs", report)
        self.assertIsInstance(report["pigean_legacy_top_level_defs"], list)
        self.assertIsInstance(report["eaggl_legacy_top_level_defs"], list)
        self.assertEqual(
            report["legacy_import_sites"]["pigean_legacy_main"],
            sorted(ALLOWED_PIGEAN_LEGACY_IMPORTERS),
        )
        self.assertEqual(
            report["legacy_import_sites"]["eaggl.legacy_main"],
            sorted(ALLOWED_EAGGL_LEGACY_IMPORTERS),
        )
        self.assertEqual(
            report["legacy_dynamic_import_sites"]["pigean_legacy_main"],
            sorted(ALLOWED_PIGEAN_LEGACY_DYNAMIC_IMPORTERS),
        )
        self.assertEqual(report["legacy_dynamic_import_sites"]["eaggl.legacy_main"], [])

    def _run_legacy_symbol_report(self) -> dict[str, object]:
        proc = subprocess.run(
            [sys.executable, "scripts/pigean/list_legacy_symbols.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        return json.loads(proc.stdout)

    def _collect_import_sites(self, module_name: str) -> list[str]:
        hits: list[str] = []
        for path in self._iter_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == module_name:
                            found = True
                            break
                elif isinstance(node, ast.ImportFrom) and node.module == module_name:
                    found = True
                if found:
                    break
            if found:
                hits.append(str(path.relative_to(REPO_ROOT)))
        return hits

    def _collect_dynamic_import_sites(self, module_name: str) -> list[str]:
        needle = 'import_module("%s")' % module_name
        hits: list[str] = []
        for path in sorted((REPO_ROOT / "src").rglob("*.py")):
            if needle in path.read_text(encoding="utf-8"):
                hits.append(str(path.relative_to(REPO_ROOT)))
        return hits

    def _iter_python_files(self) -> list[Path]:
        return sorted((REPO_ROOT / "src").rglob("*.py")) + sorted((REPO_ROOT / "tests").rglob("*.py"))


if __name__ == "__main__":
    unittest.main()
