from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class EntryPathLegacyImportsTest(unittest.TestCase):
    def _clear_modules(self, *module_names: str) -> None:
        for module_name in module_names:
            sys.modules.pop(module_name, None)

    def test_pigean_main_help_does_not_import_retired_legacy_module(self) -> None:
        self._clear_modules("pigean", "pigean_legacy_main")
        pigean = importlib.import_module("pigean")
        with self.assertRaises(SystemExit) as ctx:
            pigean.main(["gibbs", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertNotIn("pigean_legacy_main", sys.modules)

    def test_eaggl_main_help_does_not_import_retired_legacy_module(self) -> None:
        self._clear_modules("eaggl", "eaggl.legacy_main")
        eaggl = importlib.import_module("eaggl")
        with self.assertRaises(SystemExit) as ctx:
            eaggl.main(["factor", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertNotIn("eaggl.legacy_main", sys.modules)

    def test_package_roots_do_not_expose_removed_compat_attrs(self) -> None:
        self._clear_modules("pigean", "eaggl")
        pigean = importlib.import_module("pigean")
        eaggl = importlib.import_module("eaggl")
        with self.assertRaises(AttributeError):
            getattr(pigean, "_build_prefilter_keep_mask")
        with self.assertRaises(AttributeError):
            getattr(eaggl, "GeneSetData")


if __name__ == "__main__":
    unittest.main()
