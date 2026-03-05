from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pegs_utils  # noqa: E402
import pegs_utils_bundle  # noqa: E402
import pegs_utils_phewas  # noqa: E402


class PegsUtilsDomainsTest(unittest.TestCase):
    def test_bundle_constants_match_compat_shim(self) -> None:
        self.assertEqual(pegs_utils_bundle.EAGGL_BUNDLE_SCHEMA, pegs_utils.EAGGL_BUNDLE_SCHEMA)
        self.assertEqual(
            pegs_utils_bundle.EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
            pegs_utils.EAGGL_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        )

    def test_phewas_resolver_decision_matches_compat_shim(self) -> None:
        kwargs = dict(
            requested_input="/tmp/gene_phewas.tsv",
            reusable_inputs=["/tmp/gene_phewas.tsv"],
            read_gene_phewas=True,
            num_gene_phewas_filtered=0,
        )
        new_decision = pegs_utils_phewas.resolve_gene_phewas_input_decision_for_stage(**kwargs)
        shim_decision = pegs_utils.resolve_gene_phewas_input_decision_for_stage(**kwargs)
        self.assertEqual(new_decision.mode, shim_decision.mode)
        self.assertEqual(new_decision.reason, shim_decision.reason)
        self.assertEqual(new_decision.resolved_input, shim_decision.resolved_input)

    def test_bundle_tar_mode_matches_compat_shim(self) -> None:
        self.assertEqual(
            pegs_utils_bundle.get_tar_write_mode_for_bundle_path("a.tar.gz"),
            pegs_utils.get_tar_write_mode_for_bundle_path("a.tar.gz"),
        )
        self.assertEqual(
            pegs_utils_bundle.get_tar_write_mode_for_bundle_path("a.tar"),
            pegs_utils.get_tar_write_mode_for_bundle_path("a.tar"),
        )


if __name__ == "__main__":
    unittest.main()
