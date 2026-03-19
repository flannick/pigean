from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pegs_shared import output_tables as pegs_output_tables  # noqa: E402


class GeneStatsOutputTest(unittest.TestCase):
    def _runtime(self):
        return SimpleNamespace(
            genes=["GENE_A", "GENE_B", "GENE_C"],
            genes_missing=[],
            priors=np.array([0.1, 0.2, 0.3]),
            priors_r_hat=None,
            priors_mcse=None,
            priors_adj=None,
            combined_prior_Ys=np.array([2.5, 0.4, -1.8]),
            combined_prior_Ys_r_hat=None,
            combined_prior_Ys_mcse=None,
            combined_prior_Ys_adj=None,
            combined_prior_Y_ses=None,
            combined_Ds=None,
            gene_to_huge_score=None,
            gene_to_gwas_huge_score=None,
            gene_to_gwas_huge_score_uncorrected=None,
            gene_to_exomes_huge_score=None,
            gene_to_positive_controls=None,
            gene_to_case_count_logbf=None,
            Y=np.array([1.0, 0.2, 0.1]),
            Y_r_hat=None,
            Y_mcse=None,
            Y_for_regression=None,
            Y_uncorrected=None,
            priors_orig=None,
            priors_adj_orig=None,
            batches=None,
            X_orig=np.array([[1.0], [1.0], [1.0]]),
            gene_to_chrom=None,
            gene_to_pos=None,
            gene_covariate_zs=None,
            gene_covariate_names=None,
            gene_covariate_intercept_index=None,
            gene_ignored_N=None,
            gene_N=np.array([1, 1, 1]),
            gene_ignored_N_missing=None,
            gene_N_missing=np.array([]),
            priors_missing=None,
            priors_missing_orig=None,
            priors_adj_missing=None,
            priors_adj_missing_orig=None,
            combined_Ds_missing=None,
            X_orig_missing_genes=np.zeros((0, 1)),
            get_gene_N=lambda get_missing=False: np.array([]) if get_missing else np.array([1, 1, 1]),
        )

    def test_write_gene_statistics_can_filter_on_combined(self) -> None:
        runtime = self._runtime()
        with TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "gene_stats.out"
            pegs_output_tables.write_gene_statistics(
                runtime,
                str(out_path),
                max_no_write_gene_combined=1.0,
            )
            lines = out_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(lines[0].split("\t")[0], "Gene")
        body = lines[1:]
        self.assertEqual(len(body), 2)
        self.assertTrue(any(line.startswith("GENE_A\t") for line in body))
        self.assertTrue(any(line.startswith("GENE_C\t") for line in body))
        self.assertFalse(any(line.startswith("GENE_B\t") for line in body))
