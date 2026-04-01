from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import importlib
from pathlib import Path


class PigeanCliTest(unittest.TestCase):
    def _base_env(self, repo_root: Path) -> dict[str, str]:
        env = dict(os.environ)
        src_root = str(repo_root / "src")
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else src_root + os.pathsep + env["PYTHONPATH"]
        return env

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        repo_root = Path(__file__).resolve().parents[1]
        cmd = [sys.executable, "-m", "pigean", *args]
        return subprocess.run(
            cmd,
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_removed_gene_bfs_flag_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--gene-bfs-in", "dummy.txt")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --gene-bfs-in has been removed; use --gene-stats-in instead", err)

    def test_removed_gene_zs_flag_has_removed_message(self) -> None:
        proc = self._run("gibbs", "--gene-zs-in", "dummy.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --gene-zs-in has been removed and is no longer supported", err)

    def test_removed_gene_percentiles_flag_has_removed_message(self) -> None:
        proc = self._run("gibbs", "--gene-percentiles-in", "dummy.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --gene-percentiles-in has been removed and is no longer supported", err)

    def test_removed_sigma_flag_has_removed_message(self) -> None:
        proc = self._run("gibbs", "--chisq-threshold", "5")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --chisq-threshold has been removed and is no longer supported", err)

    def test_removed_run_gls_flag_has_removed_message(self) -> None:
        proc = self._run("gibbs", "--run-gls")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --run-gls has been removed and is no longer supported", err)
        self.assertNotIn("Traceback", err)

    def test_invalid_option_returns_usage_error_without_traceback(self) -> None:
        proc = self._run("gibbs", "--definitely-invalid-option")
        self.assertEqual(proc.returncode, 2)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("no such option", err)
        self.assertNotIn("Traceback", err)

    def test_missing_config_returns_config_error_without_traceback(self) -> None:
        proc = self._run("gibbs", "--config", "definitely_missing_config.json")
        self.assertEqual(proc.returncode, 2)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Could not read config file", err)
        self.assertNotIn("Traceback", err)

    def test_positive_controls_list_rejects_file_paths(self) -> None:
        proc = self._run("gibbs", "--positive-controls-list", "tests/data/t2d_smoke/mody.gene.list")
        self.assertEqual(proc.returncode, 2)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("expects a comma-separated list of gene symbols", err)
        self.assertIn("--gene-list-in", err)
        self.assertIn("--positive-controls-in", err)
        self.assertNotIn("Traceback", err)

    def test_gene_list_alias_round_trips_in_effective_config(self) -> None:
        proc = self._run(
            "gibbs",
            "--gene-list",
            "INS,GCK",
            "--gene-list-in",
            "genes.tsv",
            "--gene-list-id-col",
            "Gene",
            "--gene-list-prob-col",
            "Weight",
            "--gene-list-default-prob",
            "0.8",
            "--gene-list-all-in",
            "background.tsv",
            "--gene-list-all-id-col",
            "Symbol",
            "--gene-list-no-header",
            "--gene-list-all-no-header",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        options = payload["options"]
        self.assertEqual(options["positive_controls_list"], ["INS", "GCK"])
        self.assertEqual(options["positive_controls_in"], "genes.tsv")
        self.assertEqual(options["positive_controls_id_col"], "Gene")
        self.assertEqual(options["positive_controls_prob_col"], "Weight")
        self.assertEqual(options["positive_controls_default_prob"], 0.8)
        self.assertIsNone(options["positive_controls_all_in"])
        self.assertEqual(options["gene_universe_in"], "background.tsv")
        self.assertEqual(options["gene_universe_id_col"], "Symbol")
        self.assertFalse(options["positive_controls_has_header"])
        self.assertFalse(options["gene_universe_has_header"])

    def test_positive_controls_alias_still_round_trips_in_effective_config(self) -> None:
        proc = self._run("gibbs", "--positive-controls-list", "INS", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["options"]["positive_controls_list"], ["INS"])

    def test_gene_universe_options_round_trip_in_effective_config(self) -> None:
        proc = self._run(
            "gibbs",
            "--gene-universe-in",
            "universe.tsv",
            "--gene-universe-id-col",
            "Symbol",
            "--gene-universe-no-header",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        options = payload["options"]
        self.assertEqual(options["gene_universe_in"], "universe.tsv")
        self.assertEqual(options["gene_universe_id_col"], "Symbol")
        self.assertFalse(options["gene_universe_has_header"])

    def test_gene_list_all_alias_populates_gene_universe_options(self) -> None:
        proc = self._run(
            "gibbs",
            "--gene-list-all-in",
            "background.tsv",
            "--gene-list-all-id-col",
            "Gene",
            "--gene-list-all-no-header",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        options = payload["options"]
        self.assertEqual(options["gene_universe_in"], "background.tsv")
        self.assertEqual(options["gene_universe_id_col"], "Gene")
        self.assertFalse(options["gene_universe_has_header"])

    def test_gene_stats_input_requires_explicit_gene_universe(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gene_stats = tmp_path / "gene_stats.tsv"
            gene_stats.write_text(
                "Gene\tDirect\tCombined\nKCNJ11\t2.0\t2.5\nGCK\t1.0\t1.5\n",
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "-m",
                "pigean",
                "betas",
                "--X-in",
                "tests/data/t2d_smoke/gene_set_list_input_subset_toy.txt",
                "--gene-map-in",
                "tests/data/model_small/portal_gencode.gene.map",
                "--gene-stats-in",
                str(gene_stats),
                "--gene-stats-id-col",
                "Gene",
                "--gene-stats-log-bf-col",
                "Direct",
                "--gene-stats-combined-col",
                "Combined",
                "--gene-set-stats-out",
                str(tmp_path / "gene_set_stats.out"),
            ]
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                env=self._base_env(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("requires an explicit gene universe", err)
            self.assertIn("--gene-universe-in", err)
            self.assertIn("--gene-universe-from-y", err)
            self.assertIn("--gene-universe-from-x", err)

    def test_gene_stats_input_supports_gene_universe_from_x(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gene_stats = tmp_path / "gene_stats.tsv"
            gene_stats.write_text(
                "Gene\tDirect\tCombined\nKCNJ11\t2.0\t2.5\nGCK\t1.0\t1.5\n",
                encoding="utf-8",
            )
            cmd = [
                sys.executable,
                "-m",
                "pigean",
                "beta_tildes",
                "--deterministic",
                "--linear",
                "--X-in",
                "tests/data/t2d_smoke/gene_set_list_input_subset_toy.txt",
                "--gene-map-in",
                "tests/data/model_small/portal_gencode.gene.map",
                "--gene-loc-file",
                "tests/data/model_small/NCBI37.3.plink.gene.loc",
                "--gene-stats-in",
                str(gene_stats),
                "--gene-stats-id-col",
                "Gene",
                "--gene-stats-log-bf-col",
                "Direct",
                "--gene-stats-combined-col",
                "Combined",
                "--gene-universe-from-x",
                "--gene-set-stats-out",
                str(tmp_path / "gene_set_stats.out"),
                "--gene-stats-out",
                str(tmp_path / "gene_stats.out"),
                "--params-out",
                str(tmp_path / "params.out"),
                "--min-gene-set-size",
                "1",
                "--filter-gene-set-p",
                "1",
                "--max-gene-set-read-p",
                "1",
                "--prune-gene-sets",
                "1.1",
                "--weighted-prune-gene-sets",
                "1.1",
            ]
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                env=self._base_env(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            self.assertTrue((tmp_path / "gene_set_stats.out").exists())
            self.assertTrue((tmp_path / "gene_stats.out").exists())

    def test_gene_stats_input_respects_explicit_gene_universe_file(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gene_stats = tmp_path / "gene_stats.tsv"
            gene_stats.write_text(
                "Gene\tDirect\tCombined\nKCNJ11\t2.0\t2.5\nGCK\t1.0\t1.5\nOUTSIDE\t3.0\t3.5\n",
                encoding="utf-8",
            )
            gene_universe = tmp_path / "gene_universe.tsv"
            gene_universe.write_text(
                "Gene\nKCNJ11\nGCK\nHNF1A\nPAX4\nHNF1B\nHNF4A\nPDX1\nINS\nABCC8\nNEUROD1\n",
                encoding="utf-8",
            )
            gene_stats_out = tmp_path / "gene_stats.out"
            cmd = [
                sys.executable,
                "-m",
                "pigean",
                "beta_tildes",
                "--deterministic",
                "--linear",
                "--X-in",
                "tests/data/t2d_smoke/gene_set_list_input_subset_toy.txt",
                "--gene-map-in",
                "tests/data/model_small/portal_gencode.gene.map",
                "--gene-loc-file",
                "tests/data/model_small/NCBI37.3.plink.gene.loc",
                "--gene-stats-in",
                str(gene_stats),
                "--gene-stats-id-col",
                "Gene",
                "--gene-stats-log-bf-col",
                "Direct",
                "--gene-stats-combined-col",
                "Combined",
                "--gene-universe-in",
                str(gene_universe),
                "--gene-universe-id-col",
                "Gene",
                "--gene-set-stats-out",
                str(tmp_path / "gene_set_stats.out"),
                "--gene-stats-out",
                str(gene_stats_out),
                "--params-out",
                str(tmp_path / "params.out"),
                "--min-gene-set-size",
                "1",
                "--filter-gene-set-p",
                "1",
                "--max-gene-set-read-p",
                "1",
                "--prune-gene-sets",
                "1.1",
                "--weighted-prune-gene-sets",
                "1.1",
            ]
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                env=self._base_env(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
            output_text = gene_stats_out.read_text(encoding="utf-8")
            self.assertIn("HNF1A", output_text)
            self.assertIn("KCNJ11", output_text)
            self.assertNotIn("OUTSIDE", output_text)

    def test_multi_y_requires_gene_set_stats_out(self) -> None:
        proc = self._run("betas", "--multi-y-in", "traits.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("--multi-y-in requires --gene-set-stats-out", err)

    def test_multi_y_batch_override_requires_multi_y_input(self) -> None:
        proc = self._run("betas", "--multi-y-max-phenos-per-batch", "2")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("--multi-y-max-phenos-per-batch requires --multi-y-in", err)

    def test_multi_y_effective_config_round_trips(self) -> None:
        proc = self._run(
            "betas",
            "--multi-y-in",
            "traits.tsv",
            "--multi-y-id-col",
            "Gene",
            "--multi-y-pheno-col",
            "Trait",
            "--multi-y-log-bf-col",
            "Direct",
            "--multi-y-combined-col",
            "Combined",
            "--multi-y-prior-col",
            "Prior",
            "--multi-y-max-phenos-per-batch",
            "3",
            "--gene-set-stats-out",
            "out.tsv",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        options = payload["options"]
        self.assertEqual(options["multi_y_in"], "traits.tsv")
        self.assertEqual(options["multi_y_id_col"], "Gene")
        self.assertEqual(options["multi_y_pheno_col"], "Trait")
        self.assertEqual(options["multi_y_log_bf_col"], "Direct")
        self.assertEqual(options["multi_y_combined_col"], "Combined")
        self.assertEqual(options["multi_y_prior_col"], "Prior")
        self.assertEqual(options["multi_y_max_phenos_per_batch"], 3)

    def test_gene_stats_combined_write_filter_round_trips(self) -> None:
        proc = self._run(
            "gibbs",
            "--max-no-write-gene-combined",
            "1.5",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["options"]["max_no_write_gene_combined"], 1.5)

    def test_gene_stats_output_scope_round_trips(self) -> None:
        proc = self._run(
            "gibbs",
            "--gene-stats-output-scope",
            "current",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["options"]["gene_stats_output_scope"], "current")

    def test_gene_stats_output_scope_rejects_invalid_value(self) -> None:
        proc = self._run("gibbs", "--gene-stats-output-scope", "bad")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --gene-stats-output-scope must be one of: universe, current", err)

    def test_gene_stats_input_with_metric_z_zero_disables_qc_prefilter(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gene_stats = tmp_path / "gene_stats.tsv"
            gene_stats.write_text(
                """Gene\tDirect\tCombined
KCNJ11\t2.0\t2.5
PAX4\t1.8\t2.3
HNF1B\t1.6\t2.1
HNF4A\t1.4\t1.9
GCK\t1.2\t1.7
HNF1A\t1.1\t1.6
PDX1\t1.0\t1.5
INS\t0.9\t1.4
ABCC8\t0.8\t1.3
NEUROD1\t0.7\t1.2
""",
                encoding="utf-8",
            )
            for extra_flags in ([], ["--no-correct-betas-mean"]):
                out_prefix = tmp_path / ("out_no_correct_mean" if extra_flags else "out")
                cmd = [
                    sys.executable,
                    "-m",
                    "pigean",
                    "betas",
                    "--deterministic",
                    "--X-in",
                    "tests/data/t2d_smoke/gene_set_list_input_subset_toy.txt",
                    "--gene-map-in",
                    "tests/data/model_small/portal_gencode.gene.map",
                    "--gene-loc-file",
                    "tests/data/model_small/NCBI37.3.plink.gene.loc",
                    "--gene-stats-in",
                    str(gene_stats),
                    "--gene-stats-id-col",
                    "Gene",
                    "--gene-stats-log-bf-col",
                    "Direct",
                    "--gene-stats-combined-col",
                    "Combined",
                    "--gene-set-stats-out",
                    str(out_prefix.with_suffix('.gene_set_stats.out')),
                    "--gene-stats-out",
                    str(out_prefix.with_suffix('.gene_stats.out')),
                    "--params-out",
                    str(out_prefix.with_suffix('.params.out')),
                    "--min-gene-set-size",
                    "1",
                    "--filter-gene-set-p",
                    "1",
                    "--filter-gene-set-metric-z",
                    "0",
                    "--max-gene-set-read-p",
                    "1",
                    "--prune-gene-sets",
                    "1.1",
                    "--weighted-prune-gene-sets",
                    "1.1",
                ] + extra_flags
                proc = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    env=self._base_env(repo_root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                combined = (proc.stderr or "") + (proc.stdout or "")
                self.assertNotIn("unexpected internal error", combined)
                self.assertNotIn("unsupported operand type(s) for /: 'NoneType' and 'NoneType'", combined)
                self.assertNotIn("boolean index did not match indexed array along axis 0", combined)
                self.assertNotIn("Traceback", combined)

    def test_gene_stats_output_scope_defaults_to_active_universe(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gene_stats = tmp_path / "gene_stats.tsv"
            gene_stats.write_text(
                "Gene\tDirect\tCombined\nKCNJ11\t2.0\t2.5\nGCK\t1.0\t1.5\n",
                encoding="utf-8",
            )
            default_out = tmp_path / "default_gene_stats.out"
            current_out = tmp_path / "current_gene_stats.out"

            common_prefix = [
                sys.executable,
                "-m",
                "pigean",
                "beta_tildes",
                "--deterministic",
                "--linear",
                "--X-in",
                "tests/data/t2d_smoke/gene_set_list_input_subset_toy.txt",
                "--gene-map-in",
                "tests/data/model_small/portal_gencode.gene.map",
                "--gene-loc-file",
                "tests/data/model_small/NCBI37.3.plink.gene.loc",
                "--gene-stats-in",
                str(gene_stats),
                "--gene-stats-id-col",
                "Gene",
                "--gene-stats-log-bf-col",
                "Direct",
                "--gene-stats-combined-col",
                "Combined",
                "--gene-universe-from-y",
                "--gene-set-stats-out",
                str(tmp_path / "gene_set_stats.out"),
                "--params-out",
                str(tmp_path / "params.out"),
                "--min-gene-set-size",
                "1",
                "--filter-gene-set-p",
                "1",
                "--max-gene-set-read-p",
                "1",
                "--prune-gene-sets",
                "1.1",
                "--weighted-prune-gene-sets",
                "1.1",
            ]
            default_proc = subprocess.run(
                common_prefix + ["--gene-stats-out", str(default_out)],
                cwd=repo_root,
                env=self._base_env(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(default_proc.returncode, 0, msg=(default_proc.stderr or "") + (default_proc.stdout or ""))

            current_proc = subprocess.run(
                common_prefix + ["--gene-stats-out", str(current_out), "--gene-stats-output-scope", "current"],
                cwd=repo_root,
                env=self._base_env(repo_root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(current_proc.returncode, 0, msg=(current_proc.stderr or "") + (current_proc.stdout or ""))

            default_text = default_out.read_text(encoding="utf-8")
            current_text = current_out.read_text(encoding="utf-8")
            self.assertIn("KCNJ11", default_text)
            self.assertIn("GCK", default_text)
            self.assertNotIn("HNF1A", default_text)
            self.assertIn("HNF1A", current_text)

    def test_removed_min_post_burn_alias_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--min-post-burn-in", "50")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --min-post-burn-in has been removed; use --min-num-post-burn-in instead", err)

    def test_removed_burn_in_post_reserve_alias_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--burn-in-post-reserve", "50")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --burn-in-post-reserve has been removed; use --min-num-post-burn-in instead", err)

    def test_removed_stall_min_post_burn_alias_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--stall-min-post-burn-in", "50")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --stall-min-post-burn-in has been removed; use --stall-min-post-burn-samples instead", err)

    def test_removed_min_num_iter_alias_has_replacement_message(self) -> None:
        proc = self._run("gibbs", "--min-num-iter", "50")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --min-num-iter has been removed; use --min-num-post-burn-in instead", err)

    def test_deterministic_sets_seed_zero(self) -> None:
        proc = self._run("gibbs", "--deterministic", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 0)

    def test_deterministic_keeps_explicit_seed(self) -> None:
        proc = self._run("gibbs", "--deterministic", "--seed", "123", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 123)

    def test_experimental_hyper_threshold_requires_experimental_mode(self) -> None:
        proc = self._run(
            "gibbs",
            "--experimental-increase-hyper-if-betas-below",
            "0.01",
            "--print-effective-config",
        )
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("requires --experimental-hyper-mutation", err)

    def test_legacy_hyper_threshold_alias_warns_and_maps(self) -> None:
        proc = self._run(
            "gibbs",
            "--experimental-hyper-mutation",
            "--increase-hyper-if-betas-below",
            "0.02",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["options"]["experimental_hyper_mutation"])
        self.assertEqual(payload["options"]["experimental_increase_hyper_if_betas_below"], 0.02)
        err = proc.stderr or ""
        self.assertIn("legacy alias", err)

    def test_import_does_not_reset_python_random_seed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        snippet = r'''
import random
import sys
random.seed(12345)
expected = random.Random(12345).random()
sys.argv = ["pigean.py", "gibbs"]
import pigean  # noqa: F401
actual = random.random()
print(f"{actual:.17f}\t{expected:.17f}")
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        last_line = (proc.stdout or "").strip().splitlines()[-1]
        actual, expected = last_line.split("\t")
        self.assertEqual(actual, expected)

    def test_import_does_not_parse_invalid_argv(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        snippet = r'''
import sys
sys.argv = ["pigean.py", "--definitely-invalid-option"]
import pigean  # noqa: F401
print("ok")
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        self.assertEqual((proc.stdout or "").strip().splitlines()[-1], "ok")

    def test_main_accepts_argv_directly(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        snippet = r'''
import contextlib
import io
import json
import sys
import pigean as pigean
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = pigean.main(["gibbs", "--deterministic", "--print-effective-config"])
payload = json.loads(buf.getvalue())
print(json.dumps({"rc": rc, "mode": payload["mode"], "seed": payload["options"]["seed"]}, sort_keys=True))
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
        self.assertEqual(payload["rc"], 0)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertEqual(payload["seed"], 0)

    def test_prefilter_keep_mask_phewas_relax_uses_gene_set_axis(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        snippet = r'''
import json
import sys
import numpy as np
sys.argv = ["pigean.py", "gibbs"]
from pigean import app as pigean_app
mask = pigean_app._build_prefilter_keep_mask(
    p_values=np.array([0.8, 0.9, 1.0], dtype=float),
    beta_tildes=np.array([1.0, 1.0, 1.0], dtype=float),
    filter_gene_set_p=0.01,
    filter_using_phewas=True,
    p_values_phewas=np.array([[0.95, 0.95, 0.95], [0.95, 0.1, 0.95]], dtype=float),
    beta_tildes_phewas=np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float),
    increase_filter_gene_set_p=0.34,
    filter_negative=False,
)
print(json.dumps(mask.tolist()))
'''
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo_root,
            env=self._base_env(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
        self.assertEqual(payload, [True, True, False])

    def test_help_usage_uses_pigean_name(self) -> None:
        proc = self._run("gibbs", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Usage: python -m pigean", proc.stdout)

    def test_help_includes_core_and_expert_sections(self) -> None:
        proc = self._run("gibbs", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Core quickstart:", proc.stdout)
        self.assertIn("Alternative quickstart:", proc.stdout)
        self.assertIn("Use --help-expert", proc.stdout)
        self.assertIn("Core options:", proc.stdout)
        self.assertIn("Expert options:", proc.stdout)

    def test_default_help_hides_expert_flags(self) -> None:
        proc = self._run("gibbs", "--help")
        self.assertEqual(proc.returncode, 0)
        self.assertNotIn("--run-phewas", proc.stdout)
        self.assertNotIn("--run-phewas-from-gene-phewas-stats-in", proc.stdout)
        self.assertNotIn("--huge-statistics-in", proc.stdout)
        self.assertIn("--gene-list", proc.stdout)
        self.assertNotIn("--positive-controls-list", proc.stdout)

    def test_help_expert_includes_set_b_flags(self) -> None:
        proc = self._run("gibbs", "--help-expert")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--run-phewas", proc.stdout)
        self.assertNotIn("--run-phewas-from-gene-phewas-stats-in", proc.stdout)
        self.assertIn("--phewas-comparison-set", proc.stdout)
        self.assertIn("run the optional gene-level phewas output stage", proc.stdout)
        self.assertIn("--gene-stats-in", proc.stdout)
        self.assertIn("use precomputed gene-level statistics", proc.stdout)
        self.assertIn("--huge-statistics-in", proc.stdout)
        self.assertIn("read precomputed HuGE statistics cache", proc.stdout)
        self.assertIn("--retain-all-beta-uncorrected", proc.stdout)
        self.assertIn("--independent-betas-only", proc.stdout)

    def test_cli_manifest_tiers_cover_gene_list_and_recent_set_b_flags(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root / "src"))
        try:
            pigean_cli = importlib.import_module("pigean.cli")
        finally:
            del sys.path[0]
        metadata = pigean_cli.get_cli_manifest_metadata()

        self.assertEqual(metadata["--gene-list"]["category"], "method_required")
        self.assertEqual(metadata["--gene-list"]["public_visibility"], "normal")
        self.assertEqual(metadata["--gene-list"]["documentation_target"], "core_help")
        self.assertEqual(metadata["--gene-list-in"]["category"], "method_required")
        self.assertEqual(metadata["--gene-list-all-in"]["category"], "method_required")

        self.assertEqual(metadata["--positive-controls-list"]["category"], "compat_alias")
        self.assertEqual(metadata["--positive-controls-list"]["public_visibility"], "expert")
        self.assertEqual(metadata["--positive-controls-in"]["category"], "compat_alias")
        self.assertEqual(metadata["--positive-controls-all-in"]["category"], "compat_alias")
        self.assertEqual(metadata["--run-phewas"]["category"], "method_optional")
        self.assertEqual(metadata["--run-phewas"]["public_visibility"], "expert")
        self.assertEqual(metadata["--run-phewas"]["documentation_target"], "advanced_workflows")
        self.assertEqual(metadata["--run-phewas-from-gene-phewas-stats-in"]["category"], "compat_alias")
        self.assertEqual(metadata["--run-phewas-from-gene-phewas-stats-in"]["public_visibility"], "hidden")

        self.assertEqual(metadata["--phewas-comparison-set"]["category"], "method_optional")
        self.assertEqual(metadata["--phewas-comparison-set"]["public_visibility"], "expert")
        self.assertEqual(metadata["--phewas-comparison-set"]["documentation_target"], "advanced_workflows")
        self.assertEqual(metadata["--retain-all-beta-uncorrected"]["category"], "method_optional")
        self.assertEqual(metadata["--retain-all-beta-uncorrected"]["public_visibility"], "expert")
        self.assertEqual(metadata["--retain-all-beta-uncorrected"]["documentation_target"], "advanced_workflows")
        self.assertEqual(metadata["--independent-betas-only"]["category"], "method_optional")
        self.assertEqual(metadata["--independent-betas-only"]["public_visibility"], "expert")
        self.assertEqual(metadata["--independent-betas-only"]["documentation_target"], "advanced_workflows")

    def test_independent_betas_only_requires_betas_mode(self) -> None:
        proc = self._run("gibbs", "--independent-betas-only")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("currently supports only betas mode", err)

    def test_independent_betas_only_implies_retain_all_beta_uncorrected(self) -> None:
        proc = self._run("betas", "--independent-betas-only", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["options"]["independent_betas_only"])
        self.assertTrue(payload["options"]["retain_all_beta_uncorrected"])

    def test_huge_statistics_out_requires_gwas_in(self) -> None:
        proc = self._run("gibbs", "--huge-statistics-out", "cache_prefix")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --huge-statistics-out requires --gwas-in", err)

    def test_huge_statistics_in_and_out_conflict(self) -> None:
        proc = self._run(
            "gibbs",
            "--huge-statistics-in",
            "cache_prefix",
            "--huge-statistics-out",
            "cache_prefix_out",
        )
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Do not pass both --huge-statistics-in and --huge-statistics-out", err)

    def test_eaggl_bundle_out_requires_tar_extension(self) -> None:
        proc = self._run("gibbs", "--eaggl-bundle-out", "handoff_bundle.txt")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --eaggl-bundle-out must end with .tar, .tar.gz, or .tgz", err)

    def test_run_phewas_from_gene_phewas_stats_requires_output_path(self) -> None:
        proc = self._run("beta_tildes", "--run-phewas", "--gene-phewas-stats-in", "x.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("requires --phewas-stats-out", err)

    def test_phewas_comparison_set_requires_run_phewas_stage(self) -> None:
        proc = self._run("beta_tildes", "--phewas-comparison-set", "diagnostic")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("requires --run-phewas", err)

    def test_phewas_comparison_set_rejects_unknown_value(self) -> None:
        proc = self._run(
            "beta_tildes",
            "--run-phewas",
            "--gene-phewas-stats-in",
            "x.tsv",
            "--phewas-stats-out",
            "out.tsv",
            "--phewas-comparison-set",
            "all",
        )
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("must be one of: matched, diagnostic", err)

    def test_run_phewas_requires_gene_phewas_input(self) -> None:
        proc = self._run("beta_tildes", "--run-phewas", "--phewas-stats-out", "out.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("requires --gene-phewas-stats-in", err)

    def test_legacy_run_phewas_alias_normalizes_to_run_flag(self) -> None:
        proc = self._run(
            "beta_tildes",
            "--run-phewas-from-gene-phewas-stats-in",
            "x.tsv",
            "--phewas-stats-out",
            "out.tsv",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["options"]["run_phewas"])
        self.assertEqual(payload["options"]["gene_phewas_bfs_in"], "x.tsv")
        self.assertEqual(payload["options"]["run_phewas_input"], "x.tsv")

    def test_gene_set_stats_column_option_requires_gene_set_stats_in(self) -> None:
        proc = self._run("gibbs", "--gene-set-stats-beta-col", "beta")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --gene-set-stats-beta-col requires --gene-set-stats-in", err)

    def test_gene_stats_column_option_requires_gene_stats_in(self) -> None:
        proc = self._run("gibbs", "--gene-stats-log-bf-col", "log_bf")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --gene-stats-log-bf-col requires --gene-stats-in", err)

    def test_gene_phewas_input_requires_consumer_flag(self) -> None:
        proc = self._run("gibbs", "--gene-phewas-stats-in", "phewas.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Option --gene-phewas-stats-in requires either --betas-uncorrected-from-phewas", err)

    def test_pops_mode_defaults_are_exposed_in_effective_config(self) -> None:
        proc = self._run("pops", "--deterministic", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "pops")
        self.assertTrue(payload["options"]["linear"])
        self.assertEqual(payload["options"]["update_hyper"], "none")
        self.assertTrue(payload["options"]["cross_val"])

    def test_sim_mode_is_supported_in_effective_config(self) -> None:
        proc = self._run("sim", "--deterministic", "--print-effective-config")
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "sim")
        self.assertTrue(payload["options"]["deterministic"])
        self.assertEqual(payload["options"]["seed"], 0)

    def test_gene_stats_in_option_round_trips_in_effective_config(self) -> None:
        proc = self._run(
            "gibbs",
            "--gene-stats-in",
            "tests/data/t2d_smoke/mody.gene.list",
            "--print-effective-config",
        )
        self.assertEqual(proc.returncode, 0, msg=(proc.stderr or "") + (proc.stdout or ""))
        payload = json.loads(proc.stdout)
        self.assertEqual(payload["mode"], "gibbs")
        self.assertEqual(payload["options"]["gene_stats_in"], "tests/data/t2d_smoke/mody.gene.list")

    def test_positive_controls_only_requires_explicit_gene_universe(self) -> None:
        proc = self._run("gibbs", "--positive-controls-list", "INS")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("requires an explicit gene universe", err)
        self.assertIn("--gene-universe-in", err)
        self.assertIn("--gene-universe-from-y", err)
        self.assertIn("--gene-universe-from-x", err)

    def test_config_removed_gene_bfs_key_has_replacement_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"gene_bfs_in": "dummy.txt"}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'gene_bfs_in' has been removed", err)

    def test_config_gene_set_stats_column_requires_gene_set_stats_in(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"gene_set_stats_beta_col": "beta"}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Option --gene-set-stats-beta-col requires --gene-set-stats-in", err)

    def test_config_removed_gene_zs_key_has_removed_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"gene_zs_in": "dummy.txt"}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'gene_zs_in' has been removed", err)
            self.assertIn("is no longer supported", err)

    def test_config_removed_sigma_key_has_removed_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"chisq_threshold": 5}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'chisq_threshold' has been removed", err)
            self.assertIn("is no longer supported", err)

    def test_config_removed_run_gls_key_has_removed_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"run_gls": True}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'run_gls' has been removed", err)
            self.assertIn("is no longer supported", err)

    def test_config_removed_min_post_burn_alias_has_replacement_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"min_post_burn_in": 50}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'min_post_burn_in' has been removed", err)
            self.assertIn("use 'min_num_post_burn_in' (CLI: --min-num-post-burn-in) instead", err)

    def test_config_removed_min_num_iter_alias_has_replacement_message(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"min_num_iter": 50}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'min_num_iter' has been removed", err)
            self.assertIn("use 'min_num_post_burn_in' (CLI: --min-num-post-burn-in) instead", err)

    def test_factor_mode_disabled_in_pigean(self) -> None:
        proc = self._run("factor")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("Mode 'factor' is not available in pigean.py after repository split", err)

    def test_factor_output_flag_disabled_in_pigean(self) -> None:
        proc = self._run("gibbs", "--factors-out", "factors.tsv")
        self.assertNotEqual(proc.returncode, 0)
        err = (proc.stderr or "") + (proc.stdout or "")
        self.assertIn("option --factors-out moved to eaggl.py after repository split", err)

    def test_config_factor_option_disabled_in_pigean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "cfg.json"
            cfg_path.write_text(
                json.dumps({"mode": "gibbs", "options": {"anchor_genes": ["INS"]}}),
                encoding="utf-8",
            )
            proc = self._run("--config", str(cfg_path))
            self.assertNotEqual(proc.returncode, 0)
            err = (proc.stderr or "") + (proc.stdout or "")
            self.assertIn("Config key 'anchor_genes' moved to eaggl.py after repository split", err)


if __name__ == "__main__":
    unittest.main()
