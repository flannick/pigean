from __future__ import annotations

import json
import optparse
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
import numpy as np
import scipy.sparse as sparse


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

    def test_resolve_column_index_supports_name_and_1_based_index(self) -> None:
        header = ["gene", "log_bf", "prior"]
        self.assertEqual(pegs_utils.resolve_column_index("log_bf", header), 1)
        self.assertEqual(pegs_utils.resolve_column_index(3, header), 2)
        self.assertEqual(pegs_utils.resolve_column_index("3", header), 2)
        self.assertIsNone(pegs_utils.resolve_column_index("missing", header, require_match=False))
        with self.assertRaisesRegex(ValueError, "Could not find match for column"):
            pegs_utils.resolve_column_index("missing", header)
        with self.assertRaisesRegex(ValueError, "1-based"):
            pegs_utils.resolve_column_index(0, header)

    def test_construct_map_to_ind(self) -> None:
        mapping = pegs_utils.construct_map_to_ind(["A", "B", "C"])
        self.assertEqual(mapping, {"A": 0, "B": 1, "C": 2})

    def test_clean_chrom_name(self) -> None:
        self.assertEqual(pegs_utils.clean_chrom_name("chr1"), "1")
        self.assertEqual(pegs_utils.clean_chrom_name("1"), "1")
        self.assertEqual(pegs_utils.clean_chrom_name("chrX"), "X")
        self.assertIsNone(pegs_utils.clean_chrom_name(None))

    def test_comma_callback_helpers(self) -> None:
        class _Opt:
            dest = "value"

        class _Parser:
            class _Vals:
                value = None
            values = _Vals()

        parser = _Parser()
        pegs_utils.callback_set_comma_separated_args(_Opt(), None, "a,b,c", parser)
        self.assertEqual(parser.values.value, ["a", "b", "c"])

        pegs_utils.callback_set_comma_separated_args_as_float(_Opt(), None, "1,2.5,3", parser)
        self.assertEqual(parser.values.value, [1.0, 2.5, 3.0])

        pegs_utils.callback_set_comma_separated_args_as_set(_Opt(), None, "x,y,x", parser)
        self.assertEqual(parser.values.value, {"x", "y"})

    def test_open_optional_log_handle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "log.txt"
            fh = pegs_utils.open_optional_log_handle(str(path), default_stream=sys.stderr, mode="w")
            try:
                fh.write("hello\n")
            finally:
                fh.close()
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "hello\n")

        self.assertIs(
            pegs_utils.open_optional_log_handle(None, default_stream=sys.stderr, mode="w"),
            sys.stderr,
        )

    def test_fail_removed_cli_aliases_uses_formatter_and_exit(self) -> None:
        writes: list[str] = []
        exits: list[int] = []

        def _write(msg: str) -> None:
            writes.append(msg)

        def _exit(code: int) -> None:
            exits.append(code)
            raise RuntimeError("stop")

        def _fmt(flag: str, replacement, context: str, config_path=None) -> str:
            return "MSG %s %s %s" % (flag, replacement, context)

        with self.assertRaisesRegex(RuntimeError, "stop"):
            pegs_utils.fail_removed_cli_aliases(
                ["--old-flag", "--other"],
                {"old_flag": "--new-flag"},
                format_removed_option_message_fn=_fmt,
                stderr_write_fn=_write,
                exit_fn=_exit,
            )

        self.assertEqual(exits, [2])
        self.assertEqual(writes, ["MSG --old-flag --new-flag cli\n"])

    def test_apply_cli_config_overrides_applies_mode_and_options(self) -> None:
        parser = optparse.OptionParser()
        parser.add_option("", "--config", default=None)
        parser.add_option("", "--foo", default=None)
        parser.add_option("", "--input-file", dest="input_file", default=None)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg_path = td_path / "config.json"
            cfg = {
                "mode": "gibbs",
                "options": {
                    "foo": "from_config",
                    "input_file": "relative.tsv",
                },
            }
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

            options, args = parser.parse_args(
                ["--config", str(cfg_path), "--foo", "from_cli"]
            )
            (
                options,
                args,
                config_mode,
                cli_specified_dests,
                config_specified_dests,
            ) = pegs_utils.apply_cli_config_overrides(
                options,
                args,
                parser,
                ["--config", str(cfg_path), "--foo", "from_cli"],
                resolve_path_fn=pegs_utils.resolve_config_path_value,
                is_path_like_dest_fn=pegs_utils.is_path_like_dest,
                early_warn_fn=lambda _m: None,
                bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
                removed_option_replacements={},
                format_removed_option_message_fn=pegs_utils.format_removed_option_message,
                track_config_specified_dests=True,
            )

            self.assertEqual(config_mode, "gibbs")
            self.assertEqual(options.foo, "from_cli")
            self.assertTrue(options.input_file.endswith("relative.tsv"))
            self.assertIn("foo", cli_specified_dests)
            self.assertIn("input_file", config_specified_dests)

    def test_harmonize_cli_mode_args_prefers_cli_and_fills_from_config(self) -> None:
        warns = []
        self.assertEqual(
            pegs_utils.harmonize_cli_mode_args([], "gibbs", early_warn_fn=warns.append),
            ["gibbs"],
        )
        self.assertEqual(
            pegs_utils.harmonize_cli_mode_args(["priors"], "gibbs", early_warn_fn=warns.append),
            ["priors"],
        )
        self.assertEqual(len(warns), 1)
        self.assertIn("Config mode 'gibbs' differs from CLI mode 'priors'", warns[0])

    def test_coerce_option_int_list(self) -> None:
        self.assertEqual(pegs_utils.coerce_option_int_list(["1", "2", 3], "--x", self.fail), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "invalid integer list"):
            pegs_utils.coerce_option_int_list(
                ["1", "bad"],
                "--x",
                lambda message: (_ for _ in ()).throw(ValueError(message)),
            )

    def test_initialize_cli_logging(self) -> None:
        class _Options:
            log_file = None
            warnings_file = None
            debug_level = None

        class _Capture:
            def __init__(self):
                self.items = []

            def write(self, message):
                self.items.append(message)

            def flush(self):
                return None

        stream = _Capture()
        state = pegs_utils.initialize_cli_logging(_Options(), stderr_stream=stream, default_debug_level=1)
        state["log"]("info message", level=state["INFO"])
        state["warn"]("warn message")
        text = "".join(stream.items)
        self.assertIn("info message", text)
        self.assertIn("Warning: warn message", text)

    def test_infer_columns_from_table_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gwas.tsv"
            path.write_text(
                "CHROM\tPOS\tP\tBETA\tSE\tN\n1\t12345\t0.02\t0.1\t0.03\t10000\n",
                encoding="utf-8",
            )

            result = pegs_utils.infer_columns_from_table_file(
                str(path),
                lambda p: open(p, "rt", encoding="utf-8"),
            )
            self.assertIn("CHROM", set(result[2]))
            self.assertIn("POS", set(result[3]))
            self.assertIn("P", set(result[5]))
            self.assertIn("BETA", set(result[6]))
            self.assertIn("SE", set(result[7]))
            self.assertIn("N", set(result[9]))

    def test_needs_gwas_column_detection(self) -> None:
        self.assertTrue(
            pegs_utils.needs_gwas_column_detection(
                None, None, None, "P", None, "SE", "N", None
            )
        )
        self.assertFalse(
            pegs_utils.needs_gwas_column_detection(
                "POS", "CHROM", None, "P", None, "SE", "N", None
            )
        )

    def test_autodetect_gwas_columns(self) -> None:
        inferred = (
            np.array([]),  # gene id
            np.array([]),  # var id
            np.array(["CHROM"]),
            np.array(["POS"]),
            np.array([]),  # locus
            np.array(["P"]),
            np.array(["BETA"]),
            np.array(["SE"]),
            np.array(["AF"]),
            np.array(["N"]),
            "CHROM POS P BETA SE AF N",
        )
        out = pegs_utils.autodetect_gwas_columns(
            "dummy",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            infer_columns_fn=lambda _gwas: inferred,
            log_fn=lambda _m: None,
            bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
            debug_just_check_header=False,
        )
        self.assertEqual(out[0], "POS")
        self.assertEqual(out[1], "CHROM")
        self.assertEqual(out[3], "P")
        self.assertEqual(out[4], "BETA")
        self.assertEqual(out[5], "SE")
        self.assertEqual(out[6], "AF")
        self.assertEqual(out[7], "N")

    def test_huge_statistics_path_and_bundle_helpers(self) -> None:
        self.assertTrue(pegs_utils.is_huge_statistics_bundle_path("x.tar.gz"))
        self.assertTrue(pegs_utils.is_huge_statistics_bundle_path("x.tgz"))
        self.assertTrue(pegs_utils.is_huge_statistics_bundle_path("x.tar"))
        self.assertFalse(pegs_utils.is_huge_statistics_bundle_path("x.txt"))

        paths = pegs_utils.get_huge_statistics_paths_for_prefix("/tmp/demo")
        self.assertIn("meta", paths)
        self.assertTrue(paths["meta"].endswith(".huge.meta.json.gz"))

    def test_huge_statistics_numeric_vector_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "vals.txt"
            pegs_utils.write_numeric_vector_file(
                str(out_file),
                [1.0, 2.5, 3.0],
                open_text_fn=lambda p, mode=None: open(p, mode or "rt", encoding="utf-8"),
                value_type=float,
            )
            vals = pegs_utils.read_numeric_vector_file(
                str(out_file),
                open_text_fn=lambda p, mode=None: open(p, mode or "rt", encoding="utf-8"),
                value_type=float,
            )
            np.testing.assert_allclose(vals, np.array([1.0, 2.5, 3.0]))

    def test_build_huge_statistics_matrix_row_genes(self) -> None:
        self.assertEqual(
            pegs_utils.build_huge_statistics_matrix_row_genes(["G1"], ["E1", "E2"], 2, bail_fn=self.fail),
            ["G1", "E1"],
        )
        with self.assertRaisesRegex(ValueError, "matrix rows"):
            pegs_utils.build_huge_statistics_matrix_row_genes(
                ["G1", "G2"],
                [],
                1,
                bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
            )

    def test_coerce_runtime_state_dict(self) -> None:
        class _State:
            def __init__(self):
                self.value = 1

        self.assertEqual(pegs_utils.coerce_runtime_state_dict({"x": 1}, bail_fn=self.fail), {"x": 1})
        self.assertEqual(pegs_utils.coerce_runtime_state_dict(_State(), bail_fn=self.fail)["value"], 1)

    def test_huge_meta_apply_combine_and_validate_helpers(self) -> None:
        runtime = {}
        meta = {
            "huge_signal_max_closest_gene_prob": 0.9,
            "huge_cap_region_posterior": True,
            "huge_scale_region_posterior": False,
            "huge_phantom_region_posterior": False,
            "huge_allow_evidence_of_absence": False,
            "huge_sparse_mode": False,
            "huge_signals": [["1", 100, 1.0, None]],
            "gene_covariate_names": ["c1"],
            "gene_covariate_directions": [1.0],
            "gene_covariate_intercept_index": 0,
            "gene_covariate_slope_defaults": [0.1],
            "total_qc_metric_betas_defaults": [0.2],
            "total_qc_metric_intercept_defaults": 0.3,
            "total_qc_metric2_betas_defaults": [0.4],
            "total_qc_metric2_intercept_defaults": 0.5,
        }
        pegs_utils.apply_huge_statistics_meta_to_runtime(runtime, meta)
        self.assertEqual(runtime["huge_signal_max_closest_gene_prob"], 0.9)
        self.assertEqual(len(runtime["huge_signals"]), 1)

        runtime["gene_to_gwas_huge_score"] = {"G1": 1.0}
        runtime["gene_to_exomes_huge_score"] = {"G1": 0.5, "G2": 1.5}
        pegs_utils.combine_runtime_huge_scores(runtime)
        self.assertEqual(runtime["gene_to_huge_score"]["G1"], 1.5)
        self.assertEqual(runtime["gene_to_huge_score"]["G2"], 1.5)

        runtime["huge_signal_bfs"] = sparse.csc_matrix(np.zeros((2, 1)))
        pegs_utils.validate_huge_statistics_loaded_shapes(runtime, ["G1", "G2"], bail_fn=self.fail)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            pegs_utils.validate_huge_statistics_loaded_shapes(
                runtime,
                ["G1"],
                bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
            )

    def test_huge_sparse_vector_helpers_roundtrip_with_callbacks(self) -> None:
        values_map = {
            "bfs_data": np.array([1.0, 2.0]),
            "bfs_indices": np.array([0, 1], dtype=int),
            "bfs_indptr": np.array([0, 1, 2], dtype=int),
            "bfs_reg_data": np.array([1.5, 2.5]),
            "bfs_reg_indices": np.array([0, 1], dtype=int),
            "bfs_reg_indptr": np.array([0, 1, 2], dtype=int),
            "signal_posteriors": np.array([0.1, 0.2]),
            "signal_posteriors_for_regression": np.array([0.1, 0.2]),
            "signal_sum_gene_cond_probabilities": np.array([0.3, 0.4]),
            "signal_sum_gene_cond_probabilities_for_regression": np.array([0.3, 0.4]),
            "signal_mean_gene_pos": np.array([10.0, 20.0]),
            "signal_mean_gene_pos_for_regression": np.array([10.0, 20.0]),
        }
        paths = {k: k for k in values_map.keys()}
        runtime = {}
        meta = {
            "huge_signal_bfs_shape": [2, 2],
            "huge_signal_bfs_for_regression_shape": [2, 2],
        }

        def _reader(path, value_type=float):
            arr = values_map[path]
            if value_type == int:
                return arr.astype(int)
            return arr.astype(float)

        pegs_utils.load_huge_statistics_sparse_and_vectors(runtime, paths, meta, read_vector_fn=_reader)
        self.assertEqual(runtime["huge_signal_bfs"].shape, (2, 2))
        self.assertEqual(runtime["huge_signal_posteriors"].shape[0], 2)

        written = {}

        def _writer(path, values, value_type=float):
            written[path] = np.array(values)

        pegs_utils.write_huge_statistics_runtime_vectors(paths, runtime, write_vector_fn=_writer)
        pegs_utils.write_huge_statistics_sparse_components(
            paths,
            runtime["huge_signal_bfs"],
            runtime["huge_signal_bfs_for_regression"],
            write_vector_fn=_writer,
        )
        self.assertIn("signal_posteriors", written)
        self.assertIn("bfs_data", written)

    def test_complete_p_beta_se_fills_missing_values(self) -> None:
        p = np.array([np.nan, 0.05, 0.2], dtype=float)
        beta = np.array([np.nan, np.nan, 0.2], dtype=float)
        se = np.array([np.nan, 0.1, 0.0], dtype=float)
        warnings = []

        out_p, out_beta, out_se = pegs_utils.complete_p_beta_se(
            p,
            beta,
            se,
            warn_fn=warnings.append,
        )
        self.assertEqual(out_p.shape, (3,))
        self.assertEqual(out_beta.shape, (3,))
        self.assertEqual(out_se.shape, (3,))
        self.assertFalse(np.any(np.isnan(out_p)))
        self.assertFalse(np.any(np.isnan(out_beta)))
        self.assertFalse(np.any(np.isnan(out_se)))
        self.assertTrue(np.any(out_se == 1.0))
        self.assertGreaterEqual(len(warnings), 1)

    def test_parse_gene_bfs_file_from_log_bf(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gene_stats.tsv"
            path.write_text(
                "Gene\tlog_bf\tcombined\tprior\nGENE_A\t1.5\t0.2\t0.1\nGENE_B\tNA\t0.3\t0.2\n",
                encoding="utf-8",
            )
            parsed = pegs_utils.parse_gene_bfs_file(
                str(path),
                gene_bfs_id_col="Gene",
                gene_bfs_log_bf_col=None,
                gene_bfs_combined_col=None,
                gene_bfs_prob_col=None,
                gene_bfs_prior_col=None,
                background_log_bf=0.0,
                gene_label_map={"GENE_A": "GENE_A_ALIAS"},
                open_text_fn=lambda p: open(p, "rt", encoding="utf-8"),
                get_col_fn=lambda col, header, required=True: pegs_utils.resolve_column_index(
                    col, header, require_match=required
                ),
                warn_fn=lambda _m: None,
                bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
            )
            self.assertIn("GENE_A_ALIAS", parsed.gene_in_bfs)
            self.assertNotIn("GENE_B", parsed.gene_in_bfs)
            self.assertAlmostEqual(parsed.gene_in_combined["GENE_A_ALIAS"], 0.2)
            self.assertAlmostEqual(parsed.gene_in_priors["GENE_A_ALIAS"], 0.1)

    def test_parse_gene_bfs_file_from_prob(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gene_probs.tsv"
            path.write_text("Gene\tprob\nGENE_A\t0.8\n", encoding="utf-8")
            parsed = pegs_utils.parse_gene_bfs_file(
                str(path),
                gene_bfs_id_col="Gene",
                gene_bfs_log_bf_col=None,
                gene_bfs_combined_col=None,
                gene_bfs_prob_col="prob",
                gene_bfs_prior_col=None,
                background_log_bf=0.0,
                gene_label_map=None,
                open_text_fn=lambda p: open(p, "rt", encoding="utf-8"),
                get_col_fn=lambda col, header, required=True: pegs_utils.resolve_column_index(
                    col, header, require_match=required
                ),
                warn_fn=lambda _m: None,
                bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
            )
            expected = np.log(0.8 / 0.2)
            self.assertAlmostEqual(parsed.gene_in_bfs["GENE_A"], expected)
            self.assertIsInstance(parsed.gene_in_combined, dict)
            self.assertEqual(len(parsed.gene_in_combined), 0)

    def test_parse_gene_covariates_file_and_align_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gene_covs.tsv"
            path.write_text(
                "Gene\tC1\tC2\nGENE_A\t1.0\t2.0\nGENE_X\t3.0\t4.0\n",
                encoding="utf-8",
            )
            parsed = pegs_utils.parse_gene_covariates_file(
                str(path),
                gene_covs_id_col="Gene",
                open_text_fn=lambda p: open(p, "rt", encoding="utf-8"),
                get_col_fn=lambda col, header, required=True: pegs_utils.resolve_column_index(
                    col, header, require_match=required
                ),
                log_fn=lambda _m: None,
                warn_fn=lambda _m: None,
                bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
            )
            self.assertEqual(parsed.cov_names, ["C1", "C2"])
            self.assertIn("GENE_A", parsed.gene_to_covs)

            gene_bfs, extra_genes, extra_bfs = pegs_utils.align_gene_scalar_map(
                {"GENE_A": 1.5, "GENE_X": 2.5},
                genes=["GENE_A", "GENE_B"],
                gene_to_ind={"GENE_A": 0, "GENE_B": 1},
            )
            np.testing.assert_allclose(gene_bfs[:1], np.array([1.5]))
            self.assertTrue(np.isnan(gene_bfs[1]))
            self.assertEqual(extra_genes, ["GENE_X"])
            np.testing.assert_allclose(extra_bfs, np.array([2.5]))

            gene_covs, extra_cov_genes, extra_covs = pegs_utils.align_gene_vector_map(
                parsed.gene_to_covs,
                num_values=2,
                genes=["GENE_A", "GENE_B"],
                gene_to_ind={"GENE_A": 0, "GENE_B": 1},
            )
            np.testing.assert_allclose(gene_covs[0, :], np.array([1.0, 2.0]))
            self.assertTrue(np.isnan(gene_covs[1, 0]))
            self.assertEqual(extra_cov_genes, ["GENE_X"])
            self.assertEqual(extra_covs.shape, (1, 2))

    def test_parse_gene_set_statistics_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gene_set_stats.tsv"
            path.write_text(
                "Gene_Set\texp_beta\tp\tbeta_uncorrected\nSET_A\t2.0\t0.05\t0.4\nSET_A\t2.5\t0.05\t0.5\n",
                encoding="utf-8",
            )
            parsed = pegs_utils.parse_gene_set_statistics_file(
                str(path),
                stats_id_col="Gene_Set",
                stats_exp_beta_tilde_col="exp_beta",
                stats_beta_tilde_col=None,
                stats_p_col="p",
                stats_se_col=None,
                stats_beta_col=None,
                stats_beta_uncorrected_col="beta_uncorrected",
                ignore_negative_exp_beta=False,
                max_gene_set_p=None,
                min_gene_set_beta=None,
                min_gene_set_beta_uncorrected=None,
                open_text_fn=lambda p: open(p, "rt", encoding="utf-8"),
                get_col_fn=lambda col, header, required=True: pegs_utils.resolve_column_index(
                    col, header, require_match=required
                ),
                log_fn=lambda _m: None,
                warn_fn=lambda _m: None,
                bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
            )
            self.assertTrue(parsed.need_to_take_log)
            self.assertTrue(parsed.has_beta_tilde)
            self.assertTrue(parsed.has_p_or_se)
            self.assertFalse(parsed.has_beta)
            self.assertTrue(parsed.has_beta_uncorrected)
            self.assertEqual(len(parsed.records), 1)
            beta_tilde, p, _se, z, _beta, beta_uncorrected = parsed.records["SET_A"]
            self.assertAlmostEqual(beta_tilde, np.log(2.0))
            self.assertAlmostEqual(p, 0.05)
            self.assertTrue(np.isfinite(z))
            self.assertAlmostEqual(beta_uncorrected, 0.4)

    def test_parse_gene_phewas_bfs_file_deduplicates_and_maps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gene_phewas.tsv"
            path.write_text(
                "Gene\tPheno\tlog_bf\tcombined\tprior\n"
                "GENE_A\tP1\t1.0\t0.2\t0.1\n"
                "GENE_A\tP1\t2.0\t0.3\t0.2\n"
                "GENE_B\tP2\t3.0\t0.4\t0.3\n",
                encoding="utf-8",
            )
            parsed = pegs_utils.parse_gene_phewas_bfs_file(
                str(path),
                gene_phewas_bfs_id_col="Gene",
                gene_phewas_bfs_pheno_col="Pheno",
                gene_phewas_bfs_log_bf_col=None,
                gene_phewas_bfs_combined_col=None,
                gene_phewas_bfs_prior_col=None,
                min_value=None,
                max_num_entries_at_once=2,
                existing_phenos=["P0"],
                existing_pheno_to_ind={"P0": 0},
                gene_to_ind={"GENE_A_ALIAS": 0, "GENE_B": 1},
                gene_label_map={"GENE_A": "GENE_A_ALIAS"},
                phewas_gene_to_x_gene=None,
                open_text_fn=lambda p: open(p, "rt", encoding="utf-8"),
                get_col_fn=lambda col, header, required=True: pegs_utils.resolve_column_index(
                    col, header, require_match=required
                ),
                bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
                warn_fn=lambda _m: None,
            )

            self.assertEqual(parsed.phenos, ["P0", "P1", "P2"])
            self.assertEqual(parsed.pheno_to_ind["P1"], 1)
            self.assertEqual(parsed.pheno_to_ind["P2"], 2)
            self.assertEqual(parsed.row.shape[0], 2)
            self.assertEqual(parsed.col.shape[0], 2)
            np.testing.assert_allclose(parsed.Ys, np.array([1.0, 3.0]))
            np.testing.assert_allclose(parsed.combineds, np.array([0.2, 0.4]))
            np.testing.assert_allclose(parsed.priors, np.array([0.1, 0.3]))

    def test_parse_gene_phewas_bfs_file_fallbacks_to_tab_delim(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gene_phewas_tab.tsv"
            path.write_text(
                "Gene\tPheno\tcombined\n"
                "GENE_A\tType 2 diabetes\t0.7\n",
                encoding="utf-8",
            )
            parsed = pegs_utils.parse_gene_phewas_bfs_file(
                str(path),
                gene_phewas_bfs_id_col="Gene",
                gene_phewas_bfs_pheno_col="Pheno",
                gene_phewas_bfs_log_bf_col=None,
                gene_phewas_bfs_combined_col="combined",
                gene_phewas_bfs_prior_col=None,
                min_value=0.5,
                max_num_entries_at_once=100,
                existing_phenos=None,
                existing_pheno_to_ind=None,
                gene_to_ind={"GENE_A": 0},
                gene_label_map=None,
                phewas_gene_to_x_gene=None,
                open_text_fn=lambda p: open(p, "rt", encoding="utf-8"),
                get_col_fn=lambda col, header, required=True: pegs_utils.resolve_column_index(
                    col, header, require_match=required
                ),
                bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
                warn_fn=lambda _m: None,
            )
            self.assertEqual(parsed.phenos, ["Type 2 diabetes"])
            self.assertEqual(parsed.row.tolist(), [0])
            self.assertEqual(parsed.col.tolist(), [0])
            self.assertIsNone(parsed.Ys)
            np.testing.assert_allclose(parsed.combineds, np.array([0.7]))

    def test_runtime_state_dataclass_roundtrip_helpers(self) -> None:
        class _Runtime:
            pass

        rt = _Runtime()
        rt.Y = np.array([1.0, 2.0])
        rt.Y_for_regression = np.array([0.5, 1.5])
        rt.Y_exomes = None
        rt.Y_positive_controls = None
        rt.Y_case_counts = None
        rt.y_var = 1.2
        rt.y_corr = None
        rt.y_corr_cholesky = None
        rt.y_corr_sparse = None
        rt.Y_w = None
        rt.Y_fw = None
        rt.y_w_var = None
        rt.y_w_mean = None
        rt.y_fw_var = None
        rt.y_fw_mean = None

        rt.p = 0.1
        rt.sigma2 = 0.2
        rt.sigma_power = 2.0
        rt.sigma2_osc = None
        rt.sigma2_se = 0.01
        rt.sigma2_p = 0.5
        rt.sigma2_total_var = 1.0
        rt.sigma2_total_var_lower = 0.9
        rt.sigma2_total_var_upper = 1.1
        rt.ps = np.array([0.1, 0.2])
        rt.sigma2s = np.array([0.2, 0.3])
        rt.sigma2s_missing = None

        rt.phenos = ["P1"]
        rt.pheno_to_ind = {"P1": 0}
        rt.gene_pheno_Y = sparse.csc_matrix(([1.0], ([0], [0])), shape=(1, 1))
        rt.gene_pheno_combined_prior_Ys = None
        rt.gene_pheno_priors = None
        rt.X_phewas_beta = None
        rt.X_phewas_beta_uncorrected = None
        rt.num_gene_phewas_filtered = 3
        rt.anchor_gene_mask = np.array([True])
        rt.anchor_pheno_mask = np.array([True])

        y_state = pegs_utils.y_data_from_runtime(rt)
        hyper_state = pegs_utils.hyperparameter_data_from_runtime(rt)
        phewas_state = pegs_utils.phewas_runtime_state_from_runtime(rt)

        y_state.y_var = 2.5
        hyper_state.p = 0.25
        phewas_state.num_gene_phewas_filtered = 7

        pegs_utils.apply_y_data_to_runtime(rt, y_state)
        pegs_utils.apply_hyperparameter_data_to_runtime(rt, hyper_state)
        pegs_utils.apply_phewas_runtime_state_to_runtime(rt, phewas_state)

        self.assertAlmostEqual(rt.y_var, 2.5)
        self.assertAlmostEqual(rt.p, 0.25)
        self.assertEqual(rt.num_gene_phewas_filtered, 7)

    def test_set_runtime_hyper_helpers(self) -> None:
        class _Runtime:
            pass

        rt = _Runtime()
        rt.p = 0.2
        rt.sigma2 = None
        rt.sigma_power = 2.0
        rt.sigma2_osc = None
        rt.sigma2_se = None
        rt.sigma2_p = 0.1
        rt.sigma2_total_var = None
        rt.sigma2_total_var_lower = None
        rt.sigma2_total_var_upper = None
        rt.ps = None
        rt.sigma2s = None
        rt.sigma2s_missing = None
        rt.scale_factors = np.array([1.0, 2.0])
        rt.is_dense_gene_set = np.array([True, True])
        rt.MEAN_MOUSE_SCALE = 300.0

        state_p = pegs_utils.set_runtime_p(rt, 1.5)
        self.assertAlmostEqual(state_p.p, 1.0)
        self.assertAlmostEqual(rt.p, 1.0)

        state_sigma = pegs_utils.set_runtime_sigma(
            rt,
            sigma2=2.0,
            sigma_power=2.0,
            sigma2_osc=None,
            sigma2_se=0.1,
            sigma2_p=0.25,
            sigma2_scale_factors=np.array([1.0, 2.0]),
            convert_sigma_to_internal_units=False,
        )
        self.assertAlmostEqual(state_sigma.sigma2, 2.0)
        self.assertAlmostEqual(rt.sigma2, 2.0)
        self.assertAlmostEqual(rt.sigma2_p, 0.25)
        self.assertAlmostEqual(rt.sigma2_total_var, 10.0)
        self.assertAlmostEqual(rt.sigma2_total_var_lower, 9.02, places=2)
        self.assertAlmostEqual(rt.sigma2_total_var_upper, 10.98, places=2)

    def test_sync_runtime_state_helpers(self) -> None:
        class _Runtime:
            pass

        rt = _Runtime()
        rt.Y = np.array([1.0])
        rt.Y_for_regression = np.array([1.0])
        rt.Y_exomes = None
        rt.Y_positive_controls = None
        rt.Y_case_counts = None
        rt.y_var = 1.0
        rt.y_corr = None
        rt.y_corr_cholesky = None
        rt.y_corr_sparse = None
        rt.Y_w = None
        rt.Y_fw = None
        rt.y_w_var = None
        rt.y_w_mean = None
        rt.y_fw_var = None
        rt.y_fw_mean = None
        rt.p = 0.1
        rt.sigma2 = 0.2
        rt.sigma_power = 2.0
        rt.sigma2_osc = None
        rt.sigma2_se = None
        rt.sigma2_p = None
        rt.sigma2_total_var = None
        rt.sigma2_total_var_lower = None
        rt.sigma2_total_var_upper = None
        rt.ps = None
        rt.sigma2s = None
        rt.sigma2s_missing = None
        rt.phenos = ["P1"]
        rt.pheno_to_ind = {"P1": 0}
        rt.gene_pheno_Y = None
        rt.gene_pheno_combined_prior_Ys = None
        rt.gene_pheno_priors = None
        rt.X_phewas_beta = None
        rt.X_phewas_beta_uncorrected = None
        rt.num_gene_phewas_filtered = 0
        rt.anchor_gene_mask = None
        rt.anchor_pheno_mask = None

        y_state = pegs_utils.sync_y_state(rt)
        hyper_state = pegs_utils.sync_hyperparameter_state(rt)
        phewas_state = pegs_utils.sync_phewas_runtime_state(rt)

        self.assertTrue(np.array_equal(y_state.Y, rt.Y))
        self.assertAlmostEqual(hyper_state.p, rt.p)
        self.assertEqual(phewas_state.phenos, rt.phenos)

    def test_apply_parsed_gene_set_statistics_to_runtime(self) -> None:
        class _Runtime:
            def __init__(self) -> None:
                self.gene_sets = ["SET_A", "SET_B"]
                self.gene_set_to_ind = {"SET_A": 0, "SET_B": 1}
                self.scale_factors = np.array([2.0, 3.0])
                self.X_orig = np.zeros((2, 2))
                self.genes = ["G1", "G2"]
                self.subset_calls = []
                self.set_x_calls = []

            def subset_gene_sets(self, subset_mask, keep_missing=True):
                self.subset_calls.append((subset_mask.copy(), keep_missing))

            def _set_X(self, X_orig, genes, gene_sets, skip_N=True):
                self.set_x_calls.append((X_orig.shape, tuple(genes), tuple(gene_sets), skip_N))

        parsed = pegs_utils.ParsedGeneSetStats(
            need_to_take_log=False,
            has_beta_tilde=True,
            has_p_or_se=True,
            has_beta=True,
            has_beta_uncorrected=True,
            records={"SET_A": (1.0, 0.05, 2.0, 3.0, 4.0, 5.0)},
        )
        rt = _Runtime()
        pegs_utils.apply_parsed_gene_set_statistics_to_runtime(
            rt,
            parsed,
            return_only_ids=False,
            stats_beta_col="beta",
            warn_fn=lambda _m: None,
            bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
            log_fn=lambda _m: None,
        )

        self.assertAlmostEqual(rt.beta_tildes[0], 2.0)
        self.assertAlmostEqual(rt.p_values[0], 0.05)
        self.assertAlmostEqual(rt.ses[0], 4.0)
        self.assertAlmostEqual(rt.betas[0], 8.0)
        self.assertAlmostEqual(rt.betas_uncorrected[0], 10.0)
        self.assertEqual(len(rt.subset_calls), 1)
        np.testing.assert_array_equal(rt.subset_calls[0][0], np.array([True, False]))
        self.assertEqual(len(rt.set_x_calls), 1)

    def test_apply_parsed_gene_phewas_bfs_to_runtime(self) -> None:
        class _Runtime:
            def __init__(self) -> None:
                self.genes = ["GENE_A", "GENE_B"]
                self.phenos = None
                self.pheno_to_ind = None
                self.gene_pheno_Y = None
                self.gene_pheno_combined_prior_Ys = None
                self.gene_pheno_priors = None
                self.X_phewas_beta = None
                self.X_phewas_beta_uncorrected = None
                self.num_gene_phewas_filtered = 0
                self.anchor_gene_mask = None
                self.anchor_pheno_mask = None

        parsed = pegs_utils.ParsedGenePhewasBfs(
            phenos=["P1"],
            pheno_to_ind={"P1": 0},
            row=np.array([0], dtype=np.int32),
            col=np.array([0], dtype=np.int32),
            Ys=np.array([1.5]),
            combineds=np.array([0.5]),
            priors=np.array([0.2]),
            num_filtered=3,
        )
        rt = _Runtime()
        pegs_utils.apply_parsed_gene_phewas_bfs_to_runtime(
            rt,
            parsed,
            anchor_genes={"GENE_A"},
            anchor_phenos={"P1"},
            construct_map_to_ind_fn=pegs_utils.construct_map_to_ind,
            bail_fn=lambda m: (_ for _ in ()).throw(ValueError(m)),
            log_fn=lambda _m: None,
        )

        self.assertEqual(rt.num_gene_phewas_filtered, 3)
        self.assertEqual(rt.phenos, ["P1"])
        self.assertEqual(rt.pheno_to_ind, {"P1": 0})
        self.assertEqual(rt.gene_pheno_Y.shape, (2, 1))
        self.assertEqual(rt.gene_pheno_combined_prior_Ys.shape, (2, 1))
        self.assertEqual(rt.gene_pheno_priors.shape, (2, 1))
        np.testing.assert_array_equal(rt.anchor_gene_mask, np.array([True, False]))
        np.testing.assert_array_equal(rt.anchor_pheno_mask, np.array([True]))

    def test_remove_tag_from_input(self) -> None:
        path, tag = pegs_utils.remove_tag_from_input("mouse:data.tsv")
        self.assertEqual(path, "data.tsv")
        self.assertEqual(tag, "mouse")
        path, tag = pegs_utils.remove_tag_from_input("data.tsv")
        self.assertEqual(path, "data.tsv")
        self.assertIsNone(tag)

    def test_assign_default_batches(self) -> None:
        batches = [None, None, "X", None]
        orig_files = ["a", "a", "b", "c"]
        out = pegs_utils.assign_default_batches(
            batches=batches,
            orig_files=orig_files,
            batch_all_for_hyper=False,
            first_for_hyper=False,
        )
        self.assertEqual(out[0], out[1])
        self.assertEqual(out[2], "X")
        self.assertIsNotNone(out[3])

        out_first = pegs_utils.assign_default_batches(
            batches=[None, None, "Y"],
            orig_files=["a", "b", "c"],
            batch_all_for_hyper=False,
            first_for_hyper=True,
        )
        self.assertIsNotNone(out_first[0])
        self.assertIsNone(out_first[1])
        self.assertIsNone(out_first[2])

    def test_prepare_read_x_inputs_and_xdata_from_input_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sparse_list = root / "x_list.txt"
            dense_list = root / "xd_list.txt"
            (root / "rel_sparse.tsv").write_text("SET1 GENE1\n", encoding="utf-8")
            dense_member = root / "dense.tsv"
            dense_member.write_text("SET2\tGENE2\n", encoding="utf-8")

            sparse_list.write_text("tagS:rel_sparse.tsv\n", encoding="utf-8")
            dense_list.write_text(str(dense_member) + "\n", encoding="utf-8")

            plan = pegs_utils.prepare_read_x_inputs(
                X_in=["tagA:fileA.tsv@B0"],
                X_list=[str(sparse_list) + "@BL"],
                Xd_in=None,
                Xd_list=[str(dense_list)],
                initial_p=None,
                xin_to_p_noninf_ind=None,
                batch_separator="@",
                file_separator=None,
                sparse_list_open_fn=lambda p: open(p, "rt", encoding="utf-8"),
                dense_list_open_fn=lambda p: open(p, "rt", encoding="utf-8"),
            )

            self.assertEqual(len(plan.X_ins), 3)
            self.assertTrue(any("rel_sparse.tsv" in x for x in plan.X_ins))
            self.assertEqual(plan.batches[0], "B0")
            self.assertEqual(plan.batches[1], "BL")
            self.assertFalse(plan.is_dense[0])
            self.assertFalse(plan.is_dense[1])
            self.assertTrue(plan.is_dense[2])

            xdata = pegs_utils.xdata_from_input_plan(plan)
            self.assertEqual(xdata.gene_set_batches.shape[0], 3)
            self.assertEqual(xdata.gene_set_labels.shape[0], 3)
            self.assertEqual(xdata.is_dense_gene_set.dtype, np.bool_)


if __name__ == "__main__":
    unittest.main()
