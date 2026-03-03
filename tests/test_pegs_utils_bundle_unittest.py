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


if __name__ == "__main__":
    unittest.main()
