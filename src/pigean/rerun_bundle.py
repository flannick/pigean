from __future__ import annotations

import gzip
import os
import tarfile
import tempfile
from typing import Any

import numpy as np

from pegs_shared import bundle as pegs_bundle
from pegs_shared.cli import json_safe
from pegs_shared.io_common import open_text_auto, resolve_column_index


PIGEAN_RERUN_BUNDLE_SCHEMA = "pigean_rerun_bundle/v1"
PIGEAN_RERUN_BUNDLE_ALLOWED_DEFAULT_INPUTS = {
    "X_in",
    "gene_stats_in",
    "gene_universe_in",
}

# These option destinations affect the beta-tilde or joint beta stage and should
# be replayed by a fixed-Y rerun unless the user explicitly overrides them.
RERUN_DEFAULT_OPTION_NAMES = [
    "p_noninf",
    "sigma2",
    "sigma_power",
    "sigma_soft_threshold_95",
    "sigma_soft_threshold_5",
    "background_prior",
    "update_hyper",
    "max_allowed_batch_correlation",
    "sparse_solution",
    "sparse_frac_betas",
    "pre_filter_batch_size",
    "pre_filter_small_batch_size",
    "max_num_gene_sets",
    "max_num_gene_sets_initial",
    "max_num_gene_sets_hyper",
    "min_gene_set_size",
    "max_gene_set_size",
    "filter_gene_set_p",
    "filter_gene_set_metric_z",
    "max_gene_set_read_p",
    "min_gene_set_read_beta",
    "min_gene_set_read_beta_uncorrected",
    "filter_negative",
    "ols",
    "linear",
    "max_for_linear",
    "use_sampling_for_betas",
    "correct_betas_mean",
    "correct_betas_var",
    "retain_all_beta_uncorrected",
    "independent_betas_only",
    "track_filtered_beta_uncorrected_mode",
    "max_num_burn_in_betas",
    "max_num_iter_betas",
    "min_num_iter_betas",
    "num_chains_betas",
    "r_threshold_burn_in_betas",
    "use_max_r_for_convergence_betas",
    "max_frac_sem_betas",
    "gauss_seidel_betas",
    "seed",
    "deterministic",
    "max_gb",
]


def _open_gz_text(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _as_jsonable(value: Any) -> Any:
    return json_safe(value)


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    return [float(x) for x in arr]


def _parse_params_scalar(value: str) -> Any:
    text = str(value).strip()
    if text == "None":
        return None
    if text == "True":
        return True
    if text == "False":
        return False
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except Exception:
            return text
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except Exception:
        return text


def read_params_out(path: str, *, bail_fn) -> dict[str, list[Any]]:
    params: dict[str, list[Any]] = {}
    with open_text_auto(path, "rt") as in_fh:
        header = in_fh.readline().rstrip("\n").split("\t")
        if header[:3] != ["Parameter", "Version", "Value"]:
            bail_fn(f"--pigean-params-in expected header 'Parameter\\tVersion\\tValue' in {path}")
        for line in in_fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            param, _version, value = fields[0], fields[1], "\t".join(fields[2:])
            params.setdefault(param, []).append(_parse_params_scalar(value))
    return params


def _collapse_param_values(values: list[Any]) -> Any:
    if len(values) == 0:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _extract_label_param_map(params: dict[str, list[Any]], prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, values in params.items():
        if not key.startswith(prefix):
            continue
        label = key[len(prefix) :]
        if not values:
            continue
        result[label] = float(values[-1])
    return result


def _input_label_from_spec(spec: str) -> str:
    text = str(spec)
    if "@" in text:
        text = text.rsplit("@", 1)[0]
    if ":" in text:
        text = text.split(":", 1)[1]
    label = os.path.basename(text)
    if "." in label:
        label = ".".join(label.split(".")[:-1])
    return label


def _current_x_labels(options) -> list[str]:
    labels: list[str] = []
    for specs in (getattr(options, "X_in", None), getattr(options, "Xd_in", None)):
        if specs is None:
            continue
        if isinstance(specs, str):
            specs = [specs]
        for spec in specs:
            labels.append(_input_label_from_spec(spec))
    return labels


def record_label_hyperparameters(state) -> None:
    labels = getattr(state, "gene_set_labels", None)
    if labels is None or len(labels) == 0:
        return
    label_order: list[str] = []
    seen = set()
    for label in labels:
        label_text = str(label)
        if label_text not in seen:
            seen.add(label_text)
            label_order.append(label_text)
    state._record_param("hyperparameter_label_order", "|".join(label_order), overwrite=True)
    ps = getattr(state, "ps", None)
    sigma2s = getattr(state, "sigma2s", None)
    for label in label_order:
        mask = np.asarray(labels, dtype=object) == label
        if ps is not None:
            vals = np.asarray(ps, dtype=float)[mask]
            if vals.size > 0:
                state._record_param(f"p_by_label__{label}", float(np.median(vals)), overwrite=True)
        if sigma2s is not None:
            vals = np.asarray(sigma2s, dtype=float)[mask]
            if vals.size > 0:
                state._record_param(f"sigma2_by_label__{label}", float(np.median(vals)), overwrite=True)


def _write_gene_universe(path: str, state) -> None:
    genes = getattr(state, "genes", None)
    if genes is None:
        return
    with _open_gz_text(path, "w") as out_fh:
        out_fh.write("Gene\n")
        for gene in genes:
            out_fh.write(f"{gene}\n")


def _write_params_snapshot(path: str, state) -> None:
    with _open_gz_text(path, "w") as out_fh:
        out_fh.write("Parameter\tVersion\tValue\n")
        for param in getattr(state, "param_keys", []) or []:
            value = state.params.get(param)
            values = value if isinstance(value, list) else [value]
            for idx, item in enumerate(values, start=1):
                out_fh.write(f"{param}\t{idx}\t{item}\n")


def _require_header_column(path: str, column: str, *, option_name: str, bail_fn) -> None:
    with _open_gz_text(path, "r") as in_fh:
        header = in_fh.readline().rstrip("\n").split("\t")
    if column not in header:
        bail_fn(f"{option_name} could not write required column '{column}' in {path}")


def _stage_generated_file(stage_dir: str, bundle_name: str, write_fn, label: str, suggestion: str, *, option_name: str, bail_fn) -> str:
    path = os.path.join(stage_dir, bundle_name)
    write_fn(path)
    pegs_bundle.require_existing_nonempty_file(
        path,
        label,
        suggestion,
        option_name=option_name,
        bail_fn=bail_fn,
    )
    return path


def _collect_rerun_defaults(state, options) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for dest in RERUN_DEFAULT_OPTION_NAMES:
        if hasattr(options, dest):
            value = getattr(options, dest)
            if value is not None:
                defaults[dest] = _as_jsonable(value)

    p_value = _as_float_list(getattr(state, "p", None))
    if p_value is not None:
        defaults["p_noninf"] = p_value
    sigma2 = getattr(state, "sigma2", None)
    if sigma2 is not None:
        defaults["sigma2"] = _as_jsonable(float(sigma2))
    sigma_power = getattr(state, "sigma_power", None)
    if sigma_power is not None:
        defaults["sigma_power"] = _as_jsonable(float(sigma_power))
    # Reruns use the original run's learned/internal hyperparameters as fixed
    # inputs; they should not perform outer hyperparameter learning again.
    defaults["update_hyper"] = "none"
    return defaults


def write_pigean_rerun_bundle(services, state, options, mode: str, out_path: str) -> None:
    option_name = "--pigean-rerun-bundle-out"
    tar_mode = pegs_bundle.get_tar_write_mode_for_bundle_path(
        out_path,
        option_name=option_name,
        bail_fn=services.bail,
    )
    pegs_bundle.ensure_parent_dir_for_file(out_path)
    services.log(f"Writing PIGEAN rerun bundle to {out_path}", services.INFO)

    with tempfile.TemporaryDirectory(prefix="pigean_rerun_bundle_") as stage_dir:
        file_meta: dict[str, dict[str, Any]] = {}
        default_inputs = {
            "X_in": "X.tsv.gz",
            "gene_stats_in": "gene_stats.tsv.gz",
            "gene_universe_in": "gene_universe.tsv.gz",
        }

        required_specs = [
            (
                "X.tsv.gz",
                lambda path: state.write_X(path),
                "X matrix",
                "run with --X-in/--X-list and ensure gene sets were loaded",
            ),
            (
                "gene_stats.tsv.gz",
                lambda path: state.write_gene_statistics(path, output_detail="full"),
                "gene statistics",
                "run a mode that computes or loads gene scores with a combined column",
            ),
            (
                "gene_universe.tsv.gz",
                lambda path: _write_gene_universe(path, state),
                "active gene universe",
                "run with an initialized gene universe",
            ),
            (
                "params.tsv.gz",
                lambda path: _write_params_snapshot(path, state),
                "resolved params",
                "run after PIGEAN has recorded resolved params",
            ),
        ]
        for bundle_name, write_fn, label, suggestion in required_specs:
            staged_path = _stage_generated_file(
                stage_dir,
                bundle_name,
                write_fn,
                label,
                suggestion,
                option_name=option_name,
                bail_fn=services.bail,
            )
            file_meta[bundle_name] = pegs_bundle.collect_file_metadata(staged_path)

        _require_header_column(os.path.join(stage_dir, "gene_stats.tsv.gz"), "Gene", option_name=option_name, bail_fn=services.bail)
        _require_header_column(os.path.join(stage_dir, "gene_stats.tsv.gz"), "combined", option_name=option_name, bail_fn=services.bail)
        _require_header_column(os.path.join(stage_dir, "gene_universe.tsv.gz"), "Gene", option_name=option_name, bail_fn=services.bail)

        # Reference-only copy. Rerun bundle input intentionally never maps this to
        # --gene-set-stats-in, so beta-tildes and joint betas are recomputed.
        try:
            ref_path = os.path.join(stage_dir, "gene_set_stats.tsv.gz")
            state.write_gene_set_statistics(
                ref_path,
                max_no_write_gene_set_beta=options.max_no_write_gene_set_beta,
                max_no_write_gene_set_beta_uncorrected=options.max_no_write_gene_set_beta_uncorrected,
                output_detail="full",
            )
            if os.path.exists(ref_path) and os.path.getsize(ref_path) > 0:
                file_meta["gene_set_stats.tsv.gz"] = pegs_bundle.collect_file_metadata(ref_path)
        except Exception as exc:  # pragma: no cover - optional provenance only
            services.warn(f"Could not include reference gene_set_stats.tsv.gz in {option_name}: {exc}")

        manifest = pegs_bundle.BundleManifest.build(
            schema=PIGEAN_RERUN_BUNDLE_SCHEMA,
            source_tool="pigean.py",
            source_mode=mode,
            source_argv=services.sys.argv,
            default_inputs=default_inputs,
            files_metadata=file_meta,
        )
        manifest.manifest["column_mapping"] = {
            "gene_stats_id_col": "Gene",
            "gene_stats_combined_col": "combined",
            "gene_stats_prior_col": "prior",
            "gene_stats_log_bf_col": "log_bf",
            "gene_universe_id_col": "Gene",
            "gene_universe_has_header": True,
        }
        manifest.manifest["rerun_defaults"] = _collect_rerun_defaults(state, options)
        manifest.manifest["beta_stage_options"] = {
            key: manifest.manifest["rerun_defaults"][key]
            for key in RERUN_DEFAULT_OPTION_NAMES
            if key in manifest.manifest["rerun_defaults"]
        }
        manifest.write_manifest(stage_dir, manifest_name="manifest.json")
        with tarfile.open(out_path, tar_mode) as tar_fh:
            tar_fh.add(os.path.join(stage_dir, "manifest.json"), arcname="manifest.json")
            for bundle_name in sorted(file_meta.keys()):
                tar_fh.add(os.path.join(stage_dir, bundle_name), arcname=bundle_name)

    services.log(f"Finished writing PIGEAN rerun bundle {out_path}", services.INFO)


def _set_if_not_explicit(options, dest: str, value: Any, explicit_dests: set[str], applied: dict[str, Any]) -> None:
    if value is None:
        return
    if dest in explicit_dests:
        return
    setattr(options, dest, value)
    applied[dest] = value


def apply_pigean_rerun_bundle_defaults(options, mode: str, cli_dests: set[str], config_dests: set[str], *, bail_fn, warn_fn):
    bundle_path = getattr(options, "pigean_rerun_bundle_in", None)
    if bundle_path is None:
        return None
    if mode != "betas":
        bail_fn("Option --pigean-rerun-bundle-in requires mode 'betas'; it must not run outer Gibbs")

    bundle = pegs_bundle.BundleManifest.load_defaults(
        bundle_path=bundle_path,
        expected_schema=PIGEAN_RERUN_BUNDLE_SCHEMA,
        allowed_default_inputs=PIGEAN_RERUN_BUNDLE_ALLOWED_DEFAULT_INPUTS,
        bundle_flag_name="--pigean-rerun-bundle-in",
        temp_prefix="pigean_rerun_bundle_in_",
        bail_fn=bail_fn,
    )
    explicit = set(cli_dests or set()) | set(config_dests or set())
    applied: dict[str, Any] = {}

    has_explicit_x_source = any(getattr(options, key, None) is not None for key in ("X_in", "X_list", "Xd_in", "Xd_list"))
    if "X_in" in bundle.default_inputs and not has_explicit_x_source:
        options.X_in = [bundle.default_inputs["X_in"]]
        applied["X_in"] = options.X_in
    _set_if_not_explicit(options, "gene_stats_in", bundle.default_inputs.get("gene_stats_in"), explicit, applied)
    _set_if_not_explicit(options, "gene_universe_in", bundle.default_inputs.get("gene_universe_in"), explicit, applied)

    mapping = bundle.manifest.get("column_mapping", {})
    for dest in (
        "gene_stats_id_col",
        "gene_stats_combined_col",
        "gene_stats_prior_col",
        "gene_stats_log_bf_col",
        "gene_universe_id_col",
        "gene_universe_has_header",
    ):
        if dest in mapping:
            _set_if_not_explicit(options, dest, mapping[dest], explicit, applied)

    rerun_defaults = bundle.manifest.get("rerun_defaults", {})
    if not isinstance(rerun_defaults, dict):
        bail_fn("--pigean-rerun-bundle-in manifest key 'rerun_defaults' must be an object")
    for dest, value in rerun_defaults.items():
        if dest == "gene_set_stats_in":
            continue
        if dest == "p_noninf_by_gene_set":
            setattr(options, "pigean_replay_ps", value)
            applied["pigean_replay_ps"] = value
            continue
        if dest == "sigma2_by_gene_set":
            setattr(options, "pigean_replay_sigma2s", value)
            applied["pigean_replay_sigma2s"] = value
            continue
        if not hasattr(options, dest):
            continue
        _set_if_not_explicit(options, dest, value, explicit, applied)

    options.pigean_rerun_bundle_applied_defaults = applied
    options.pigean_rerun_bundle_manifest = {
        "schema": bundle.manifest.get("schema"),
        "source": bundle.manifest.get("source"),
        "default_inputs": bundle.manifest.get("default_inputs", {}),
        "applied_defaults": applied,
    }
    if not applied:
        warn_fn(f"Loaded --pigean-rerun-bundle-in {bundle_path}; no defaults applied because explicit inputs/options took precedence")
    return bundle


def apply_pigean_params_defaults(options, mode: str, cli_dests: set[str], config_dests: set[str], *, bail_fn, warn_fn):
    params_path = getattr(options, "pigean_params_in", None)
    if params_path is None:
        return None
    if mode != "betas":
        bail_fn("Option --pigean-params-in requires mode 'betas'; it must not run outer Gibbs")
    params = read_params_out(params_path, bail_fn=bail_fn)
    explicit = set(cli_dests or set()) | set(config_dests or set())
    applied: dict[str, Any] = {}

    # Replay learned hyperparameters, not just the originally requested option
    # defaults. Vector p/sigma2 values are later mapped onto loaded gene sets by
    # annotation-library label.
    if "p" in params and "p_noninf" not in explicit:
        value = [float(x) for x in params["p"]]
        options.p_noninf = value
        applied["p_noninf"] = value
        setattr(options, "pigean_replay_ps", value)
        applied["pigean_replay_ps"] = value
    label_p = _extract_label_param_map(params, "p_by_label__")
    if label_p:
        setattr(options, "pigean_replay_ps_by_label", label_p)
        applied["pigean_replay_ps_by_label"] = label_p
    if "sigma2" in params and "sigma2" not in explicit:
        sigma2_values = [float(x) for x in params["sigma2"]]
        setattr(options, "pigean_replay_sigma2s", sigma2_values)
        applied["pigean_replay_sigma2s"] = sigma2_values
        if len(sigma2_values) > 0:
            options.sigma2 = float(np.mean(sigma2_values))
            applied["sigma2"] = options.sigma2
    label_sigma2 = _extract_label_param_map(params, "sigma2_by_label__")
    if label_sigma2:
        setattr(options, "pigean_replay_sigma2s_by_label", label_sigma2)
        applied["pigean_replay_sigma2s_by_label"] = label_sigma2
    if "sigma_power" in params and "sigma_power" not in explicit:
        value = float(params["sigma_power"][-1])
        options.sigma_power = value
        applied["sigma_power"] = value

    for dest in RERUN_DEFAULT_OPTION_NAMES:
        if dest in ("p_noninf", "sigma2", "sigma_power", "update_hyper"):
            continue
        param_name = f"option_{dest}"
        if param_name not in params:
            continue
        _set_if_not_explicit(options, dest, _collapse_param_values(params[param_name]), explicit, applied)

    # A params replay is a fixed-parameter beta-stage rerun. The original
    # params file may say option_update_hyper=p from the Gibbs run, but replaying
    # that would intentionally re-learn the parameters instead of reusing them.
    if "update_hyper" not in explicit:
        options.update_hyper = "none"
        applied["update_hyper"] = "none"

    setattr(options, "pigean_params_applied_defaults", applied)
    setattr(options, "pigean_params_source", params_path)
    if not applied:
        warn_fn(f"Loaded --pigean-params-in {params_path}; no defaults applied because explicit options took precedence")
    return params


def _map_replay_values_to_gene_sets(values: Any, labels: np.ndarray, x_labels: list[str], name: str, *, bail_fn, values_by_label: dict[str, float] | None = None) -> np.ndarray | None:
    if values_by_label:
        missing = sorted({str(label) for label in labels if str(label) not in values_by_label})
        if missing:
            bail_fn(
                f"Replayed {name} label map is missing loaded gene-set labels; first missing labels: "
                + ", ".join(missing[:5])
            )
        return np.asarray([float(values_by_label[str(label)]) for label in labels], dtype=float)
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) == 0:
        return None
    if len(arr) == len(labels):
        return arr.copy()
    if len(arr) == 1:
        return np.full(len(labels), float(arr[0]), dtype=float)
    if len(arr) != len(x_labels):
        bail_fn(
            f"Replayed {name} has {len(arr)} values, but current inputs have {len(x_labels)} direct X/Xd labels "
            f"and {len(labels)} retained gene sets"
        )
    label_to_value = dict(zip(x_labels, arr))
    missing = sorted({str(label) for label in labels if str(label) not in label_to_value})
    if missing:
        bail_fn(
            f"Replayed {name} could not be mapped to loaded gene-set labels; first missing labels: "
            + ", ".join(missing[:5])
        )
    return np.asarray([label_to_value[str(label)] for label in labels], dtype=float)


def apply_replayed_params_to_loaded_gene_sets(state, options=None, *, replay_ps=None, replay_sigma2s=None, replay_x_labels=None, replay_ps_by_label=None, replay_sigma2s_by_label=None, bail_fn, log_fn=None) -> None:
    labels = getattr(state, "gene_set_labels", None)
    if labels is None or len(labels) == 0:
        return
    if options is not None:
        if replay_ps is None:
            replay_ps = getattr(options, "pigean_replay_ps", None)
        if replay_sigma2s is None:
            replay_sigma2s = getattr(options, "pigean_replay_sigma2s", None)
        if replay_x_labels is None:
            replay_x_labels = _current_x_labels(options)
        if replay_ps_by_label is None:
            replay_ps_by_label = getattr(options, "pigean_replay_ps_by_label", None)
        if replay_sigma2s_by_label is None:
            replay_sigma2s_by_label = getattr(options, "pigean_replay_sigma2s_by_label", None)
    if replay_x_labels is None:
        replay_x_labels = []
    ps = _map_replay_values_to_gene_sets(
        replay_ps,
        np.asarray(labels),
        list(replay_x_labels),
        "p values",
        bail_fn=bail_fn,
        values_by_label=replay_ps_by_label,
    )
    sigma2s = _map_replay_values_to_gene_sets(
        replay_sigma2s,
        np.asarray(labels),
        list(replay_x_labels),
        "sigma2 values",
        bail_fn=bail_fn,
        values_by_label=replay_sigma2s_by_label,
    )
    if ps is None and sigma2s is None:
        return
    if ps is not None:
        state.ps = ps
        state.set_p(float(np.mean(ps)))
        state._record_param("pigean_params_replay_p_values_applied", len(ps), overwrite=True)
    if sigma2s is not None:
        state.sigma2s = sigma2s
        state.set_sigma(float(np.mean(sigma2s)), state.sigma_power)
        state._record_param("pigean_params_replay_sigma2_values_applied", len(sigma2s), overwrite=True)
    if log_fn is not None:
        log_fn("Applied replayed PIGEAN p/sigma2 values to loaded gene sets")


def read_gene_set_exclude_ids(path: str, *, id_col: str | None = None, has_header: bool = False, bail_fn=None) -> list[str]:
    if bail_fn is None:
        bail_fn = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    ids: list[str] = []
    with open_text_auto(path, "rt") as in_fh:
        header = None
        col_idx = 0
        if has_header:
            header = in_fh.readline().rstrip("\n").split("\t")
            if id_col is not None:
                col_idx = resolve_column_index(id_col, header, bail_fn=bail_fn)
        elif id_col is not None:
            try:
                col_idx = int(id_col) - 1
            except Exception:
                bail_fn("Option --gene-set-exclude-id-col must be a 1-based integer when --gene-set-exclude-no-header is used")
            if col_idx < 0:
                bail_fn("Option --gene-set-exclude-id-col must be >= 1")
        for raw in in_fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if col_idx >= len(fields):
                continue
            value = fields[col_idx].strip()
            if value:
                ids.append(value)
    return ids


def apply_gene_set_exclusions(
    state,
    requested: list[str],
    *,
    source_path: str | None,
    warn_fn,
    log_fn,
    info_level: int = 1,
) -> None:
    requested_unique = list(dict.fromkeys(requested))
    requested_set = set(requested_unique)
    current = list(getattr(state, "gene_sets", []) or [])
    found_set = requested_set.intersection(current)
    not_found = [item for item in requested_unique if item not in found_set]

    if requested_unique and not found_set:
        preview = ", ".join(not_found[:10])
        warn_fn(
            f"None of {len(requested_unique)} requested --gene-set-exclude-in IDs were found in loaded annotations; first missing IDs: {preview}"
        )
    if found_set:
        mask = np.array([gene_set not in found_set for gene_set in current], dtype=bool)
        state.subset_gene_sets(mask, keep_missing=False, ignore_missing=False, skip_V=True, filter_reason="gene_set_exclude_in")

    params = {
        "gene_set_exclude_in": source_path,
        "gene_set_exclude_requested_count": len(requested_unique),
        "gene_set_exclude_found_count": len(found_set),
        "gene_set_exclude_not_found_count": len(not_found),
        "gene_set_exclude_first_not_found_ids": ",".join(not_found[:10]),
        "gene_set_exclude_retained_gene_set_count": len(getattr(state, "gene_sets", []) or []),
    }
    state._record_params(params, overwrite=True)
    log_fn(
        "Applied gene-set exclusion: requested=%d found=%d not_found=%d retained=%d"
        % (len(requested_unique), len(found_set), len(not_found), len(getattr(state, "gene_sets", []) or [])),
        info_level,
    )


def apply_gene_set_exclusions_if_requested(services, state, options) -> None:
    path = getattr(options, "gene_set_exclude_in", None)
    if path is None:
        return
    requested = read_gene_set_exclude_ids(
        path,
        id_col=getattr(options, "gene_set_exclude_id_col", None),
        has_header=bool(getattr(options, "gene_set_exclude_has_header", False)),
        bail_fn=services.bail,
    )
    apply_gene_set_exclusions(
        state,
        requested,
        source_path=path,
        warn_fn=services.warn,
        log_fn=services.log,
        info_level=services.INFO,
    )
