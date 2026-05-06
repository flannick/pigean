from __future__ import annotations

import json


FACTOR_WORKFLOW_STRATEGY_META = {
    "F1": {
        "required_inputs": [],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": False,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": None,
    },
    "F2": {
        "required_inputs": [],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": False,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": "gene_list",
    },
    "F3": {
        "required_inputs": [],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": False,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": None,
    },
    "F4": {
        "required_inputs": [],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": None,
    },
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _has_labeled_spec(value):
    return any("=" in str(item) for item in _as_list(value))


def _labeled_spec_labels(value):
    labels = []
    for item in _as_list(value):
        spec = str(item)
        if "=" in spec:
            label, path = spec.split("=", 1)
            if label and path:
                labels.append(label)
    return labels


def _single_complete_labeled_stats_pair(options):
    gene_labels = _labeled_spec_labels(options.gene_stats_in)
    gene_set_labels = _labeled_spec_labels(options.gene_set_stats_in)
    return bool(
        len(gene_labels) == 1
        and len(gene_set_labels) == 1
        and gene_labels[0] == gene_set_labels[0]
        and len(_as_list(options.gene_stats_in)) == 1
        and len(_as_list(options.gene_set_stats_in)) == 1
    )


def _labeled_stats_label_mismatch(options):
    gene_labels = set(_labeled_spec_labels(options.gene_stats_in))
    gene_set_labels = set(_labeled_spec_labels(options.gene_set_stats_in))
    if not gene_labels and not gene_set_labels:
        return None
    if gene_labels != gene_set_labels:
        return "Labeled --gene-stats-in and --gene-set-stats-in entries must use the same complete trait labels"
    return None


def has_multi_pheno_factor_inputs(options):
    has_phewas_pair = bool(
        options.gene_phewas_bfs_in is not None
        and options.gene_set_phewas_stats_in is not None
    )
    if _single_complete_labeled_stats_pair(options) and not has_phewas_pair:
        return False
    return bool(
        has_phewas_pair
        or len(_as_list(options.gene_stats_in)) > 1
        or len(_as_list(options.gene_set_stats_in)) > 1
        or _has_labeled_spec(options.gene_stats_in)
        or _has_labeled_spec(options.gene_set_stats_in)
    )


def workflow_required_inputs_satisfied(workflow_id, options):
    return []


def build_factor_workflow_error(workflow_id, missing_inputs):
    if len(missing_inputs) == 0:
        return None
    return "Missing required inputs: %s" % ", ".join(missing_inputs)


def has_potentially_ignored_factor_inputs(options):
    return bool(
        options.gene_set_stats_in
        or options.gene_stats_in
        or options.gene_list_in
        or options.gene_list is not None
        or options.positive_controls_in
        or options.positive_controls_list is not None
    )


def has_standalone_gene_list_inputs(options):
    return bool(
        options.gene_list_in is not None
        or options.gene_list is not None
        or options.positive_controls_in is not None
        or options.positive_controls_list is not None
    )


def get_selected_anchor_modes(options):
    selected = []
    if getattr(options, "anchor_phenos", None) is not None:
        selected.append(("anchor_phenos", options.anchor_phenos))
    if getattr(options, "anchor_any_pheno", False):
        selected.append(("anchor_any_pheno", True))
    if getattr(options, "anchor_genes", None) is not None:
        selected.append(("anchor_genes", options.anchor_genes))
    if getattr(options, "anchor_any_gene", False):
        selected.append(("anchor_any_gene", True))
    if getattr(options, "anchor_gene_set", False):
        selected.append(("anchor_gene_set", True))
    return selected


def get_routing_family(options, workflow, projection_only=False):
    if projection_only:
        return "projection_only_precomputed"
    workflow_id = workflow.get("id") if isinstance(workflow, dict) else None
    if workflow_id == "F4":
        return "phenotype_input_factoring"
    if workflow_id == "F2":
        return "standalone_gene_list"
    if workflow_id == "F3":
        return "default_stats_plus_projection"
    return "default_stats"


def format_anchor_values_for_label(values):
    if values is None:
        return "None"
    if isinstance(values, set):
        values = sorted(list(values))
    elif isinstance(values, (tuple, list)):
        values = list(values)
    else:
        return str(values)
    return "{%s}" % ", ".join(["'%s'" % x for x in values])


def classify_factor_workflow(options):
    has_gene_set_phewas = options.gene_set_phewas_stats_in is not None
    has_gene_phewas = options.gene_phewas_bfs_in is not None
    projection_source = options.gene_set_phewas_stats_in if has_gene_set_phewas else options.gene_phewas_bfs_in

    if has_multi_pheno_factor_inputs(options):
        workflow_id = "F4"
        workflow_label = "phenotype-input anchoring across all complete input traits"
    else:
        workflow_label = "single phenotype anchoring using default statistics"
        if has_standalone_gene_list_inputs(options):
            workflow_id = "F2"
            workflow_label = "standalone gene-list enrichment"
        elif projection_source is not None:
            workflow_id = "F3"
            workflow_label = "%s. Will project using %s" % (workflow_label, projection_source)
        else:
            workflow_id = "F1"

    strategy = FACTOR_WORKFLOW_STRATEGY_META[workflow_id]
    missing_inputs = workflow_required_inputs_satisfied(workflow_id, options)

    return {
        "id": workflow_id,
        "label": workflow_label,
        "error": build_factor_workflow_error(workflow_id, missing_inputs),
        "required_inputs": list(strategy["required_inputs"]),
        "missing_required_inputs": missing_inputs,
        "factor_gene_set_x_pheno": bool(strategy["factor_gene_set_x_pheno"]),
        "use_phewas_for_factoring": bool(strategy["use_phewas_for_factoring"]),
        "expand_gene_sets": bool(strategy["expand_gene_sets"]),
        "warn_ignored_y_inputs_mode": strategy["warn_ignored_y_inputs_mode"],
        "has_gene_set_phewas": has_gene_set_phewas,
        "has_gene_phewas": has_gene_phewas,
    }


def validate_factor_workflow_selection(options, workflow, projection_only, bail_fn):
    selected_anchor_modes = get_selected_anchor_modes(options)
    if len(selected_anchor_modes) > 0:
        bail_fn(
            "Explicit anchor workflow flags were removed. Provide phenotype inputs with "
            "--gene-stats-in LABEL=path/--gene-set-stats-in LABEL=path and/or "
            "--gene-phewas-stats-in/--gene-set-phewas-stats-in; all complete traits are used as anchors."
        )

    workflow_id = workflow.get("id") if isinstance(workflow, dict) else None
    labeled_stats_error = _labeled_stats_label_mismatch(options)
    if labeled_stats_error is not None:
        bail_fn(labeled_stats_error)
    standalone_gene_list = has_standalone_gene_list_inputs(options)
    has_x_source = any(
        x is not None
        for x in [options.X_in, options.X_list, options.Xd_in, options.Xd_list]
    )

    if projection_only:
        projection_needs_x = bool(
            (
                getattr(options, "factor_gene_clusters_in", None) is not None
                and (
                    getattr(options, "gene_set_clusters_out", None) is not None
                    or getattr(options, "gene_clusters_full_out", None) is not None
                )
            )
            or (
                getattr(options, "factor_gene_set_clusters_in", None) is not None
                and getattr(options, "gene_clusters_full_out", None) is not None
            )
        )
        if has_x_source and not projection_needs_x:
            bail_fn(
                "--factor-gene-clusters-in/--factor-gene-set-clusters-in are projection-only inputs and "
                "can only be combined with --X-in/--X-list/--Xd-in/--Xd-list when projecting across gene and gene-set bases."
            )
        if standalone_gene_list:
            bail_fn(
                "--factor-gene-clusters-in/--factor-gene-set-clusters-in cannot be combined with "
                "--gene-list/--gene-list-in/--positive-controls-* workflows."
            )
        if options.gene_set_stats_in is not None:
            if not projection_needs_x:
                bail_fn(
                    "--factor-gene-clusters-in/--factor-gene-set-clusters-in skip clustering and cannot be combined with "
                    "--gene-set-stats-in unless projecting across gene and gene-set bases with --X-in/--X-list/--Xd-in/--Xd-list."
                )
        return

    if workflow_id == "F4" and standalone_gene_list:
        bail_fn("Phenotype-input anchoring cannot be combined with standalone gene-list flags.")
    elif workflow_id == "F2":
        conflicting = []
        if options.gene_stats_in is not None:
            conflicting.append("--gene-stats-in")
        if options.gene_set_stats_in is not None:
            conflicting.append("--gene-set-stats-in")
        if conflicting:
            bail_fn(
                "Standalone gene-list enrichment cannot be combined with %s."
                % ", ".join(conflicting)
            )


def build_clustering_provenance(options, mode_state, outputs_written=None):
    workflow = mode_state.get("factor_workflow") if isinstance(mode_state, dict) else None
    routing_family = get_routing_family(
        options,
        workflow,
        projection_only=bool(mode_state.get("factor_projection_only")) if isinstance(mode_state, dict) else False,
    )
    if bool(workflow and workflow.get("use_phewas_for_factoring")):
        anchor_mode = "phenotype_inputs"
        anchor_values = []
    elif not bool(mode_state.get("factor_projection_only")) and (
        options.gene_stats_in is not None or options.gene_set_stats_in is not None
    ):
        anchor_mode = "default_stats"
        anchor_values = ["input_gene_stats"]
    else:
        anchor_mode = "none"
        anchor_values = []

    trait_linkage_enabled = bool(
        not getattr(options, "no_trait_linkage", False)
        or getattr(options, "trait_factor_links_out", None) is not None
        or getattr(options, "pheno_clusters_out", None) is not None
    )
    trait_linkage_basis = "gene_set" if getattr(options, "project_phenos_from_gene_sets", False) else "gene"
    factorization_source = "precomputed_factor_clusters" if mode_state.get("factor_projection_only") else (
        "phenotype_inputs" if bool(workflow and workflow.get("use_phewas_for_factoring")) else "default_stats"
    )

    return {
        "workflow_id": workflow.get("id") if isinstance(workflow, dict) else None,
        "workflow_label": workflow.get("label") if isinstance(workflow, dict) else None,
        "routing_family": routing_family,
        "factorization_source": factorization_source,
        "clustering_executed": bool(mode_state.get("run_factor") and not mode_state.get("factor_projection_only")),
        "projection_only": bool(mode_state.get("factor_projection_only")),
        "precomputed_factors_loaded": bool(mode_state.get("factor_projection_only")),
        "anchor_mode": anchor_mode,
        "anchor_values": anchor_values,
        "anchor_count": len(anchor_values),
        "inputs": {
            "X_in": options.X_in,
            "X_list": options.X_list,
            "Xd_in": options.Xd_in,
            "Xd_list": options.Xd_list,
            "gene_stats_in": options.gene_stats_in,
            "gene_set_stats_in": options.gene_set_stats_in,
            "gene_phewas_stats_in": options.gene_phewas_bfs_in,
            "gene_set_phewas_stats_in": options.gene_set_phewas_stats_in,
            "factor_gene_clusters_in": options.factor_gene_clusters_in,
            "factor_gene_set_clusters_in": options.factor_gene_set_clusters_in,
        },
        "trait_linkage": {
            "enabled": trait_linkage_enabled,
            "basis": trait_linkage_basis,
            "source": getattr(options, "trait_linkage_source", "combined"),
            "threshold": getattr(options, "trait_linkage_threshold", 1.0),
            "computation_mode": getattr(options, "trait_linkage_computation_mode", "sparse_full"),
            "capture_input": getattr(options, "pheno_capture_input", None),
        },
        "factor_phewas": {
            "enabled": bool(getattr(options, "run_factor_phewas", False)),
            "mode": getattr(options, "factor_phewas_mode", None),
            "modes": getattr(options, "factor_phewas_modes", None),
        },
        "filters": {
            "min_gene_set_size": getattr(options, "min_gene_set_size", None),
            "max_gene_set_size": getattr(options, "max_gene_set_size", None),
            "prune_gene_sets": getattr(options, "prune_gene_sets", None),
            "weighted_prune_gene_sets": getattr(options, "weighted_prune_gene_sets", None),
            "threshold_weights": getattr(options, "threshold_weights", None),
            "cap_weights": getattr(options, "cap_weights", None),
            "max_num_gene_sets_initial": getattr(options, "max_num_gene_sets_initial", None),
            "max_num_gene_sets": getattr(options, "max_num_gene_sets", None),
            "max_num_gene_sets_hyper": getattr(options, "max_num_gene_sets_hyper", None),
            "max_gene_set_read_p": getattr(options, "max_gene_set_read_p", None),
            "min_gene_set_read_beta": getattr(options, "min_gene_set_read_beta", None),
            "min_gene_set_read_beta_uncorrected": getattr(options, "min_gene_set_read_beta_uncorrected", None),
        },
        "outputs_requested": {
            "factors_out": getattr(options, "factors_out", None),
            "factor_metrics_out": getattr(options, "factor_metrics_out", None),
            "gene_set_clusters_out": getattr(options, "gene_set_clusters_out", None),
            "gene_clusters_out": getattr(options, "gene_clusters_out", None),
            "gene_clusters_full_out": getattr(options, "gene_clusters_full_out", None),
            "trait_factor_links_out": getattr(options, "trait_factor_links_out", None),
            "factor_phewas_stats_out": getattr(options, "factor_phewas_stats_out", None),
            "params_out": getattr(options, "params_out", None),
            "clustering_params_out": getattr(options, "clustering_params_out", None),
        },
        "outputs_written": outputs_written or {},
    }


def format_clustering_provenance_summary(provenance):
    trait_linkage = provenance.get("trait_linkage", {})
    outputs_requested = provenance.get("outputs_requested", {})
    output_targets = [key for key, value in outputs_requested.items() if value]
    summary_lines = [
        "EAGGL workflow=%s routing=%s clustering=%s source=%s"
        % (
            provenance.get("workflow_id"),
            provenance.get("routing_family"),
            "run" if provenance.get("clustering_executed") else "skipped",
            provenance.get("factorization_source"),
        ),
        "EAGGL anchors=%s%s trait_linkage=%s(%s,%s,%s)"
        % (
            provenance.get("anchor_mode"),
            (" %s" % json.dumps(provenance.get("anchor_values"))) if provenance.get("anchor_values") else "",
            "on" if trait_linkage.get("enabled") else "off",
            trait_linkage.get("basis"),
            trait_linkage.get("source"),
            trait_linkage.get("computation_mode"),
        ),
        "EAGGL outputs=%s"
        % (", ".join(output_targets) if len(output_targets) > 0 else "none"),
    ]
    return summary_lines


def warn_for_factor_workflow_inputs(options, workflow, warn_fn):
    add_gene_set_flags_present = (
        options.add_gene_sets_by_enrichment_p is not None
        or options.add_gene_sets_by_fraction is not None
    )
    if add_gene_set_flags_present:
        warn_fn("Ignoring options to add gene sets based on gene anchoring; explicit gene anchoring was removed")

    warning_mode = workflow.get("warn_ignored_y_inputs_mode")
    if warning_mode == "gene_list" and (options.gene_stats_in is not None or options.gene_set_stats_in is not None):
        warn_fn("Ignoring precomputed gene/gene-set stats in standalone gene-list mode")


def enforce_factor_only_input_boundary(options, mode_state, bail_fn):
    if not mode_state.get("run_factor"):
        return

    if mode_state.get("factor_projection_only"):
        has_x_source = any(
            x is not None
            for x in [options.X_in, options.X_list, options.Xd_in, options.Xd_list]
        )
        if has_x_source:
            bail_fn(
                "--factor-gene-clusters-in/--factor-gene-set-clusters-in skip clustering and cannot be combined with "
                "--X-in/--X-list/--Xd-in/--Xd-list."
            )
        return

    has_x_source = any(
        x is not None
        for x in [options.X_in, options.X_list, options.Xd_in, options.Xd_list]
    )
    if not has_x_source:
        bail_fn(
            "EAGGL requires an X matrix input. Provide --X-in/--X-list/--Xd-in/--Xd-list "
            "(or use --eaggl-bundle-in with an X default)."
        )

    workflow = mode_state.get("factor_workflow")
    use_phewas_for_factoring = bool(workflow and workflow.get("use_phewas_for_factoring"))
    workflow_id = workflow.get("id") if isinstance(workflow, dict) else None
    if workflow_id == "F2":
        return
    if not use_phewas_for_factoring:
        missing = []
        if options.gene_stats_in is None:
            missing.append("--gene-stats-in")
        if options.gene_set_stats_in is None:
            missing.append("--gene-set-stats-in")
        if len(missing) > 0:
            bail_fn(
                "EAGGL factor workflows require precomputed PIGEAN stats: missing %s "
                "(or provide them in --eaggl-bundle-in)." % ", ".join(missing)
            )
