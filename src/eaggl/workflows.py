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
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": "anchor_phenos",
    },
    "F5": {
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": False,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": "anchor_phenos",
    },
    "F6": {
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": True,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": "anchor_genes",
    },
    "F7": {
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": True,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": True,
        "warn_ignored_y_inputs_mode": "anchor_genes",
    },
    "F8": {
        "required_inputs": ["--gene-set-phewas-stats-in", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": True,
        "use_phewas_for_factoring": True,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": "anchor_genes",
    },
    "F9": {
        "required_inputs": ["--run-phewas", "--gene-phewas-stats-in"],
        "factor_gene_set_x_pheno": True,
        "use_phewas_for_factoring": False,
        "expand_gene_sets": False,
        "warn_ignored_y_inputs_mode": None,
    },
}


def workflow_required_inputs_satisfied(workflow_id, options):
    required_inputs = FACTOR_WORKFLOW_STRATEGY_META[workflow_id]["required_inputs"]
    missing_inputs = []
    for flag in required_inputs:
        if flag == "--gene-set-phewas-stats-in":
            if options.gene_set_phewas_stats_in is None:
                missing_inputs.append(flag)
        elif flag == "--gene-phewas-stats-in":
            if options.gene_phewas_bfs_in is None:
                missing_inputs.append(flag)
        elif flag == "--run-phewas":
            if not options.run_phewas:
                missing_inputs.append(flag)
        elif flag == "--gene-phewas-stats-in":
            if options.run_phewas_input is None:
                missing_inputs.append(flag)
    return missing_inputs


def build_factor_workflow_error(workflow_id, missing_inputs):
    if len(missing_inputs) == 0:
        return None
    if workflow_id in ("F4", "F5", "F6", "F7", "F8"):
        return "Require --gene-set-phewas-stats-in and --gene-phewas-stats-in"
    if workflow_id == "F9":
        return "Require --run-phewas and --gene-phewas-stats-in"
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
    if options.anchor_phenos is not None:
        selected.append(("anchor_phenos", options.anchor_phenos))
    if options.anchor_any_pheno:
        selected.append(("anchor_any_pheno", True))
    if options.anchor_genes is not None:
        selected.append(("anchor_genes", options.anchor_genes))
    if options.anchor_any_gene:
        selected.append(("anchor_any_gene", True))
    if options.anchor_gene_set:
        selected.append(("anchor_gene_set", True))
    return selected


def get_routing_family(options, workflow, projection_only=False):
    if projection_only:
        return "projection_only_precomputed"
    workflow_id = workflow.get("id") if isinstance(workflow, dict) else None
    if workflow_id in set(["F4", "F5"]):
        return "anchor_phenos_phewas_factoring"
    if workflow_id in set(["F6", "F7", "F8"]):
        return "anchor_genes_phewas_factoring"
    if workflow_id == "F9":
        return "anchor_gene_set"
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

    workflow_id = None
    workflow_label = None

    if options.anchor_genes is not None and len(options.anchor_genes) == 1:
        workflow_id = "F6"
        workflow_label = "single gene anchoring (to %s)" % format_anchor_values_for_label(options.anchor_genes)
    elif options.anchor_genes is not None and len(options.anchor_genes) > 1:
        workflow_id = "F7"
        workflow_label = "multiple gene anchoring (to %s)" % format_anchor_values_for_label(options.anchor_genes)
    elif options.anchor_any_gene:
        workflow_id = "F8"
        workflow_label = "any gene anchoring"
    elif options.anchor_gene_set:
        workflow_id = "F9"
        workflow_label = "gene set anchoring (to input phenotype/gene set)"
    elif options.anchor_phenos is not None and len(options.anchor_phenos) == 1:
        workflow_id = "F4"
        workflow_label = "single phenotype anchoring (to %s) but with phewas statistics used" % format_anchor_values_for_label(options.anchor_phenos)
    elif options.anchor_phenos is not None and len(options.anchor_phenos) > 1:
        workflow_id = "F4"
        workflow_label = "multiple phenotype anchoring (to %s)" % format_anchor_values_for_label(options.anchor_phenos)
    elif options.anchor_any_pheno:
        workflow_id = "F5"
        workflow_label = "any phenotype anchoring"
    else:
        workflow_label = "single phenotype anchoring (to %s) using default statistics" % options.anchor_phenos
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
    if len(selected_anchor_modes) > 1:
        bail_fn(
            "Conflicting anchor workflow flags: %s. Select exactly one of "
            "--anchor-phenos/--anchor-any-pheno/--anchor-genes/--anchor-any-gene/--anchor-gene-set."
            % ", ".join(["--" + mode[0].replace("_", "-") for mode in selected_anchor_modes])
        )

    workflow_id = workflow.get("id") if isinstance(workflow, dict) else None
    standalone_gene_list = has_standalone_gene_list_inputs(options)
    has_x_source = any(
        x is not None
        for x in [options.X_in, options.X_list, options.Xd_in, options.Xd_list]
    )

    if projection_only:
        if has_x_source:
            bail_fn(
                "--factor-gene-clusters-in/--factor-gene-set-clusters-in are projection-only inputs and "
                "cannot be combined with --X-in/--X-list/--Xd-in/--Xd-list."
            )
        if standalone_gene_list:
            bail_fn(
                "--factor-gene-clusters-in/--factor-gene-set-clusters-in cannot be combined with "
                "--gene-list/--gene-list-in/--positive-controls-* workflows."
            )
        if len(selected_anchor_modes) > 0:
            bail_fn(
                "--factor-gene-clusters-in/--factor-gene-set-clusters-in cannot be combined with anchor workflow flags; "
                "choose either projection-only outputs from precomputed factors or a refit clustering workflow."
            )
        if options.gene_set_stats_in is not None:
            bail_fn(
                "--factor-gene-clusters-in/--factor-gene-set-clusters-in skip clustering and cannot be combined with "
                "--gene-set-stats-in. Use --gene-stats-in only as auxiliary metadata if needed."
            )
        return

    if standalone_gene_list and len(selected_anchor_modes) > 0:
        bail_fn(
            "--gene-list/--gene-list-in/--positive-controls-* cannot be combined with anchor workflow flags; "
            "choose one clustering workflow family."
        )

    if workflow_id in set(["F4", "F5"]):
        conflicting = []
        if options.gene_stats_in is not None:
            conflicting.append("--gene-stats-in")
        if options.gene_set_stats_in is not None:
            conflicting.append("--gene-set-stats-in")
        if standalone_gene_list:
            conflicting.append("standalone gene-list flags")
        if conflicting:
            bail_fn(
                "--anchor-phenos/--anchor-any-pheno select PheWAS-driven factorization and cannot be combined with %s."
                % ", ".join(conflicting)
            )
    elif workflow_id in set(["F6", "F7", "F8"]):
        conflicting = []
        if options.gene_stats_in is not None:
            conflicting.append("--gene-stats-in")
        if options.gene_set_stats_in is not None:
            conflicting.append("--gene-set-stats-in")
        if standalone_gene_list:
            conflicting.append("standalone gene-list flags")
        if conflicting:
            bail_fn(
                "--anchor-genes/--anchor-any-gene select PheWAS-driven factorization and cannot be combined with %s."
                % ", ".join(conflicting)
            )
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
    selected_anchor_modes = get_selected_anchor_modes(options)
    if len(selected_anchor_modes) == 0:
        if not bool(mode_state.get("factor_projection_only")) and (
            options.gene_stats_in is not None or options.gene_set_stats_in is not None
        ):
            anchor_mode = "default_stats"
            anchor_values = ["input_gene_stats"]
        else:
            anchor_mode = "none"
            anchor_values = []
    else:
        anchor_mode = selected_anchor_modes[0][0]
        anchor_value = selected_anchor_modes[0][1]
        if isinstance(anchor_value, set):
            anchor_values = sorted(list(anchor_value))
        elif isinstance(anchor_value, (tuple, list)):
            anchor_values = list(anchor_value)
        elif anchor_value is True:
            anchor_values = []
        else:
            anchor_values = [anchor_value]

    trait_linkage_enabled = bool(
        not getattr(options, "no_trait_linkage", False)
        or getattr(options, "trait_factor_links_out", None) is not None
        or getattr(options, "pheno_clusters_out", None) is not None
    )
    trait_linkage_basis = "gene_set" if getattr(options, "project_phenos_from_gene_sets", False) else "gene"
    factorization_source = "precomputed_factor_clusters" if mode_state.get("factor_projection_only") else (
        "phewas" if bool(workflow and workflow.get("use_phewas_for_factoring")) else "default_stats"
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
            "factors_anchor_out": getattr(options, "factors_anchor_out", None),
            "gene_set_clusters_out": getattr(options, "gene_set_clusters_out", None),
            "gene_clusters_out": getattr(options, "gene_clusters_out", None),
            "trait_factor_links_out": getattr(options, "trait_factor_links_out", None),
            "pheno_clusters_out": getattr(options, "pheno_clusters_out", None),
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
    if add_gene_set_flags_present and not workflow["expand_gene_sets"]:
        warn_fn("Ignoring options to add gene sets based on association with anchor genes because only 1 anchor gene was specified")

    if options.anchor_gene_set:
        return

    warning_mode = workflow.get("warn_ignored_y_inputs_mode")
    if warning_mode == "gene_list":
        if options.gene_stats_in is not None or options.gene_set_stats_in is not None:
            warn_fn("Ignoring precomputed gene/gene-set stats in standalone gene-list mode")
        return

    if not has_potentially_ignored_factor_inputs(options):
        return

    if warning_mode == "anchor_phenos":
        warn_fn("Ignoring all arguments for reading Y or reading betas in --anchor-phenos mode")
    elif warning_mode == "anchor_genes":
        warn_fn("Ignoring all arguments for reading Y or reading betas in --anchor-genes mode")


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
