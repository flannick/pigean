from __future__ import annotations


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
        "required_inputs": ["--run-phewas-from-gene-phewas-stats-in"],
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
        elif flag == "--run-phewas-from-gene-phewas-stats-in":
            if options.run_phewas_from_gene_phewas_stats_in is None:
                missing_inputs.append(flag)
    return missing_inputs


def build_factor_workflow_error(workflow_id, missing_inputs):
    if len(missing_inputs) == 0:
        return None
    if workflow_id in ("F4", "F5", "F6", "F7", "F8"):
        return "Require --gene-set-phewas-stats-in and --gene-phewas-stats-in"
    if workflow_id == "F9":
        return "Require --run-phewas-from-gene-phewas-stats"
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
