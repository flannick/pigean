import copy
import os

from pegs_shared.cli import _default_bail
from pegs_shared.types import ReadXPipelineConfig, XData, XInputPlan, XReadIngestionOptions, XReadPostOptions


def _normalize_input_specs(input_specs):
    if input_specs is None:
        return ([], [])
    if type(input_specs) == str:
        return ([input_specs], [input_specs])
    if type(input_specs) == list:
        return (input_specs, copy.copy(input_specs))
    return ([], [])


def build_read_x_pipeline_config(X_in, overrides=None, *, bail_fn=None):
    if bail_fn is None:
        bail_fn = _default_bail

    if isinstance(overrides, ReadXPipelineConfig):
        config = copy.deepcopy(overrides)
        if X_in is not None:
            config.X_in = X_in
    else:
        config = ReadXPipelineConfig(X_in=X_in)
        if overrides is not None:
            for key, value in overrides.items():
                if not hasattr(config, key):
                    bail_fn("Unknown read-X pipeline option '%s'" % key)
                setattr(config, key, value)

    if config.x_sparsify is None:
        config.x_sparsify = [50, 100, 200, 500, 1000]
    else:
        config.x_sparsify = list(config.x_sparsify)

    if config.ignore_genes is None:
        config.ignore_genes = set(["NA"])
    else:
        config.ignore_genes = set(config.ignore_genes)

    return config


def _append_initial_p_indices(initial_ps, input_specs, xin_to_p_noninf_ind):
    if initial_ps is None:
        return
    for input_spec in input_specs:
        assert(input_spec in xin_to_p_noninf_ind)
        initial_ps.append(xin_to_p_noninf_ind[input_spec])


def _map_initial_p_indices_to_values(initial_ps, initial_p):
    if initial_ps is None:
        return
    assert(type(initial_p) is list)
    for i in range(len(initial_ps)):
        assert(initial_ps[i]) >= 0 and initial_ps[i] < len(initial_p)
        initial_ps[i] = initial_p[initial_ps[i]]


def _expand_x_inputs(x_inputs, orig_files, batch_separator="@", file_separator=None):
    expanded_inputs = []
    batches = []
    labels = []
    expanded_orig_files = []
    for i in range(len(x_inputs)):
        x_input = x_inputs[i]
        orig_file = orig_files[i]
        batch = None
        label = os.path.basename(orig_file)
        if "." in label:
            label = ".".join(label.split(".")[:-1])
        if batch_separator in x_input:
            batch = x_input.split(batch_separator)[-1]
            label = batch
            x_input = batch_separator.join(x_input.split(batch_separator)[:-1])

        (x_input, tag) = remove_tag_from_input(x_input)
        if tag is not None:
            label = tag

        if file_separator is not None:
            x_to_add = x_input.split(file_separator)
        else:
            x_to_add = [x_input]

        expanded_inputs += x_to_add
        batches += [batch] * len(x_to_add)
        labels += [label] * len(x_to_add)
        expanded_orig_files += [orig_file] * len(x_to_add)
    return (expanded_inputs, batches, labels, expanded_orig_files)


def _append_inputs_from_list_files(
    list_specs,
    dest_inputs,
    dest_orig_files,
    list_open_fn,
    strip_fn,
    resolve_relative_paths=False,
    skip_empty_lines=True,
    initial_ps=None,
    xin_to_p_noninf_ind=None,
    batch_separator="@",
):
    if list_specs is None:
        return

    if type(list_specs) == str:
        list_specs = [list_specs]

    for list_spec in list_specs:
        original_list_spec = list_spec
        batch = None
        if batch_separator in list_spec:
            batch = list_spec.split(batch_separator)[-1]
            list_spec = batch_separator.join(list_spec.split(batch_separator)[:-1])

        list_path, list_tag = remove_tag_from_input(list_spec)
        if _is_direct_sparse_matrix_spec(list_path):
            direct_input = add_tag_to_input(list_path, list_tag)
            if batch is not None and batch_separator not in direct_input:
                direct_input = "%s%s%s" % (direct_input, batch_separator, batch)
            dest_inputs.append(direct_input)
            if initial_ps is not None:
                assert(original_list_spec in xin_to_p_noninf_ind)
                initial_ps.append(xin_to_p_noninf_ind[original_list_spec])
            dest_orig_files.append(original_list_spec)
            continue

        list_dir = os.path.dirname(os.path.abspath(list_spec))
        with list_open_fn(list_spec) as list_fh:
            for raw_line in list_fh:
                line = strip_fn(raw_line)
                if skip_empty_lines and len(line) == 0:
                    continue

                if resolve_relative_paths:
                    (path, label) = remove_tag_from_input(line)
                    if path and not os.path.isabs(path):
                        path = os.path.normpath(os.path.join(list_dir, path))
                    line = add_tag_to_input(path, label)

                if batch is not None and batch_separator not in line:
                    line = "%s%s%s" % (line, batch_separator, batch)

                dest_inputs.append(line)
                if initial_ps is not None:
                    assert(list_spec in xin_to_p_noninf_ind)
                    initial_ps.append(xin_to_p_noninf_ind[list_spec])
                dest_orig_files.append(list_spec)


def prepare_read_x_inputs(
    X_in,
    X_list,
    Xd_in,
    Xd_list,
    initial_p,
    xin_to_p_noninf_ind,
    batch_separator,
    file_separator,
    *,
    sparse_list_open_fn,
    dense_list_open_fn,
):
    initial_ps = None
    if initial_p is not None:
        if type(initial_p) is not list:
            initial_p = [initial_p]
        initial_ps = []
        assert(xin_to_p_noninf_ind is not None)

    (X_ins, orig_files) = _normalize_input_specs(X_in)
    _append_initial_p_indices(initial_ps, X_ins, xin_to_p_noninf_ind)
    _append_inputs_from_list_files(
        list_specs=X_list,
        dest_inputs=X_ins,
        dest_orig_files=orig_files,
        list_open_fn=sparse_list_open_fn,
        strip_fn=lambda line: line.strip(),
        resolve_relative_paths=True,
        skip_empty_lines=True,
        initial_ps=initial_ps,
        xin_to_p_noninf_ind=xin_to_p_noninf_ind,
        batch_separator=batch_separator,
    )
    X_ins, batches, labels, orig_files = _expand_x_inputs(
        X_ins,
        orig_files,
        batch_separator=batch_separator,
        file_separator=file_separator,
    )
    is_dense = [False for _ in X_ins]

    (Xd_ins, orig_dfiles) = _normalize_input_specs(Xd_in)
    _append_initial_p_indices(initial_ps, Xd_ins, xin_to_p_noninf_ind)
    _append_inputs_from_list_files(
        list_specs=Xd_list,
        dest_inputs=Xd_ins,
        dest_orig_files=orig_dfiles,
        list_open_fn=dense_list_open_fn,
        strip_fn=lambda line: line.strip("\n"),
        resolve_relative_paths=False,
        skip_empty_lines=False,
        initial_ps=initial_ps,
        xin_to_p_noninf_ind=xin_to_p_noninf_ind,
        batch_separator=batch_separator,
    )
    Xd_ins, batches2, labels2, orig_dfiles = _expand_x_inputs(
        Xd_ins,
        orig_dfiles,
        batch_separator=batch_separator,
        file_separator=file_separator,
    )

    _map_initial_p_indices_to_values(initial_ps, initial_p)

    X_ins += Xd_ins
    batches += batches2
    labels += labels2
    orig_files += orig_dfiles
    is_dense += [True for _ in Xd_ins]
    return XInputPlan(
        initial_ps=initial_ps,
        X_ins=X_ins,
        batches=batches,
        labels=labels,
        orig_files=orig_files,
        is_dense=is_dense,
    )


def xdata_from_input_plan(input_plan):
    return XData.from_input_plan(input_plan)


def build_read_x_ingestion_options(local_vars):
    return XReadIngestionOptions(
        batch_all_for_hyper=local_vars["batch_all_for_hyper"],
        first_for_hyper=local_vars["first_for_hyper"],
        update_hyper_sigma=local_vars["update_hyper_sigma"],
        update_hyper_p=local_vars["update_hyper_p"],
        first_for_sigma_cond=local_vars["first_for_sigma_cond"],
        run_corrected_ols=local_vars["run_corrected_ols"],
        gene_cor_file=local_vars["gene_cor_file"],
        gene_loc_file=local_vars["gene_loc_file"],
        gene_cor_file_gene_col=local_vars["gene_cor_file_gene_col"],
        gene_cor_file_cor_start_col=local_vars["gene_cor_file_cor_start_col"],
        run_logistic=local_vars["run_logistic"],
        max_for_linear=local_vars["max_for_linear"],
        only_ids=local_vars["only_ids"],
        add_all_genes=local_vars["add_all_genes"],
        only_inc_genes=local_vars["only_inc_genes"],
        fraction_inc_genes=local_vars["fraction_inc_genes"],
        ignore_genes=local_vars["ignore_genes"],
        max_num_entries_at_once=local_vars["max_num_entries_at_once"],
        filter_gene_set_p=local_vars["filter_gene_set_p"],
        filter_gene_set_metric_z=local_vars["filter_gene_set_metric_z"],
        filter_using_phewas=local_vars["filter_using_phewas"],
    )


def build_read_x_post_options(local_vars, *, batches, num_ignored_gene_sets, ignored_for_fraction_inc):
    return XReadPostOptions(
        ignored_for_fraction_inc=ignored_for_fraction_inc,
        filter_gene_set_p=local_vars["filter_gene_set_p"],
        correct_betas_mean=local_vars["correct_betas_mean"],
        correct_betas_var=local_vars["correct_betas_var"],
        filter_using_phewas=local_vars["filter_using_phewas"],
        prune_gene_sets=local_vars["prune_gene_sets"],
        weighted_prune_gene_sets=local_vars["weighted_prune_gene_sets"],
        prune_deterministically=local_vars["prune_deterministically"],
        max_num_gene_sets_initial=local_vars["max_num_gene_sets_initial"],
        skip_betas=local_vars["skip_betas"],
        initial_p=local_vars["initial_p"],
        update_hyper_p=local_vars["update_hyper_p"],
        sigma_power=local_vars["sigma_power"],
        initial_sigma2_cond=local_vars["initial_sigma2_cond"],
        update_hyper_sigma=local_vars["update_hyper_sigma"],
        initial_sigma2=local_vars["initial_sigma2"],
        sigma_soft_threshold_95=local_vars["sigma_soft_threshold_95"],
        sigma_soft_threshold_5=local_vars["sigma_soft_threshold_5"],
        batches=batches,
        num_ignored_gene_sets=num_ignored_gene_sets,
        first_for_hyper=local_vars["first_for_hyper"],
        max_num_gene_sets_hyper=local_vars["max_num_gene_sets_hyper"],
        first_for_sigma_cond=local_vars["first_for_sigma_cond"],
        first_max_p_for_hyper=local_vars["first_max_p_for_hyper"],
        max_num_burn_in=local_vars["max_num_burn_in"],
        max_num_iter_betas=local_vars["max_num_iter_betas"],
        min_num_iter_betas=local_vars["min_num_iter_betas"],
        num_chains_betas=local_vars["num_chains_betas"],
        r_threshold_burn_in_betas=local_vars["r_threshold_burn_in_betas"],
        use_max_r_for_convergence_betas=local_vars["use_max_r_for_convergence_betas"],
        max_frac_sem_betas=local_vars["max_frac_sem_betas"],
        max_allowed_batch_correlation=local_vars["max_allowed_batch_correlation"],
        sigma_num_devs_to_top=local_vars["sigma_num_devs_to_top"],
        p_noninf_inflate=local_vars["p_noninf_inflate"],
        sparse_solution=local_vars["sparse_solution"],
        sparse_frac_betas=local_vars["sparse_frac_betas"],
        betas_trace_out=local_vars["betas_trace_out"],
        increase_filter_gene_set_p=local_vars["increase_filter_gene_set_p"],
        min_gene_set_size=local_vars["min_gene_set_size"],
        max_gene_set_size=local_vars["max_gene_set_size"],
        filter_gene_set_metric_z=local_vars["filter_gene_set_metric_z"],
        max_num_gene_sets=local_vars["max_num_gene_sets"],
    )


def initialize_matrix_and_gene_index_state(runtime, batch_size):
    XData.initialized_runtime_state(batch_size).apply_to_runtime(runtime)


def _is_direct_sparse_matrix_spec(input_path):
    lower_input_path = input_path.lower()
    return os.path.isfile(input_path) and (
        lower_input_path.endswith(".gmt") or lower_input_path.endswith(".gmt.gz")
    )


def remove_tag_from_input(x_in, tag_separator=":"):
    tag = None
    if tag_separator in x_in:
        tag_index = x_in.index(tag_separator)
        tag = x_in[:tag_index]
        x_in = x_in[tag_index + 1 :]
        if len(tag) == 0:
            tag = None
    return (x_in, tag)


def add_tag_to_input(x_in, tag, tag_separator=":"):
    if tag is None:
        return x_in
    return tag_separator.join([tag, x_in])
