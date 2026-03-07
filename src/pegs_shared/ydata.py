from pegs_shared.types import HyperparameterData, PhewasRuntimeState, RuntimeStateBundle, YData


def y_data_from_runtime(runtime):
    return YData.from_runtime(runtime)


def build_y_data_from_inputs(
    runtime,
    Y,
    Y_for_regression=None,
    Y_exomes=None,
    Y_positive_controls=None,
    Y_case_counts=None,
    Y_corr_m=None,
    store_corr_sparse=False,
    min_correlation=0,
):
    return YData.from_inputs(
        runtime,
        Y,
        Y_for_regression=Y_for_regression,
        Y_exomes=Y_exomes,
        Y_positive_controls=Y_positive_controls,
        Y_case_counts=Y_case_counts,
        Y_corr_m=Y_corr_m,
        store_corr_sparse=store_corr_sparse,
        min_correlation=min_correlation,
    )


def apply_y_data_to_runtime(runtime, y_data):
    y_data.apply_to_runtime(runtime)


def set_runtime_y_from_inputs(
    runtime,
    Y,
    Y_for_regression=None,
    Y_exomes=None,
    Y_positive_controls=None,
    Y_case_counts=None,
    Y_corr_m=None,
    store_corr_sparse=False,
    min_correlation=0,
):
    y_data = build_y_data_from_inputs(
        runtime=runtime,
        Y=Y,
        Y_for_regression=Y_for_regression,
        Y_exomes=Y_exomes,
        Y_positive_controls=Y_positive_controls,
        Y_case_counts=Y_case_counts,
        Y_corr_m=Y_corr_m,
        store_corr_sparse=store_corr_sparse,
        min_correlation=min_correlation,
    )
    apply_y_data_to_runtime(runtime, y_data)
    return y_data


def hyperparameter_data_from_runtime(runtime):
    return HyperparameterData.from_runtime(runtime)


def apply_hyperparameter_data_to_runtime(runtime, hyper_data):
    hyper_data.apply_to_runtime(runtime)


def ensure_hyperparameter_state(runtime):
    hyper_state = getattr(runtime, "hyperparameter_state", None)
    if isinstance(hyper_state, HyperparameterData):
        return hyper_state
    hyper_state = hyperparameter_data_from_runtime(runtime)
    runtime.hyperparameter_state = hyper_state
    return hyper_state


def phewas_runtime_state_from_runtime(runtime):
    return PhewasRuntimeState.from_runtime(runtime)


def apply_phewas_runtime_state_to_runtime(runtime, phewas_state):
    phewas_state.apply_to_runtime(runtime)


def sync_y_state(runtime):
    y_state = y_data_from_runtime(runtime)
    apply_y_data_to_runtime(runtime, y_state)
    return y_state


def sync_hyperparameter_state(runtime):
    hyper_state = ensure_hyperparameter_state(runtime)
    apply_hyperparameter_data_to_runtime(runtime, hyper_state)
    return hyper_state


def sync_phewas_runtime_state(runtime):
    phewas_state = phewas_runtime_state_from_runtime(runtime)
    apply_phewas_runtime_state_to_runtime(runtime, phewas_state)
    return phewas_state


def runtime_state_bundle_from_runtime(runtime):
    return RuntimeStateBundle.from_runtime(runtime)


def apply_runtime_state_bundle_to_runtime(runtime, runtime_state_bundle):
    runtime_state_bundle.apply_to_runtime(runtime)
    return runtime_state_bundle


def sync_runtime_state_bundle(runtime):
    runtime_state_bundle = runtime_state_bundle_from_runtime(runtime)
    apply_runtime_state_bundle_to_runtime(runtime, runtime_state_bundle)
    return runtime_state_bundle
