from __future__ import annotations

from . import main_support as pigean_main_support
from . import multi_y as pigean_multi_y
from . import outputs as pigean_outputs
from . import pipeline as pigean_pipeline


def run_main_pipeline(options, mode, services=None):
    if services is None:
        services = pigean_main_support.build_cli_services()
    if not options.hide_opts:
        services.log("Python version: %s" % services.sys.version)
        services.log("Numpy version: %s" % services.np.__version__)
        services.log("Scipy version: %s" % services.scipy.__version__)
        services.log("Options: %s" % options)
    if getattr(options, "multi_y_in", None) is not None:
        return pigean_multi_y.run_multi_y_pipeline(services=services, options=options, mode=mode)
    state = pigean_main_support.build_runtime_state(options)
    mode_state = pigean_main_support.build_mode_state(mode, options.run_phewas_from_gene_phewas_stats_in)

    sigma2_cond = pigean_main_support.configure_hyperparameters_for_main(state, options)
    y_not_loaded = pigean_main_support.load_main_y_inputs(state, options, mode_state)
    pigean_main_support.record_resolved_runtime_options(
        state,
        options,
        mode,
        mode_state,
        sigma2_cond,
        y_not_loaded,
    )

    non_huge_result = None
    if not mode_state["run_huge"]:
        non_huge_result = pigean_pipeline.run_main_non_huge_pipeline(
            services=services,
            state=state,
            options=options,
            mode_state=mode_state,
            sigma2_cond=sigma2_cond,
            y_not_loaded=y_not_loaded,
        )

    pigean_outputs.write_main_outputs_and_optional_phewas(
        services=services,
        state=state,
        options=options,
        mode_state=mode_state,
        mode=mode,
    )

    return pigean_pipeline.MainPipelineResult(
        state=state,
        mode_state=mode_state,
        sigma2_cond=sigma2_cond,
        y_not_loaded=y_not_loaded,
        non_huge=non_huge_result,
    )
