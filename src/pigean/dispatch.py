from __future__ import annotations

from . import outputs as pigean_outputs
from . import pipeline as pigean_pipeline


def run_main_pipeline(domain, options, mode):
    if not options.hide_opts:
        domain.log("Python version: %s" % domain.sys.version)
        domain.log("Numpy version: %s" % domain.np.__version__)
        domain.log("Scipy version: %s" % domain.scipy.__version__)
        domain.log("Options: %s" % options)
    state = domain._build_runtime_state(options)
    mode_state = domain._build_mode_state(mode, options.run_phewas_from_gene_phewas_stats_in)

    sigma2_cond = domain._configure_hyperparameters_for_main(state, options)
    y_not_loaded = domain._load_main_Y_inputs(state, options, mode_state)

    non_huge_result = None
    if not mode_state["run_huge"]:
        non_huge_result = pigean_pipeline.run_main_non_huge_pipeline(
            domain=domain,
            state=state,
            options=options,
            mode_state=mode_state,
            sigma2_cond=sigma2_cond,
            y_not_loaded=y_not_loaded,
        )

    pigean_outputs.write_main_outputs_and_optional_phewas(
        domain=domain,
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
