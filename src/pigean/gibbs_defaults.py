"""Select a CLI sampling schedule without changing the native controller."""

# Native inclusive burn boundary: updates 1--250 discarded, 251--1250 retained.
RETAINED_SCHEDULE = {
    "min_num_burn_in": 251,
    "max_num_burn_in": 251,
    "min_num_post_burn_in": 1000,
    "max_num_post_burn_in": 1000,
    "max_num_iter": 1251,
    "total_num_iter_gibbs": 1251,
    "gibbs_reruns": 1,
    "max_num_restarts": 0,
    "disable_stall_detection": True,
}

# Explicit controller customization opts out of automatic schedule selection.
# Precision thresholds can still be set independently; the retained schedule's
# minimum sample count prevents them from shortening its qualified budget.
CONTROLLER_OVERRIDES = set(RETAINED_SCHEDULE) | {
    "burn_in_stall_window", "stall_window", "stall_recent_window",
    "stall_min_burn_in", "stall_min_post_burn_in",
}


def apply_sampling_schedule(options, mode, explicit, bail):
    requested = options.gibbs_schedule
    genetic_gibbs = mode == "gibbs" and (
        options.gwas_in is not None or options.huge_statistics_in is not None
    )
    use_retained = requested == "retained" or (
        requested == "auto" and genetic_gibbs and not explicit & CONTROLLER_OVERRIDES
    )
    if requested == "retained" and mode != "gibbs":
        bail("Option --gibbs-schedule retained requires gibbs mode")
    if use_retained:
        conflicts = [name for name in sorted(explicit & CONTROLLER_OVERRIDES)
                     if name not in RETAINED_SCHEDULE or
                     getattr(options, name) != RETAINED_SCHEDULE[name]]
        if conflicts:
            bail("Option --gibbs-schedule retained conflicts with explicit controller settings: "
                 + ", ".join("--" + name.replace("_", "-") for name in conflicts)
                 + "; use --gibbs-schedule legacy for a custom budget")
        for name, value in RETAINED_SCHEDULE.items():
            setattr(options, name, value)
    options.gibbs_schedule_resolved = "retained" if use_retained else "legacy"
