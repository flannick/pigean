"""Research union factor QC applied to every candidate before phi selection."""
import numpy as np

from eaggl_capacity_research.ard_transfer import state, match
from eaggl_capacity_research.reproducibility_qc import flag_matches
from .selection import Policy, leaf_metrics, restart_support


def factor_qc(weights, raw_starts, replicate=None, policy=Policy()):
    """Return one record per consensus column, including every flag reason."""
    if len(raw_starts) != 5:
        raise ValueError("Restart support requires five primary starts")
    counts = restart_support(weights, raw_starts, policy.support_cosine)
    rows, summary = leaf_metrics(weights, counts, policy)
    replication = {}
    if replicate is not None:
        if weights.shape[0] != replicate.shape[0]:
            raise ValueError("Replicate must use the same gene axis")
        primary = state(weights, 0.)
        secondary = state(replicate, 0.)
        # Zero-contribution columns have no defined direction. Exclude them
        # from assignment and label primary columns explicitly below.
        for basis in (primary, secondary):
            valid = basis["norm"] > 0
            basis["Q"] = basis["Q"][:, valid]
            basis["active"] = basis["active"][valid]
            basis["norm"] = basis["norm"][valid]
        pairs = match(primary, secondary)
        both = flag_matches(pairs, weights.shape[1], replicate.shape[1],
                            cosine_min=policy.replication_cosine,
                            amplitude_min=policy.replication_amplitude_min,
                            amplitude_max=policy.replication_amplitude_max)
        replication = both[both.ensemble == "A"].set_index("factor").to_dict("index")
        for index in set(range(len(rows))) - set(primary["active"]):
            replication[index+1]["status"] = "unassessed"

    for row in rows:
        other = replication.get(row["factor"], {})
        status = other.get("status", "unassessed")
        reasons = [key for key in ("small","uniform","duplicate","unsupported") if row[key]]
        if status in ("profile_not_reproduced","strength_variable"):
            reasons.append(status)
        row.update(replication=status, counterpart=other.get("counterpart"),
                   pair_cosine=other.get("pair_cosine"),
                   amplitude_forward=other.get("amplitude_to_other"),
                   amplitude_reverse=other.get("amplitude_from_other"),
                   flag_reasons="|".join(reasons), flagged=bool(reasons), retained=True)
    flagged = [r for r in rows if r["flagged"]]
    fraction = len(flagged)/len(rows) if rows else 1.
    energy = sum(r["energy"] for r in rows)
    energy_fraction = sum(r["energy"] for r in flagged)/energy if energy else 1.
    summary.update(flagged=len(flagged), flagged_fraction=fraction, flagged_energy=energy_fraction,
                   independent_ensemble_available=replicate is not None,
                   replication_assessed=sum(r["replication"] != "unassessed" for r in rows),
                   leaf_usable=bool(rows and fraction <= policy.maximum_flagged_fraction+1e-12
                                    and energy_fraction <= policy.maximum_flagged_energy+1e-12))
    return rows, summary
