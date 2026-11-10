"""Bounded scheduling of complete requirements fits; no fit-score retuning."""
import math

from .phi_requirements import PHIS, ERROR_TOLERANCE

MODES = ('improvement', 'noninferior', 'none')
INITIAL_ORDER = (0.01, 0.005, 0.02)


def validate(extension, max_extra):
    if extension not in MODES:
        raise ValueError('phi search extension must be improvement, noninferior or none')
    if isinstance(max_extra, bool) or not isinstance(max_extra, int) or max_extra < 0:
        raise ValueError('phi search max extra must be a nonnegative integer')


def next_step(decision, extension='improvement', max_extra=2):
    """Require an admissible edge and a material gain over *each* interior view.

    A gain is a direct local/global normalized-error decrease beyond the existing
    preservation tolerance, not a p-value. All previously tested evidence remains
    in the evaluator. No numerical failure or redundancy-only loss drives search.
    Shallower edges run first, with lower phi breaking ties; the budget is shared.
    """
    validate(extension, max_extra)
    views = {v['phi']: v for v in decision['views']}
    phis = sorted(views)
    if not set(PHIS).issubset(views):
        raise ValueError('The initial three candidates must be assessed first')
    result = dict(next_phi=None, reason=None, edges=[], candidates_tested=len(phis),
                  extra_candidates=len(phis)-3, extension=extension, max_extra=max_extra)
    if not decision['all_candidates_qualified']:
        return dict(result, reason='numerical_uncertainty')
    if extension == 'none':
        return dict(result, reason='extension_disabled')
    if not decision['selection_qualified']:
        return dict(result, reason='unresolved_comparison')
    errors = {p: {} for p in phis}
    for row in decision.get('errors', []):
        if row['evaluable']:
            key = (row['source_phi'], row['source_factor'], row['other_phi'], row['other_factor'])
            errors[row['candidate_phi']][key] = row['error']
    eligible = []
    for edge, direction in [(phis[0], 'lower'), (phis[-1], 'upper')]:
        record = dict(phi=edge, direction=direction, eligible=False, comparisons=[])
        if not views[edge]['admissible']:
            record['reason'] = 'edge_rejected'
        elif extension == 'noninferior':
            record.update(eligible=True, reason='acceptable_edge')
        else:
            for interior in phis[1:-1]:
                if errors[interior].keys() != errors[edge].keys():
                    raise ValueError('Inconsistent regional comparisons')
                local = max((errors[interior][k]-errors[edge][k] for k in errors[edge]), default=0.)
                global_gain = views[interior]['global_error']-views[edge]['global_error']
                material = (not views[interior]['admissible'] and
                            max(local, global_gain) > ERROR_TOLERANCE + 1e-12)
                record['comparisons'].append(dict(interior_phi=interior,
                    max_local_error_gain=local, global_error_gain=global_gain, material=material))
            record['eligible'] = bool(record['comparisons'] and all(c['material'] for c in record['comparisons']))
            record['reason'] = 'meaningful_improvement' if record['eligible'] else 'no_meaningful_improvement'
        result['edges'].append(record)
        if record['eligible']:
            eligible.append(record)
    if not eligible:
        return dict(result, reason='no_eligible_edge')
    if result['extra_candidates'] >= max_extra:
        return dict(result, reason='extra_budget_exhausted')
    chosen = min(eligible, key=lambda e: (abs(math.log2(e['phi']/.01)), e['phi']))
    proposed = chosen['phi'] / 2 if chosen['direction'] == 'lower' else chosen['phi'] * 2
    if not math.isfinite(proposed) or proposed <= 0 or proposed in views:
        return dict(result, reason='numeric_range_limit')
    return dict(result, next_phi=proposed, reason=chosen['reason'], direction=chosen['direction'])
