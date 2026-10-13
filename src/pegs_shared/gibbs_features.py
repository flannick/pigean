"""Explicit, scoped Gibbs feature selection; presets retain historical behavior."""
from contextvars import ContextVar


FEATURES = (
    'sparse-regression', 'grouped-regression', 'annotation-cache',
    'compact-covariance', 'lean-regression', 'cached-inner',
    'compiled-covariance', 'parallel-prefilter', 'inner-matmul', 'inner-fused', 'inner-batching',
    'compiled-prefilter',
    'inner-cache64',
)
PRESETS = dict(zip(
    ('legacy', 'sparse', 'grouped', 'grouped-cached', 'compact-covariance',
     'lean-regression', 'cached-inner', 'compiled-covariance', 'parallel-prefilter'),
    (frozenset(FEATURES[:i]) for i in range(9)),
))
ALIASES = {'sparse-regression': 'sparse', 'grouped-regression': 'grouped',
           'cached-grouped-regression': 'grouped-cached'}
REQUIRES = {
    'inner-cache64': {'cached-inner'},
    'compiled-prefilter': {'sparse-regression'},
    'annotation-cache': {'grouped-regression'},
    'compact-covariance': {'grouped-regression'},
    'lean-regression': {'grouped-regression'},
    'compiled-covariance': {'compact-covariance'},
    'parallel-prefilter': {'sparse-regression'},
}
ACTIVE = ContextVar('gibbs_active_features', default=frozenset())


def resolve_features(preset='legacy', enable=(), disable=()):
    preset = ALIASES.get(preset, preset)
    if preset not in PRESETS:
        raise ValueError('Unknown Gibbs preset: '+preset)
    enable, disable = set(enable or ()), set(disable or ())
    unknown = (enable | disable) - set(FEATURES)
    if unknown:
        raise ValueError('Unknown Gibbs features: '+', '.join(sorted(unknown)))
    if enable & disable:
        raise ValueError('Cannot both enable and disable: '+', '.join(sorted(enable & disable)))
    requested = (set(PRESETS[preset]) | enable) - disable
    active = set(requested)
    while True:
        reduced = {f for f in active if REQUIRES.get(f, set()) <= active}
        if reduced == active:
            break
        active = reduced
    return {'preset': preset, 'requested': sorted(requested), 'active': sorted(active),
            'inactive': {f: sorted(REQUIRES[f] - active) for f in sorted(requested-active)}}


def enabled(feature):
    return feature in ACTIVE.get()
