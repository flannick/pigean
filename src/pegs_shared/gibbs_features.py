"""Scoped Gibbs features; numerical defaults are independent of arithmetic presets."""
from contextlib import contextmanager
from contextvars import ContextVar


FEATURES = (
    'sparse-regression', 'grouped-regression', 'annotation-cache',
    'compact-covariance', 'lean-regression', 'cached-inner',
    'compiled-covariance', 'parallel-prefilter', 'inner-matmul', 'inner-fused', 'inner-batching',
    'compiled-prefilter',
    'inner-cache64',
    'huge-prior-softmax',
    'static-correlation',
    'inner-rhat-stable',
    'compiled-moments',
    'warm-start-layout',
    'inner-centered-variance',
)
PRESETS = dict(zip(
    ('legacy', 'sparse', 'grouped', 'grouped-cached', 'compact-covariance',
     'lean-regression', 'cached-inner', 'compiled-covariance', 'parallel-prefilter'),
    (frozenset(FEATURES[:i]) for i in range(9)),
))
ALIASES = {'sparse-regression': 'sparse', 'grouped-regression': 'grouped',
           'cached-grouped-regression': 'grouped-cached'}
REQUIRES = {
    'compiled-moments': {'grouped-regression'},
    'inner-cache64': {'cached-inner'},
    'compiled-prefilter': {'sparse-regression'},
    'annotation-cache': {'grouped-regression'},
    'compact-covariance': {'grouped-regression'},
    'lean-regression': {'grouped-regression'},
    'compiled-covariance': {'compact-covariance'},
    'parallel-prefilter': {'sparse-regression'},
}
DEFAULT_FEATURES = frozenset({'inner-rhat-stable', 'warm-start-layout'})
# CLI genetic-input Gibbs defaults; deliberately excluded from preprocessing and
# direct Python contexts. The CLI passes these as ordinary stage enable flags.
GIBBS_STAGE_DEFAULT_FEATURES = frozenset({'inner-centered-variance'})
PORTABLE_EXTENSIONS = frozenset({'inner-matmul', 'inner-cache64', 'static-correlation'})
COMPILED_EXTENSIONS = frozenset({
    'inner-fused', 'inner-batching', 'compiled-prefilter', 'compiled-moments',
    'huge-prior-softmax',
})
ACTIVE = ContextVar('gibbs_active_features', default=DEFAULT_FEATURES)


def _feature_names(values):
    """Flatten repeated options and comma lists without accepting empty names."""
    if isinstance(values, str):
        values = (values,)
    names = set()
    for value in values or ():
        for part in value.split(','):
            name = part.strip()
            if not name:
                raise ValueError('Empty Gibbs feature name in: %r' % value)
            names.add(name)
    return names


def resolve_features(preset='legacy', enable=(), disable=()):
    preset = ALIASES.get(preset, preset)
    if preset not in PRESETS:
        raise ValueError('Unknown Gibbs preset: '+preset)
    enable, disable = _feature_names(enable), _feature_names(disable)
    unknown = (enable | disable) - set(FEATURES)
    if unknown:
        raise ValueError('Unknown Gibbs features: '+', '.join(sorted(unknown)))
    if enable & disable:
        raise ValueError('Cannot both enable and disable: '+', '.join(sorted(enable & disable)))
    requested = (set(PRESETS[preset]) | DEFAULT_FEATURES | enable) - disable
    active = set(requested)
    while True:
        reduced = {f for f in active if REQUIRES.get(f, set()) <= active}
        if reduced == active:
            break
        active = reduced
    return {'preset': preset, 'requested': sorted(requested), 'active': sorted(active),
            'inactive': {f: sorted(REQUIRES[f] - active) for f in sorted(requested-active)}}


def enable_gibbs_stage_defaults(enable=(), disable=(), *, extensions=()):
    """Add CLI stage defaults while retaining explicit opt-outs and conflicts."""
    defaults = GIBBS_STAGE_DEFAULT_FEATURES.union(extensions)
    return sorted(_feature_names(enable) | (defaults - _feature_names(disable)))


def genetic_gibbs_defaults(engine, enable=(), disable=()):
    """Choose the automatic speed features only when no engine was specified."""
    extensions = frozenset()
    if engine is None:
        try:
            from numba import njit
        except (ImportError, OSError):
            # The parallel-prefilter preset includes compiled covariance. Its
            # threaded NumPy prefilter can also run on the cached-inner preset.
            engine = 'cached-inner'
            extensions = PORTABLE_EXTENSIONS | {'parallel-prefilter'}
        else:
            engine = 'parallel-prefilter'
            extensions = PORTABLE_EXTENSIONS | COMPILED_EXTENSIONS
    return engine, enable_gibbs_stage_defaults(enable, disable, extensions=extensions)


def enabled(feature):
    return feature in ACTIVE.get()


@contextmanager
def default_feature_scope(*, enable=(), disable=()):
    """Apply numerical defaults/opt-outs during preprocessing as well as Gibbs.

    Arithmetic features remain scoped by marginal_backend at the Gibbs stage.
    Preserve any arithmetic context supplied by a Python caller.
    """
    selected = set(resolve_features(enable=enable, disable=disable)['active'])
    token = ACTIVE.set((ACTIVE.get() - DEFAULT_FEATURES) | (selected & DEFAULT_FEATURES))
    try:
        yield
    finally:
        ACTIVE.reset(token)
