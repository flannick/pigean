"""Validate cross-trait source identity before exposing overlays from another matrix."""
import base64
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eaggl.multitrait_hierarchy import integrate
from pigean.dashboard_factor_atlas import build_factor_atlas


def test_named_leaf_and_gene_join(tmp_path):
    files = []
    hierarchies = {}
    for trait, factor in [('A', 9), ('B', 2)]:
        path = tmp_path / f'{trait}.tsv'
        path.write_text(f'Gene\tFactor{factor}\n{trait}\t0.3\ncommon\t0.7\n')
        files.append((trait, path))
        data = np.array([[0., .7],[0., .3]], dtype='<f4')
        hierarchies[trait] = dict(views=dict(keys=['n98', f'f{factor}'], genes=dict(
            rows=['common',trait], data=base64.b64encode(data).decode())))
    dest = tmp_path/'integrated'
    integrate(files, dest)
    atlas = build_factor_atlas(dest, hierarchies)
    assert [(p['trait'],p['factor'],p['column']) for p in atlas['points']]==[('A','Factor9',1),('B','Factor2',1)]
    with pytest.raises(ValueError, match='corresponding'):
        build_factor_atlas(dest, {'A':hierarchies['A']})
    hierarchies['B']['views']['genes']['rows'].reverse()
    with pytest.raises(ValueError, match='mismatch'):
        build_factor_atlas(dest, hierarchies)
