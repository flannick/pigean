"""Canonical in-repo EAGGL package snapshot.

This package is introduced during repo consolidation. The current runtime
implementation still lives in ``legacy_main.py`` until the entrypoint
refactor lands.
"""

from .legacy_main import *  # noqa: F401,F403
