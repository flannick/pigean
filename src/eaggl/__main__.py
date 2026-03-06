"""Temporary package entrypoint for the canonical in-repo EAGGL snapshot."""

from . import legacy_main


if __name__ == "__main__":
    raise SystemExit(legacy_main.main())
