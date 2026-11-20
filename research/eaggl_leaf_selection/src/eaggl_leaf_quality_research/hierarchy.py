"""Expose the shared tree implementation to standalone research scripts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from eaggl.hierarchy import RULE_FIELDS, build_tree, cut_tree, distances, profile_distances

__all__ = ["RULE_FIELDS", "build_tree", "cut_tree", "distances", "profile_distances"]
