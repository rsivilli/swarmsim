"""RF emitter simulation and SigMF IQ generation (see docs/design/rf-sigmf.md).

Phase 0 provides the emission schedule, link budget, ground-truth tables and storage sizing;
IQ synthesis and SigMF writing follow in later phases.
"""

from __future__ import annotations

from .config import parse_rf
from .library import build_library, load_builtin_profiles
from .runner import plan_rf, write_outputs

__all__ = ["parse_rf", "build_library", "load_builtin_profiles", "plan_rf", "write_outputs"]
