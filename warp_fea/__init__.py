"""
warp_fea — GPU linear-static structural FEA on NVIDIA Warp, shaped to drop into
the LLM Judge System's Stage 3a in place of the CalculiX runner.

    from warp_fea import solve_structural
    result = solve_structural("part.msh", load_case)
    result.measured["max_von_mises_stress"].value    # Pa
    result.solver_status.converged                   # never trust measured if False

All inputs and outputs are SI (Pa, m, N). See PHASE3_SPEC.md for the contract.
"""

from .results import FEAResult, Location, Measurement, SolverStatus
from .mesh_io import FEMesh, Region, read_mesh
from .solver import solve_structural, solve_inp
from .modal import solve_modal
from .buckling import solve_buckling
from .load_case_adapter import CanonicalLoadCase, parse_load_case
from .validation import counter_report, validate_inputs

__all__ = [
    "solve_structural", "solve_inp", "solve_modal", "solve_buckling",
    "FEAResult", "Measurement", "Location", "SolverStatus",
    "FEMesh", "Region", "read_mesh",
    # PHASE4 §B input validation + §A adapter seam
    "validate_inputs", "counter_report", "parse_load_case", "CanonicalLoadCase",
]
