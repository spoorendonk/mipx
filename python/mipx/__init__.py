from importlib.metadata import PackageNotFoundError, version

from ._core import (
    HeuristicRuntimeMode,
    LpProblem,
    MipPreRootStats,
    MipResult,
    MipSolver,
    ModelFormat,
    ModelFormatError,
    ParallelMode,
    RootLpPolicy,
    SearchProfile,
    Sense,
    Status,
    VarType,
    read_lp,
    read_model,
    read_mps,
    solve_file,
    solve_mps,
    write_mps,
)

try:
    __version__ = version("mipx")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "HeuristicRuntimeMode",
    "LpProblem",
    "MipPreRootStats",
    "MipResult",
    "MipSolver",
    "ModelFormat",
    "ModelFormatError",
    "ParallelMode",
    "RootLpPolicy",
    "SearchProfile",
    "Sense",
    "Status",
    "VarType",
    "read_lp",
    "read_model",
    "read_mps",
    "solve_file",
    "solve_mps",
    "write_mps",
]
