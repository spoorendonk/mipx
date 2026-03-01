#!/usr/bin/env python3
"""Minimal HiGHS CLI-compatible wrapper backed by Python highspy.

This wrapper supports the subset of CLI flags used by `run_presolve_compare.py`:
- positional model path
- --options_file
- --solver
- --presolve
- --parallel
- --time_limit

It prints key lines in HiGHS-like format so existing parsers can consume:
- Model status
- Objective / primal bound
- Iteration and node counts
- HiGHS run time
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import tempfile
import time
from pathlib import Path

import highspy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("model")
    p.add_argument("--options_file", default="")
    p.add_argument("--solver", default="choose")
    p.add_argument("--presolve", default="choose")
    p.add_argument("--parallel", default="off")
    p.add_argument("--time_limit", type=float, default=None)
    args, _ = p.parse_known_args()
    return args


def parse_options_file(path: str) -> dict[str, str]:
    opts: dict[str, str] = {}
    if not path:
        return opts
    p = Path(path)
    if not p.is_file():
        return opts
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        opts[key.strip()] = value.strip()
    return opts


def model_path_for_highspy(model: Path) -> tuple[Path, Path | None]:
    if model.suffix != ".gz":
        return model, None
    fd, tmp_name = tempfile.mkstemp(suffix=".mps")
    os.close(fd)
    tmp_path = Path(tmp_name)
    with gzip.open(model, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return tmp_path, tmp_path


def status_text(status: highspy.HighsModelStatus) -> str:
    mapping = {
        highspy.HighsModelStatus.kOptimal: "Optimal",
        highspy.HighsModelStatus.kInfeasible: "Infeasible",
        highspy.HighsModelStatus.kUnbounded: "Unbounded",
        highspy.HighsModelStatus.kUnboundedOrInfeasible: "Unbounded or infeasible",
        highspy.HighsModelStatus.kTimeLimit: "Time limit",
        highspy.HighsModelStatus.kIterationLimit: "Iteration limit",
        highspy.HighsModelStatus.kSolutionLimit: "Solution limit",
        highspy.HighsModelStatus.kObjectiveBound: "Objective bound",
        highspy.HighsModelStatus.kObjectiveTarget: "Objective target",
        highspy.HighsModelStatus.kInterrupt: "Interrupt",
        highspy.HighsModelStatus.kHighsInterrupt: "Interrupt",
        highspy.HighsModelStatus.kModelEmpty: "Model empty",
        highspy.HighsModelStatus.kNotset: "Not set",
        highspy.HighsModelStatus.kLoadError: "Load error",
        highspy.HighsModelStatus.kPresolveError: "Presolve error",
        highspy.HighsModelStatus.kSolveError: "Solve error",
        highspy.HighsModelStatus.kPostsolveError: "Postsolve error",
        highspy.HighsModelStatus.kModelError: "Model error",
        highspy.HighsModelStatus.kMemoryLimit: "Memory limit",
    }
    return mapping.get(status, str(status))


def main() -> int:
    args = parse_args()
    model = Path(args.model)
    if not model.is_file():
        print("Model status : Load error")
        return 1

    file_opts = parse_options_file(args.options_file)
    threads = int(file_opts.get("threads", "1"))

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("solver", args.solver)
    highs.setOptionValue("presolve", args.presolve)
    highs.setOptionValue("parallel", args.parallel)
    highs.setOptionValue("threads", max(1, threads))
    if args.time_limit is not None and args.time_limit > 0:
        highs.setOptionValue("time_limit", float(args.time_limit))

    actual_model, temp_model = model_path_for_highspy(model)
    try:
        read_status = highs.readModel(str(actual_model))
        if read_status != highspy.HighsStatus.kOk:
            print("Model status : Load error")
            return 1

        lp = highs.getLp()
        is_mip = len(lp.integrality_) > 0

        t0 = time.perf_counter()
        run_status = highs.run()
        elapsed = time.perf_counter() - t0
        if run_status != highspy.HighsStatus.kOk:
            print("Model status : Solve error")
            return 1

        model_status = highs.getModelStatus()
        info = highs.getInfo()
        objective = highs.getObjectiveValue()

        print(f"Model status : {status_text(model_status)}")
        if is_mip:
            nodes = int(max(0, info.mip_node_count))
            lp_iters = int(max(0, info.simplex_iteration_count))
            print(f"Primal bound      {objective:.12g}")
            print(f"Nodes             {nodes}")
            print(f"LP iterations     {lp_iters}")
        else:
            simplex_iters = int(max(0, info.simplex_iteration_count))
            print(f"Objective value : {objective:.12g}")
            print(f"Simplex iterations: {simplex_iters}")
        print(f"HiGHS run time : {elapsed:.6f}")
        return 0
    finally:
        if temp_model is not None:
            try:
                temp_model.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
