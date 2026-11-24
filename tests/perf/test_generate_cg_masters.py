"""Tests for selecting a later CG master and its recorded objective (#358)."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parent / "generate_cg_masters.py"


def load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_cg_masters", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_cg_masters"] = module
    spec.loader.exec_module(module)
    return module


gen = load_generator()


def load_gate() -> ModuleType:
    """run_instance_gate.py, which reads the .solu this writes."""
    path = SCRIPT.parent / "run_instance_gate.py"
    spec = importlib.util.spec_from_file_location("run_instance_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_instance_gate"] = module
    spec.loader.exec_module(module)
    return module


MPS_DIAGONAL = (
    "NAME m\nROWS\nN obj\nG r0\nL r1\nCOLUMNS\nc0 obj 1\nc0 r0 1\nc0 r1 0\n"
    "c1 obj 2\nc1 r1 1\nRHS\nrhs r0 1\nrhs r1 3\nENDATA\n"
)
MPS_COUPLED = MPS_DIAGONAL.replace("c0 r1 0", "c0 r1 1")
MPS_FREE_ROW = MPS_COUPLED.replace("N obj\n", "N obj\nN free\n")


def test_capacity_and_demand_must_share_a_nonzero_column(tmp_path: Path) -> None:
    mps = tmp_path / "master.mps"
    mps.write_text(MPS_DIAGONAL, encoding="utf-8")
    assert not gen.has_capacity_path(mps)
    mps.write_text(MPS_COUPLED, encoding="utf-8")
    assert gen.has_capacity_path(mps)


def test_selects_first_coupled_later_snapshot_and_its_own_solve(tmp_path: Path) -> None:
    ops = tmp_path / "ops.txt"
    ops.write_text(
        "solve 0 optimal 9 3 0.1\ncol_costs 1 0 8\nsolve 1 optimal 8 4 0.2\n"
        "coeffs 1 1 0 1\nsolve 2 optimal 5 2 0.3\nsolve 3 optimal 4 2 0.4\n",
        encoding="utf-8",
    )
    for seq, mps in enumerate([MPS_DIAGONAL, MPS_DIAGONAL, MPS_COUPLED, MPS_COUPLED]):
        (tmp_path / f"master_{seq:04d}.mps").write_text(mps, encoding="utf-8")
    assert gen.later_master(tmp_path) == (tmp_path / "master_0002.mps", 5)


@pytest.mark.parametrize("objective", ["nan", "inf", "-inf"])
def test_later_master_requires_a_finite_reference(tmp_path: Path, objective: str) -> None:
    (tmp_path / "ops.txt").write_text(
        f"solve 0 optimal 9 3 0.1\nsolve 1 optimal {objective} 4 0.2\n", encoding="utf-8"
    )
    (tmp_path / "master_0001.mps").write_text(MPS_COUPLED, encoding="utf-8")
    with pytest.raises(ValueError, match="finite optimal objective"):
        gen.later_master(tmp_path)


def test_read_manifest_splits_on_the_first_equals(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("sense = min\nobj_offset=0\nnote=a=b\nno separator\n", encoding="utf-8")
    assert gen.read_manifest(manifest) == {"sense": "min", "obj_offset": "0", "note": "a=b"}


# A stand-in for mcf: writes the dump that the instance's name asks for, or a
# good one for every instance when FAKE_MCF_GOOD is set.
FAKE_MCF = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

source = Path(sys.argv[1]).name
if os.environ.get("FAKE_MCF_GOOD"):
    source = "good"
dump = Path(next(a for a in sys.argv if a.startswith("lp_dump=")).split("=", 1)[1])
if source.startswith("grid2"):
    sys.exit(3)  # no dump at all
lp = dump / "lp_0000"
lp.mkdir(parents=True)
fmt = "mcf-lpseq 2" if source.startswith("grid8") else "mcf-lpseq 1"
sense = "max" if source.startswith("grid3") else "min"
offset = "1.5" if source.startswith("grid4") else "0"
if not source.startswith("grid9"):
    (lp / "manifest.txt").write_text(f"format = {fmt}\\nsense={sense}\\nobj_offset={offset}\\n")
assert "lp_dump_every=1" in sys.argv
(lp / "master_0000.mps").write_text(@MPS_DIAGONAL@)
mps = @MPS_FREE_ROW@ if source.startswith("grid5") else @MPS_COUPLED@
if source.startswith("planar30.gz"):
    mps = @MPS_DIAGONAL@
(lp / "master_0001.mps").write_text(mps)
status = "infeasible" if source.startswith("grid6") else "optimal"
(lp / "ops.txt").write_text(f"solve 0 optimal 55 4 0.1\\nsolve 1 {status} 42.25 7 0.2\\n")
sys.exit(1)  # the CG run may hit the iteration limit
"""

REFUSED = {
    "grid2": "wrote no master",
    "grid3": "sense or offset",
    "grid4": "sense or offset",
    "grid5": "2 N rows",
    "grid6": "infeasible",
    "grid7": "no instance file",
    "grid8": "unsupported dump format",
    "grid9": "manifest.txt",
    "planar30": "no later master",
}


def fake_mcf(tmp_path: Path) -> tuple[Path, Path]:
    """The fake mcf and an instance directory without grid7's file."""
    mcf = tmp_path / "mcf"
    script = (
        FAKE_MCF.replace("@MPS_DIAGONAL@", repr(MPS_DIAGONAL))
        .replace("@MPS_COUPLED@", repr(MPS_COUPLED))
        .replace("@MPS_FREE_ROW@", repr(MPS_FREE_ROW))
    )
    mcf.write_text(script, encoding="utf-8")
    mcf.chmod(mcf.stat().st_mode | stat.S_IXUSR)
    instances = tmp_path / "commalab"
    for name in gen.INSTANCES:
        if name == "grid7":
            continue
        path = gen.instance_file(instances, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return mcf, instances


def test_generate_refuses_each_bad_master_and_then_copies_none(tmp_path: Path) -> None:
    mcf, instances = fake_mcf(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    work = tmp_path / "work"
    work.mkdir()

    solu, failures = gen.generate(mcf, instances, out, work)

    checked = [n for n in gen.INSTANCES if n not in REFUSED]
    assert solu == [f"=opt= {n} 42.25" for n in checked]
    # New masters beside an old .solu would read as a barrier regression.
    assert list(out.glob("*.mps")) == []
    assert len(failures) == len(REFUSED)
    for name, needle in REFUSED.items():
        assert any(f.startswith(f"{name}: ") and needle in f for f in failures), (name, failures)


def test_generate_reports_a_missing_mcf(tmp_path: Path) -> None:
    _, instances = fake_mcf(tmp_path)
    missing = tmp_path / "no-mcf"
    solu, failures = gen.generate(missing, instances, tmp_path, tmp_path)
    assert solu == []
    # Every instance but grid7, which has no file, fails on the missing mcf.
    on_mcf = [f for f in failures if str(missing) in f]
    assert len(on_mcf) == len(gen.INSTANCES) - 1
    assert len(failures) == len(gen.INSTANCES)


def test_main_writes_a_solu_the_gate_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mcf, instances = fake_mcf(tmp_path)
    gen.instance_file(instances, "grid7").write_bytes(b"")
    monkeypatch.setenv("FAKE_MCF_GOOD", "1")
    out = tmp_path / "out"
    assert gen.main(["--mcf", str(mcf), "--instances", str(instances), "--out", str(out)]) == 0
    refs = load_gate().read_solu(out / gen.SOLU)
    assert sorted(refs) == sorted(gen.INSTANCES)
    assert all(ref.kind == "opt" and ref.value == 42.25 for ref in refs.values())
    assert sorted(p.stem for p in out.glob("*.mps")) == sorted(gen.INSTANCES)
    assert all(p.read_text(encoding="utf-8") == MPS_COUPLED for p in out.glob("*.mps"))
    assert [p.name for p in out.iterdir() if p.name.startswith(".dump_")] == []


def test_main_changes_nothing_when_a_master_fails(tmp_path: Path) -> None:
    mcf, instances = fake_mcf(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / gen.SOLU).write_text("=opt= grid1 1.0\n", encoding="utf-8")
    (out / "grid1.mps").write_text("old\n", encoding="utf-8")
    assert gen.main(["--mcf", str(mcf), "--instances", str(instances), "--out", str(out)]) == 1
    assert (out / gen.SOLU).read_text(encoding="utf-8") == "=opt= grid1 1.0\n"
    assert (out / "grid1.mps").read_text(encoding="utf-8") == "old\n"
    assert sorted(p.name for p in out.iterdir()) == ["cg_masters.solu", "grid1.mps"]
