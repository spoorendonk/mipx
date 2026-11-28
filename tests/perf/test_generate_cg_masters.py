"""Tests for the v1 barrier-cg source and master recipe."""

from __future__ import annotations

import hashlib
import io
import tarfile
from typing import TYPE_CHECKING

import generate_cg_masters as gen
import pytest

if TYPE_CHECKING:
    from pathlib import Path

SOURCE = b"4 5 1\n1 3 1 8\n1 2 1 4\n3 4 2 4\n2 4 2 6\n1 2 1 9\n1 4 7\n"
MASTER = """NAME master
ROWS
 N obj
 G d0
 L a1
 L a3
COLUMNS
 u0 obj 7
 u0 d0 1
 p0 obj 3
 p0 d0 1
 p0 a1 1
 p0 a3 1
RHS
 rhs d0 7
 rhs a1 4
 rhs a3 6
ENDATA
"""


def test_later_master_prices_one_coupled_path_with_deterministic_ties() -> None:
    assert gen.master_mps(SOURCE) == MASTER.encode("ascii")


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"2 0 1\n1 2 5\n", "node 2 is unreachable from node 1"),
        (b"1 0 1\n1 1 5\n", "no path linking demand and capacity rows"),
    ],
)
def test_later_master_requires_coupled_paths(data: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gen.master_mps(data)


def test_panel_checks_archives_and_copies_matching_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "grid.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("grid1")
        member.size = len(SOURCE)
        tar.addfile(member, io.BytesIO(SOURCE))

    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksums = tmp_path / "checksums"
    checksums.write_text(f"{checksum}  grid.tgz\n", encoding="ascii")
    references = tmp_path / "v1.solu"
    references.write_text("=opt= grid1 33\n", encoding="ascii")
    masters = tmp_path / "v1.sha256"
    checksum = hashlib.sha256(MASTER.encode("ascii")).hexdigest()
    masters.write_text(f"{checksum}  grid1.mps\n", encoding="ascii")
    monkeypatch.setattr(gen, "CHECKSUMS", checksums)
    monkeypatch.setattr(gen, "REFERENCES", references)
    monkeypatch.setattr(gen, "MASTER_CHECKSUMS", masters)
    monkeypatch.setattr(gen, "CASES", {"grid.tgz": ["grid1"]})

    dest = tmp_path / "generated"
    gen.generate_panel(tmp_path, dest)
    assert (dest / "grid1.mps").read_text(encoding="ascii") == MASTER
    assert (dest / gen.SOLU).read_bytes() == references.read_bytes()

    rejected = tmp_path / "rejected"
    masters.write_text(f"{'0' * 64}  grid1.mps\n", encoding="ascii")
    with pytest.raises(ValueError, match=r"generated grid1\.mps differs from the pinned v1 model"):
        gen.generate_panel(tmp_path, rejected)
    assert not rejected.exists()

    monkeypatch.setattr(gen, "CASES", {"grid.tgz": ["grid2"]})
    with pytest.raises(ValueError, match=r"grid2 is missing from grid\.tgz"):
        gen.generate_panel(tmp_path, rejected)
    assert not rejected.exists()

    archive.write_bytes(b"corrupt archive")
    with pytest.raises(ValueError, match=r"incorrect SHA-256.*grid\.tgz"):
        gen.generate_panel(tmp_path, rejected)
    assert not rejected.exists()
