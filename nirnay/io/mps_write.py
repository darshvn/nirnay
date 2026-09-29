"""Free-format MPS writer — the inverse of `nirnay.io.mps.read_mps`.

    write_mps(model, "out.mps")        # or "out.mps.gz"

What is written:
  * OBJSENSE MAX when the model was a maximisation (Model.sense == -1). The objective, its
    constant and Q are then written in the user's (maximisation) sense, as the reader expects.
  * ROWS: E for rl == ru, L for (-inf, ru], G for [rl, +inf), and for a ranged row [rl, ru] a G or
    L row plus a RANGES entry. The side and range are chosen so that the reader's arithmetic
    (rl + R or ru - R) reproduces the other side bit for bit.
  * COLUMNS with 'MARKER' 'INTORG' / 'INTEND' around every run of integer columns. A column with
    no objective coefficient and no matrix entry is written with an explicit zero objective entry
    so that it is not lost.
  * RHS, including the objective constant as minus the RHS of the objective row.
  * BOUNDS using FR, MI, PL, BV, LO, UP, FX. Integer columns always get explicit bounds, so that no
    reader has to apply a default for integer columns without bounds.
  * QUADOBJ with the lower triangle of Q (the reader mirrors it back to the full matrix).
  * Numbers use Python's shortest round-trip representation (repr), so reading the file back
    gives the same doubles.

A row that is free on both sides cannot be written as an E/L/G row; it is written as an extra N
row, which readers (this one and HiGHS) drop. Names containing whitespace, empty names, names
starting with '*' (a comment marker) and duplicate names are replaced by generated names R<i> and
C<j> for the whole row or column set.
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path

import numpy as np

from ..model import Model


def _num(v: float) -> str:
    v = float(v)
    if v == int(v) and abs(v) < 1e15:
        s = str(int(v))
        return "-0" if (v == 0.0 and math.copysign(1.0, v) < 0) else s
    return repr(v)


def _names(names, count: int, prefix: str, reserved: set) -> list[str]:
    ok = len(names) == count and len(set(names)) == count and all(
        isinstance(s, str) and s and not s.startswith("*") and not any(ch.isspace() for ch in s)
        and s not in reserved for s in names)
    if ok:
        return list(names)
    width = len(str(max(count - 1, 0)))
    return [f"{prefix}{k:0{width}d}" for k in range(count)]


def _range_for(rl: float, ru: float) -> tuple[str, float, float]:
    """(row type, rhs, range) such that the reader rebuilds exactly [rl, ru]."""
    R = ru - rl
    # G row: reader sets ru = rl + |R|.  L row: reader sets rl = ru - |R|.
    cands = [R, np.nextafter(R, np.inf), np.nextafter(R, -np.inf)]
    for r in cands:
        if r > 0 and rl + r == ru:
            return "G", rl, float(r)
    for r in cands:
        if r > 0 and ru - r == rl:
            return "L", ru, float(r)
    # widen the search a little before giving up exactness
    r = R
    for _ in range(64):
        r = np.nextafter(r, np.inf)
        if rl + r == ru:
            return "G", rl, float(r)
    return "G", rl, float(R)


def write_mps(model: Model, path, name: str | None = None) -> None:
    path = Path(path)
    m, n = model.m, model.n
    sense = model.sense
    # file values are in the user's sense
    c_file = sense * np.asarray(model.c, dtype=float)
    c0_file = sense * float(model.c0)

    row_names = _names(model.row_names, m, "R", set())
    col_names = _names(model.col_names, n, "C", set())
    obj_name = "OBJ"
    taken = set(row_names)
    k = 0
    while obj_name in taken:
        k += 1
        obj_name = f"OBJ{k}"

    rl, ru = np.asarray(model.rl, float), np.asarray(model.ru, float)
    lb, ub = np.asarray(model.lb, float), np.asarray(model.ub, float)
    integer = np.asarray(model.integer, bool)

    out: list[str] = []
    w = out.append
    title = (name or model.name or "NIRNAY").split()
    w(f"NAME {title[0] if title else 'NIRNAY'}")
    if sense == -1:
        w("OBJSENSE")
        w("    MAX")

    # ---- ROWS ------------------------------------------------------------------------------
    w("ROWS")
    w(f" N  {obj_name}")
    rhs: list[tuple[str, float]] = []
    ranges: list[tuple[str, float]] = []
    for i in range(m):
        lo, up, rn = rl[i], ru[i], row_names[i]
        if lo == up:
            w(f" E  {rn}")
            if lo != 0.0:
                rhs.append((rn, lo))
        elif np.isinf(lo) and np.isinf(up):
            w(f" N  {rn}")                         # free row: dropped by readers
        elif np.isinf(lo):
            w(f" L  {rn}")
            if up != 0.0:
                rhs.append((rn, up))
        elif np.isinf(up):
            w(f" G  {rn}")
            if lo != 0.0:
                rhs.append((rn, lo))
        else:
            kind, b, r = _range_for(lo, up)
            w(f" {kind}  {rn}")
            if b != 0.0:
                rhs.append((rn, b))
            ranges.append((rn, r))

    # ---- COLUMNS ---------------------------------------------------------------------------
    w("COLUMNS")
    A = model.A
    colptr, rowidx, vals = A.colptr, A.rowidx, A.vals
    in_int = False
    marker = 0
    for j in range(n):
        if integer[j] and not in_int:
            w(f"    MARKER{marker:04d}  'MARKER'  'INTORG'")
            marker += 1
            in_int = True
        elif not integer[j] and in_int:
            w(f"    MARKER{marker:04d}  'MARKER'  'INTEND'")
            marker += 1
            in_int = False
        cn = col_names[j]
        wrote = False
        if c_file[j] != 0.0:
            w(f"    {cn}  {obj_name}  {_num(c_file[j])}")
            wrote = True
        for p in range(colptr[j], colptr[j + 1]):
            if vals[p] != 0.0:
                w(f"    {cn}  {row_names[rowidx[p]]}  {_num(vals[p])}")
                wrote = True
        if not wrote:
            w(f"    {cn}  {obj_name}  0")
    if in_int:
        w(f"    MARKER{marker:04d}  'MARKER'  'INTEND'")

    # ---- RHS -------------------------------------------------------------------------------
    if rhs or c0_file != 0.0:
        w("RHS")
        if c0_file != 0.0:
            w(f"    RHS  {obj_name}  {_num(-c0_file)}")
        for rn, v in rhs:
            w(f"    RHS  {rn}  {_num(v)}")

    # ---- RANGES ----------------------------------------------------------------------------
    if ranges:
        w("RANGES")
        for rn, v in ranges:
            w(f"    RNG  {rn}  {_num(v)}")

    # ---- BOUNDS ----------------------------------------------------------------------------
    bnd: list[str] = []
    b = bnd.append
    for j in range(n):
        lo, up, cn = lb[j], ub[j], col_names[j]
        if integer[j] and lo == 0.0 and up == 1.0:
            b(f" BV BND  {cn}")
            continue
        if lo == up:
            b(f" FX BND  {cn}  {_num(lo)}")
            continue
        if np.isinf(lo) and np.isinf(up):
            b(f" FR BND  {cn}")
            continue
        if np.isinf(lo):
            b(f" MI BND  {cn}")
        elif lo != 0.0 or (math.copysign(1.0, lo) < 0):
            b(f" LO BND  {cn}  {_num(lo)}")
        if np.isinf(up):
            if integer[j]:
                b(f" PL BND  {cn}")
        else:
            b(f" UP BND  {cn}  {_num(up)}")
    if bnd:
        w("BOUNDS")
        out.extend(bnd)

    # ---- QUADOBJ ---------------------------------------------------------------------------
    if model.is_qp:
        Q = model.Q
        qcols = np.repeat(np.arange(Q.n, dtype=np.int64), np.diff(Q.colptr))
        lower = Q.rowidx >= qcols                       # (row i, col j) with i >= j
        if not lower.any():
            raise ValueError("Q has no lower-triangle entries; store Q in full (both triangles)")
        w("QUADOBJ")
        for i, j, v in zip(Q.rowidx[lower], qcols[lower], Q.vals[lower]):
            if v != 0.0:
                w(f"    {col_names[j]}  {col_names[i]}  {_num(sense * v)}")

    w("ENDATA")
    text = "\n".join(out) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "wt", newline="\n") as fh:
            fh.write(text)
    else:
        with open(path, "w", newline="\n") as fh:
            fh.write(text)
