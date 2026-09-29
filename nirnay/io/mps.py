"""MPS and QPS reader — the format every public benchmark library ships in.

Handles fixed and free MPS, gzip, OBJSENSE, integer MARKER blocks, RANGES, every bound type in
common use (UP LO FX FR MI PL BV LI UI), an RHS entry on the objective row (a constant), and the
QP sections QUADOBJ (one triangle), QMATRIX and QSECTION (both triangles).

Conventions follow the ones the benchmark libraries assume, so a known optimal value in their
solution files is the optimal value here:
  * a variable inside an integer MARKER block with no bounds gets [0, +inf)
  * UP with a negative value on a variable whose lower bound is still 0 sets the lower bound to -inf
  * RANGES on E rows extend upward for R > 0 and downward for R < 0
  * the objective constant is minus the RHS given on the objective row
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from ..model import CSC, INF, Model

SECTIONS = {"NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS", "SOS", "ENDATA", "OBJSENSE",
            "OBJSENS", "QUADOBJ", "QMATRIX", "QSECTION", "OBJNAME"}


class MPSError(ValueError):
    pass


def _open(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "rt", errors="replace")


def _fixed_fields(line: str) -> list[str]:
    """Split a fixed-format MPS data line by column position (names may contain spaces)."""
    spans = [(1, 3), (4, 12), (14, 22), (24, 36), (39, 47), (49, 61)]
    out = [line[a:b].strip() for a, b in spans if a < len(line)]
    return [f for f in out if f != ""] if out else []


def read_mps(path, name: str | None = None) -> Model:
    path = Path(path)
    row_index: dict[str, int] = {}
    row_type: list[str] = []
    row_names: list[str] = []
    obj_row = None
    extra_n_rows: set[str] = set()

    col_index: dict[str, int] = {}
    col_names: list[str] = []
    integer_flags: list[bool] = []
    obj: dict[int, float] = {}
    trip_r: list[int] = []
    trip_c: list[int] = []
    trip_v: list[float] = []

    rhs: dict[int, float] = {}
    rng: dict[int, float] = {}
    obj_constant = 0.0
    bounds: dict[int, list] = {}       # col -> [lb or None, ub or None]
    q_r: list[int] = []
    q_c: list[int] = []
    q_v: list[float] = []
    sense = 1
    model_name = name or path.name.split(".")[0]

    section = None
    in_integer = False
    fixed_format = False      # set when a name turns out to contain spaces: parse by column
    free_rhs_name = None     # the first RHS / RANGES / BOUNDS vector name; later ones are ignored
    free_rng_name = None
    free_bnd_name = None

    def col_of(cname: str) -> int:
        j = col_index.get(cname)
        if j is None:
            j = len(col_names)
            col_index[cname] = j
            col_names.append(cname)
            integer_flags.append(in_integer)
        return j

    with _open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n\r")
            if not line.strip() or line.lstrip().startswith("*"):
                continue
            if not line[0].isspace():
                head = line.split()
                key = head[0].upper()
                if key in SECTIONS:
                    section = key
                    if key == "NAME" and len(head) > 1 and name is None:
                        model_name = head[1]
                    elif key in ("OBJSENSE", "OBJSENS") and len(head) > 1:
                        sense = -1 if head[1].upper().startswith("MAX") else 1
                    if key == "ENDATA":
                        break
                    continue
                # some writers put data lines at column 0 in free format; fall through
            tokens = line.split()
            if not tokens:
                continue
            if fixed_format and section not in ("ROWS", "OBJSENSE", "OBJSENS"):
                tokens = _fixed_fields(line)

            if section in ("OBJSENSE", "OBJSENS"):
                sense = -1 if tokens[0].upper().startswith("MAX") else 1

            elif section == "ROWS":
                if len(tokens) != 2:
                    # fixed MPS allows spaces inside names: type in cols 2-3, name in cols 5-12
                    fixed_format = True
                    tokens = [line[1:3].strip(), line[4:12].strip()]
                kind, rname = tokens[0].upper(), tokens[1]
                if kind == "N":
                    if obj_row is None:
                        obj_row = rname
                    else:
                        extra_n_rows.add(rname)     # free rows beyond the objective are dropped
                    continue
                if kind not in ("E", "L", "G"):
                    raise MPSError(f"{path.name}: unknown row type {kind!r}")
                row_index[rname] = len(row_names)
                row_names.append(rname)
                row_type.append(kind)

            elif section == "COLUMNS":
                if len(tokens) >= 3 and tokens[1].strip("'\"").upper() == "MARKER":
                    tag = tokens[2].strip("'\"").upper()
                    if tag == "INTORG":
                        in_integer = True
                    elif tag == "INTEND":
                        in_integer = False
                    continue
                if len(tokens) not in (3, 5):
                    tokens = _fixed_fields(line)
                    if len(tokens) not in (3, 5):
                        raise MPSError(f"{path.name}: cannot parse COLUMNS line: {line!r}")
                j = col_of(tokens[0])
                for rname, sval in zip(tokens[1::2], tokens[2::2]):
                    v = float(sval)
                    if rname == obj_row:
                        obj[j] = obj.get(j, 0.0) + v
                    elif rname in extra_n_rows:
                        continue
                    else:
                        i = row_index.get(rname)
                        if i is None:
                            raise MPSError(f"{path.name}: column {tokens[0]} refers to unknown row {rname}")
                        trip_r.append(i)
                        trip_c.append(j)
                        trip_v.append(v)

            elif section in ("RHS", "RANGES"):
                # "name row value [row value]" or, without a vector name, "row value [row value]"
                if len(tokens) % 2 == 1:
                    vec, pairs = tokens[0], tokens[1:]
                else:
                    vec, pairs = "", tokens
                if section == "RHS":
                    if free_rhs_name is None:
                        free_rhs_name = vec
                    elif vec != free_rhs_name:
                        continue
                else:
                    if free_rng_name is None:
                        free_rng_name = vec
                    elif vec != free_rng_name:
                        continue
                for rname, sval in zip(pairs[0::2], pairs[1::2]):
                    v = float(sval)
                    if rname == obj_row:
                        if section == "RHS":
                            obj_constant = -v
                        continue
                    if rname in extra_n_rows:
                        continue
                    i = row_index.get(rname)
                    if i is None:
                        raise MPSError(f"{path.name}: {section} refers to unknown row {rname}")
                    (rhs if section == "RHS" else rng)[i] = v

            elif section == "BOUNDS":
                btype = tokens[0].upper()
                if btype in ("FR", "MI", "PL", "BV") and len(tokens) in (2, 3):
                    # value is optional for these; vector name may be missing
                    if len(tokens) == 3 and btype != "BV":
                        vec, cname, sval = tokens[1], tokens[2], None
                    elif len(tokens) == 3:
                        vec, cname, sval = tokens[1], tokens[2], None
                    else:
                        vec, cname, sval = "", tokens[1], None
                    if len(tokens) == 3 and btype == "BV":
                        # "BV BND x" or "BV x 1"
                        if tokens[2] in col_index or tokens[1] not in col_index:
                            vec, cname = tokens[1], tokens[2]
                        else:
                            vec, cname = "", tokens[1]
                elif len(tokens) == 4:
                    vec, cname, sval = tokens[1], tokens[2], tokens[3]
                elif len(tokens) == 3:
                    vec, cname, sval = "", tokens[1], tokens[2]
                else:
                    raise MPSError(f"{path.name}: cannot parse BOUNDS line: {line!r}")
                if free_bnd_name is None:
                    free_bnd_name = vec
                elif vec != free_bnd_name:
                    continue
                j = col_index.get(cname)
                if j is None:
                    raise MPSError(f"{path.name}: bound on unknown column {cname}")
                b = bounds.setdefault(j, [None, None])
                v = float(sval) if sval is not None else None
                if btype == "UP":
                    b[1] = v
                    if v < 0 and b[0] is None:
                        b[0] = -INF
                elif btype == "LO":
                    b[0] = v
                elif btype == "FX":
                    b[0] = b[1] = v
                elif btype == "FR":
                    b[0], b[1] = -INF, INF
                elif btype == "MI":
                    b[0] = -INF
                elif btype == "PL":
                    b[1] = INF
                elif btype == "BV":
                    b[0], b[1] = 0.0, 1.0
                    integer_flags[j] = True
                elif btype == "LI":
                    b[0] = v
                    integer_flags[j] = True
                elif btype == "UI":
                    b[1] = v
                    integer_flags[j] = True
                    if v < 0 and b[0] is None:
                        b[0] = -INF
                elif btype == "SC":
                    raise MPSError(f"{path.name}: semi-continuous bounds (SC) are not supported yet")
                else:
                    raise MPSError(f"{path.name}: unknown bound type {btype}")

            elif section in ("QUADOBJ", "QMATRIX", "QSECTION"):
                if len(tokens) < 3:
                    continue
                a, b_ = col_index.get(tokens[0]), col_index.get(tokens[1])
                if a is None or b_ is None:
                    raise MPSError(f"{path.name}: {section} refers to unknown column")
                v = float(tokens[2])
                q_r.append(a)
                q_c.append(b_)
                q_v.append(v)
                if section == "QUADOBJ" and a != b_:
                    q_r.append(b_)      # QUADOBJ gives one triangle; store the full matrix
                    q_c.append(a)
                    q_v.append(v)

            elif section == "SOS":
                raise MPSError(f"{path.name}: SOS constraints are not supported yet")

    m, n = len(row_names), len(col_names)
    c = np.zeros(n)
    for j, v in obj.items():
        c[j] = v
    A = CSC.from_triplets(m, n, trip_r, trip_c, trip_v)

    rl = np.full(m, -INF)
    ru = np.full(m, INF)
    for i, kind in enumerate(row_type):
        b = rhs.get(i, 0.0)
        if kind == "E":
            rl[i] = ru[i] = b
        elif kind == "L":
            ru[i] = b
        else:
            rl[i] = b
        if i in rng:
            r = rng[i]
            if kind == "E":
                if r > 0:
                    ru[i] = b + abs(r)
                elif r < 0:
                    rl[i] = b - abs(r)
            elif kind == "L":
                rl[i] = b - abs(r)
            else:
                ru[i] = b + abs(r)

    integer = np.array(integer_flags, dtype=bool) if n else np.zeros(0, dtype=bool)
    lb = np.zeros(n)
    ub = np.full(n, INF)
    for j, (lo, up) in bounds.items():
        if lo is not None:
            lb[j] = lo
        if up is not None:
            ub[j] = up

    Q = None
    if q_v:
        Q = CSC.from_triplets(n, n, q_r, q_c, q_v)

    if sense == -1:           # store as a minimisation; Model.sense restores the user's view
        c = -c
        obj_constant = -obj_constant
        if Q is not None:
            Q.vals = -Q.vals

    model = Model(name=model_name, c=c, A=A, rl=rl, ru=ru, lb=lb, ub=ub, integer=integer, Q=Q,
                  c0=obj_constant, sense=sense, col_names=col_names, row_names=row_names)
    model.validate()
    return model
