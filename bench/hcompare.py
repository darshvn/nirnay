"""Cross-check the MPS reader against HiGHS's reader (comparison only; not part of the solver)."""
import gzip, shutil, sys, tempfile
from pathlib import Path
import numpy as np, highspy
from nirnay.io.mps import read_mps

def highs_lp(path):
    tmp = Path(tempfile.gettempdir()) / (Path(path).name.replace(".gz", ""))
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as a, open(tmp, "wb") as b:
            shutil.copyfileobj(a, b)
    else:
        tmp = Path(path)
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.readModel(str(tmp))
    return h

if __name__ == "__main__":
    for f in sorted(Path("data").rglob("*.mps.gz")):
        m = read_mps(f)
        lp = highs_lp(f).getLp()
        checks = {
            "dims": m.n == lp.num_col_ and m.m == lp.num_row_,
            "cost": np.allclose(m.sense * m.c, lp.col_cost_),
            "colbounds": np.array_equal(m.lb, lp.col_lower_) and np.array_equal(m.ub, lp.col_upper_),
            "rowbounds": np.array_equal(m.rl, lp.row_lower_) and np.array_equal(m.ru, lp.row_upper_),
            "nnz": m.A.nnz == len(lp.a_matrix_.value_),
            "integer": np.array_equal(m.integer, np.array([t != highspy.HighsVarType.kContinuous for t in lp.integrality_], dtype=bool)) if len(lp.integrality_) else not m.is_mip,
        }
        bad = [k for k, v in checks.items() if not v]
        print(f"{'MATCH' if not bad else 'DIFF ' + ','.join(bad):<22} {m.summary()}")
