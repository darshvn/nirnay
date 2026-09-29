"""Which presolve reductions actually fire on our benchmark sets? (comparator analysis only)

Runs HiGHS presolve alone (presolve_rule_logging on) over every instance in data/netlib and
data/miplib, parses HiGHS's per-rule table and prints totals of rows/cols removed per rule.
    python tools/presolve_census.py [--out census.csv]
"""
import argparse
import collections
import csv
import io
import os
import re
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import comparators  # noqa: E402

ROW = re.compile(r"^(?P<rule>[A-Za-z][A-Za-z ()/-]*?)\s{2,}(?P<rows>\d+)\s+(?P<cols>\d+)\s+(?P<calls>\d+)\s*$")


def census(path):
    import highspy
    h = highspy.Highs()
    log = Path(tempfile.gettempdir()) / "nirnay_comparators" / "presolve_census.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        log.unlink()
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", str(log))
    h.setOptionValue("presolve_rule_logging", True)
    h.setOptionValue("time_limit", 60.0)
    h.readModel(comparators.plain_mps(path))
    lp = h.getLp()
    m0, n0 = lp.num_row_, lp.num_col_
    h.presolve()
    pl = h.getPresolvedLp()
    h.setOptionValue("log_file", "")
    text = log.read_text(errors="replace") if log.exists() else ""
    rules = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("Presolve rule removed"):
            in_table = True
            continue
        if in_table and line.startswith("Total reductions"):
            in_table = False
            continue
        if in_table:
            mm = ROW.match(line.strip())
            if mm:
                rules[mm["rule"].strip()] = (int(mm["rows"]), int(mm["cols"]), int(mm["calls"]))
    return m0, n0, pl.num_row_, pl.num_col_, rules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    per_set = {}
    rows_out = []
    for sub in ("netlib", "miplib"):
        tot = collections.defaultdict(lambda: [0, 0, 0, 0])   # rows, cols, calls, instances fired
        m_sum = n_sum = m_after = n_after = 0
        for f in sorted((repo / "data" / sub).glob("*.mps.gz")):
            try:
                m0, n0, m1, n1, rules = census(f)
            except Exception as e:  # keep going
                print(f"{f.name}: {e}", file=sys.stderr)
                continue
            m_sum += m0; n_sum += n0; m_after += m1; n_after += n1
            for r, (dr, dc, calls) in rules.items():
                t = tot[r]
                t[0] += dr; t[1] += dc; t[2] += calls
                t[3] += 1 if (dr or dc) else 0
                rows_out.append(dict(set=sub, instance=f.name.split(".")[0], rule=r, rows=dr, cols=dc, calls=calls))
        per_set[sub] = (tot, m_sum, n_sum, m_after, n_after)
    for sub, (tot, m_sum, n_sum, m_after, n_after) in per_set.items():
        print(f"\n== {sub}: rows {m_sum} -> {m_after} ({100*(1-m_after/max(m_sum,1)):.1f}% removed), "
              f"cols {n_sum} -> {n_after} ({100*(1-n_after/max(n_sum,1)):.1f}% removed)")
        print(f"{'rule':<34}{'rows':>9}{'cols':>9}{'calls':>9}{'#inst':>7}")
        for r, (dr, dc, calls, k) in sorted(tot.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
            print(f"{r:<34}{dr:>9}{dc:>9}{calls:>9}{k:>7}")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, ["set", "instance", "rule", "rows", "cols", "calls"])
            w.writeheader()
            w.writerows(rows_out)


if __name__ == "__main__":
    main()
