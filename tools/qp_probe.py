"""Quick QP IPM probe on a list of Maros-Meszaros instances, with option overrides."""
import json, sys, time
from nirnay.io.mps import read_mps
from nirnay.qp.ipm import solve
names = sys.argv[1].split(",")
opts = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
for nm in names:
    m = read_mps(f"data/qp/mm/{nm}.QPS")
    t = time.perf_counter()
    r = solve(m, time_limit=60, **opts)
    print(f"{nm:10s} {r.status:16s} {r.objective:+.10e} it {r.iterations:3d} t {time.perf_counter() - t:6.2f}", flush=True)
