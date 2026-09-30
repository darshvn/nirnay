# NIRNAY

A sovereign optimisation solver for LP, MILP and convex QP, written from the mathematics up.
No solver library inside: the sparse LU, Cholesky and LDLᵀ factorisations, the orderings, the
simplex, interior-point, branch-and-bound and cutting-plane code are all in this repository.

Built for Smart India Hackathon 2026, problem statement **SIH26119** (MRPL): *Indigenous
GPU-Accelerated Optimization Solver (Sovereign Alternative to Xpress / CPLEX)*.

## What it solves

| Class | Engine | Module |
|---|---|---|
| LP | Dual simplex (bounded, dual steepest edge, bound flipping) | `nirnay/lp/simplex.py` |
| LP | Interior point (Mehrotra predictor-corrector, normal equations) | `nirnay/lp/ipm.py` |
| LP | PDLP (restarted PDHG), CPU and GPU via CuPy | `nirnay/lp/pdlp.py` |
| MILP | Branch-and-bound on the warm-started dual simplex: reliability branching, bound propagation, Gomory cuts, diving | `nirnay/mip/` |
| Convex QP | Interior point on the quasi-definite augmented system, own LDLᵀ | `nirnay/qp/ipm.py` |

Linear algebra written here: Gilbert–Peierls sparse LU with eta updates and basis repair
(`linalg/lu.py`), up-looking sparse Cholesky (`linalg/cholesky.py`), quasi-definite LDLᵀ
(`linalg/ldl.py`), minimum-degree ordering (`linalg/ordering.py`). Hot loops are Numba-compiled.

## Results

Every instance is solved in its own process and re-solved by a reference solver that is used
*only* for checking: HiGHS for LP and MILP, Clarabel for QP. See `results/*.csv`.

| Benchmark | Instances | NIRNAY solved | Criterion |
|---|---|---|---|
| Netlib LP, dual simplex | 90 | **90** | objective within 1e-6 of HiGHS |
| Netlib LP, interior point | 90 | 86 | objective within 1e-6 of HiGHS |
| MIPLIB 3, branch-and-bound, 60 s | 64 | see `results/miplib3_bnb_*.csv` | proven optimal, gap 1e-4 |
| Maros–Mészáros QP | 138 | see `results/maros_qpipm_*.csv` | objective within 1e-6 of Clarabel |

Speed is honest, not competitive yet: on Netlib the dual simplex is about 6× slower than
HiGHS in shifted geometric mean. The kernels are compiled, but the driver loops are Python.

## Use

```bash
pip install -e .            # numpy, numba; optional: pip install -e ".[gpu,bench]"
nirnay solve model.mps.gz                    # LP -> dual simplex, MILP -> branch-and-bound, QP -> interior point
nirnay solve model.mps --method ipm -v
nirnay info model.mps
```

```python
from nirnay import solve
from nirnay.io.mps import read_mps
r = solve(read_mps("data/netlib/afiro.mps.gz"), method="simplex")
print(r.status, r.objective, r.x)
```

## Benchmarks

```bash
python bench/run_lp.py  data/netlib  --method simplex --limit 240 --out results/netlib_simplex.csv
python bench/run_mip.py data/miplib  --limit 60  --out results/miplib3_bnb.csv
python bench/run_qp.py  data/qp/mm   --limit 120 --out results/maros_qp.csv
python -m pytest tests -q
```

Benchmark data is not versioned. Netlib and MIPLIB 3 come from the lists in `data/*_list.txt`;
Maros–Mészáros from `doc.ic.ac.uk/~im/QPDATA{1,2,3}.ZIP`.

## Layout

```
nirnay/      solver package (model, io, presolve, linalg, lp, mip, qp, cases, cli)
bench/       benchmark harnesses and comparator scripts
tests/       regression tests against published optima
docs/        case studies, competitor analysis
deliverables/deck    SIH idea deck (pptxgenjs, numbers read from results/)
deliverables/report  LaTeX technical report (tables and plots generated from results/)
results/     benchmark CSVs and logs
```

## References

Primary sources are cited in each module's docstring: Mehrotra (1992), Wright (1997),
Koberstein (2005), Forrest & Goldfarb (1992), Harris (1973), Maros (2003), Fourer (1994),
Gilbert & Peierls (1988), Davis (2005, 2006), Vanderbei (1995), Friedlander & Orban (2012),
Achterberg (2007), Achterberg, Koch & Martin (2005), Balas et al. (1996), Applegate et al.
(2021), Lu & Yang (2023).

Licence: Apache-2.0.
