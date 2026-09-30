# Jury Q&A: SIH26119, NIRNAY

Short answers first; the evidence is in the file named after each one. Numbers come from
`results/*.csv` and can be regenerated with the scripts in `bench/`.

## Is it really from scratch?

| Question | Answer | Evidence |
|---|---|---|
| Which libraries does the solver import? | NumPy (arrays), Numba (compiles our loops to machine code), CuPy (GPU arrays). No optimisation or sparse-factorisation library. | `pyproject.toml` dependencies; `grep -r import nirnay/` |
| Where are HiGHS, Clarabel, SCIP used? | Only in `bench/` and `tools/`, to produce reference answers. Never imported by `nirnay/`. | `bench/run_*.py` |
| Who wrote the LU / Cholesky? | We did: left-looking Gilbert–Peierls LU with threshold pivoting, eta updates and basis repair; up-looking sparse Cholesky; quasi-definite LDLᵀ; minimum-degree ordering. | `nirnay/linalg/` |
| Isn't SciPy used? | Only in tests and the QP benchmark's reference builder. The solver never calls it. | `tests/`, `bench/run_qp.py` |

## Does it work?

| Question | Answer |
|---|---|
| LP | Dual simplex solves all 90 Netlib LPs to within 1e-6 of HiGHS, with or without presolve; interior point 86/90. Duals are exact: `tools/dual_check.py` finds no primal or dual infeasibility on 85 checked instances. |
| MILP | Branch-and-bound proves optimality on the MIPLIB 3 instances listed in `results/miplib3_bnb_v2.csv` within 60 s; the rest end with a proven gap. HiGHS solves more; we say so on the slide. |
| QP | Interior point solves most Maros–Mészáros problems to within 1e-6 of a tight Clarabel reference (`results/maros_qpipm_v2.csv`). The failures are the set's known degenerate instances (LISWET, YAO, UBH1, KSIP), where Clarabel itself reports only "AlmostSolved". |
| How do you know the answers are right? | Every instance is re-solved by an independent solver; we record relative error and the violation of every row and bound. Where a published optimum exists (Netlib, MIPLIB, Maros–Mészáros), we match it. |
| Did you find anything wrong in the reference solvers? | Yes. HiGHS 1.15.1 misreads the RHS section of DPKLO1.QPS and drops coefficients below 1e-9 in KSIP/HUESTIS; our reader reproduces the published optima (`tools/qps_compare.py`). The comparator study also found cuOpt 26.8 reporting a wrong optimum on p0201 (`docs/COMPETITOR_ANALYSIS.md`). |

## Is it fast?

| Question | Answer |
|---|---|
| Against HiGHS? | Slower. On Netlib the dual simplex is about 6× slower in shifted geometric mean; on MIPLIB it solves fewer instances in the same time. |
| Why? | The numerical kernels are compiled, but the simplex iteration and the branch-and-bound driver are still Python. HiGHS is 15 years of tuned C++. |
| Plan | Move the whole simplex iteration into one compiled kernel (profiling shows the per-iteration overhead dominates), then hypersparse FTRAN/BTRAN and a Markowitz LU. The kernel boundary is already clean, so a C++ port is also possible. |
| Where does the GPU help? | Very large LPs at moderate accuracy, via PDLP (only matrix–vector products). The comparator study shows cuOpt's GPU PDLP 10× faster than CPU PDLP on qap15/ex10, and slower than simplex on small LPs. So the GPU engine is one of three, not the default. |

## Why these algorithms?

| Choice | Reason |
|---|---|
| Dual simplex as the default LP engine | Gives exact vertices and warm starts; every leading solver uses it by default; branch-and-bound needs it (a child node is one bound change from its parent). |
| Interior point as well | Polynomial iteration count; robust on large sparse LPs; basis of our QP solver. |
| PDLP on GPU | Only needs A·x and Aᵀ·y, which GPUs do well; scales to problems whose factorisation does not fit in memory. |
| Branch-and-bound + Gomory cuts + reliability branching | The standard modern recipe (Achterberg 2007). The cuts are validated: none of ~1,440 generated on 12 instances cuts off the known optimum. |
| Presolve with exact postsolve | Shrinks models and returns duals of the original model, which sensitivity analysis and refinery planners use (shadow prices of crude and product constraints). |

## Hard questions

| Question | Honest answer |
|---|---|
| Can a student team really compete with CPLEX? | Not on speed this year. The problem statement asks for an indigenous, inspectable core that is correct on the standard benchmarks; that is what we show, with the gap to HiGHS measured, not hidden. |
| What fails today? | The degenerate QP instances listed above; hard MIPLIB instances such as pk1 and fixnet6 (weak heuristics and too few cut families); gt2 when presolve changes the search path (tracked as an expected failure in `tests/test_solvers.py`). |
| Numerical trouble you hit? | (1) A Harris ratio test bound computed from the wrong candidate made pilot/perold report false infeasibility. (2) Phase 1 without cost perturbation cycled on 25fv47. (3) Singleton-row duals depend on the order reductions are undone. (4) A KKT solve that is finite but wrong (QRECIPE, |dx| around 1e46). Each is fixed and has a regression test. |
| Licence? | Apache-2.0, so PSUs and start-ups can adopt it without a vendor. |
| What would MRPL get in 6 months? | Faster compiled simplex, crossover from interior point, more cut families (MIR, cover, clique), RINS/RENS heuristics, and their own planning models run on the solver with the gap to their current tool reported per model. |

## Industrial relevance

`docs/CASE_STUDIES.md` has seven models built from public Indian and standard data: refinery
planning (LP) and multi-period planning (MILP) with DOE assays and PPAC prices, crude
scheduling (Lee et al. 1996), product distribution from PPAC state-wise sales, unit commitment
and economic dispatch from PGLib, and facility location from OR-Library. NIRNAY solves the LP
and QP cases to the HiGHS optimum; the large MILP cases are beyond it today and are listed as
such.
