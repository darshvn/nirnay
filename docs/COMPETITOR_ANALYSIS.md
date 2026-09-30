# Competitor analysis: what the established LP / MILP / QP solvers actually do

Hands-on study of the solvers NIRNAY will be benchmarked against, done on the SIH 2026 development
machine (Windows 11, i5-11400H 6C/12T, 8 GB RAM, RTX 3050 Laptop 4 GB, WSL2 Ubuntu 24.04). Everything
below comes from installing the tools, running their own examples, and reading the logs of the
cross-solver runs in `results/comparators_lp.csv`, `results/comparators_mip.csv` and
`results/comparators_qp.csv` (driver: `bench/comparators.py`). These tools are comparators and
references only; nothing here is imported by `nirnay/`.

Date of the runs: 30 September 2026. All timings are single runs on a shared laptop, so treat
anything below 50 ms as noise and differences under 2x as not significant.

## 1. Install matrix

| Solver / package | Version | Where it runs | Install command | Status |
|---|---|---|---|---|
| HiGHS (`highspy`) | 1.15.1 (git 04024d7) | Windows | pre-installed | works: dual simplex, IPX interior point + crossover, cuPDLP-C (`solver=pdlp`), HiPDLP (`solver=hipdlp`), MIP, QP (`qpasm`). `solver=hipo` refuses: wheel lacks amd/blas/metis/rcm |
| SCIP (`PySCIPOpt`) | PySCIPOpt 6.2.1, SCIP 10.0.2, SoPlex 8.0.2, Ipopt 3.14.19, nauty 2.8.8 | Windows | `pip install pyscipopt` | works, LP and MIP |
| Google OR-Tools | 9.15.6755 | Windows | `pip install ortools` | works: GLOP, PDLP, CP-SAT, bundled CLP, CBC, SCIP, HiGHS. `GLPK`, `XPRESS`, `GUROBI`, `CPLEX` backends present but "not linked in / no license" |
| CBC (python-mip + cbcbox) | mip 2.0.0, cbcbox 2.935 (`CBC devel git:146ce89`) | **WSL2 only** | `pip install mip` (Windows and WSL venv) | Windows: **blocked**, `libCbc-0.dll` and `cbc.exe` fail with error 0x11c7 "An Application Control policy has blocked this file" (Smart App Control is on; not changed). Works in WSL |
| PuLP | 4.0.0 | Windows | `pip install pulp` | imports; `listSolvers(onlyAvailable=True)` = PYGLPK, CPSAT, COIN_CMD, SCIP_PY, HiGHS. Not used by the driver (its CBC is the same blocked binary) |
| GLPK (`swiglpk`) | 5.0.13 (GLPK 5.0) | Windows | `pip install swiglpk` | works: simplex, interior point, branch-and-cut. `glpsol` binary not installed (not needed) |
| OSQP | 1.1.3 | Windows | `pip install osqp` | works |
| Clarabel | 0.11.1 (Rust, faer direct solver) | Windows | `pip install clarabel` | works |
| qpsolvers | 4.13.0 | Windows | `pip install qpsolvers` | works; `available_solvers = ['clarabel', 'highs', 'osqp']`. `qpsolvers[open_source_solvers]` fails on Windows because `proxsuite` has to be compiled (cmake / cmeel build error) |
| NVIDIA cuOpt | 26.8.0 (git 400863c1), `libcuopt-cu13` wheel with `cuopt_cli` | **WSL2 + GPU** | `python3.12 -m venv --without-pip /opt/cuopt/venv` (apt lock was held by another agent, so pip was bootstrapped with `python3 -m pip --python /opt/cuopt/venv/bin/python install pip`), then `pip install --extra-index-url=https://pypi.nvidia.com libcuopt-cu13==26.8.0` | works on the RTX 3050 (CUDA 13.3 runtime, driver 616.92). PDLP, dual simplex, barrier (cuDSS), concurrent, MIP |
| cuOpt Python package (`cuopt-cu13`) | 26.8.0 | WSL2 | attempted | **not installed on purpose**: it pulls `cudf`, `cupy`, `librmm`, `libcudf` (multi-GB). `libcuopt` alone (~1.9 GB of wheels: libcuopt 450 MB, cublas 439 MB, cusparse 170 MB, cudss 65 MB, nvjitlink 43 MB, ...) gives the C library and `cuopt_cli`, which reads MPS directly |
| cuPDLP-C (COPT-Public) | - | - | not built | needs `nvcc`; no CUDA toolkit compiler in WSL and the pip `cuda-toolkit` metapackage does not ship one. HiGHS bundles the CPU build of cuPDLP-C (`Solving with cuPDLP-C` in its log), and cuOpt's PDLP is the GPU implementation of the same algorithm family, so both sides are covered by other rows |
| CPLEX / Gurobi / Xpress | - | - | not installed | commercial; only licence/cost facts below. The vendors publish size-limited free editions (see section 9) that could be installed later if the team wants direct numbers |

Total downloads: about 2.1 GB (2.0 GB of that is the cuOpt/CUDA wheel set in WSL's pip cache).
No GPU memory is held between runs; `nvidia-smi` shows no compute processes after each batch.

Demos and tutorials that were run (all in `tools/demos/`, all completed with expected output):
HiGHS `call_highs_from_python_highspy.py`, `chip.py`, `knapsack.py`, `minimal.py`, `network_flow.py`;
PySCIPOpt `atsp.py`, `bpp.py`, `diet.py`, `flp.py`, `kmedian.py`, `lotsizing_lazy.py`, `sudoku.py`;
OR-Tools `simple_pdlp_program.py`, `simple_mip_program.py`, `stigler_diet.py`, `simple_sat_program.py`,
`nurses_sat.py`; cuOpt `basic_lp_example.sh`, `basic_milp_example.sh` (cuopt_cli). The cuOpt Python
demo `mps_example.py` needs the `cuopt` Python package and was not run.

## 2. Cross-solver runs

Ten Netlib LPs (afiro adlittle blend sc50a share2b 25fv47 degen2 pilot4 greenbea stocfor1), ten
MIPLIB 3 instances (p0033 p0201 p0282 lseu stein27 egout mod008 bell5 flugpl misc03), ten small
Maros-Meszaros QPs, 60 s limit each, every solver at its default settings unless the name says
otherwise. Each (instance, solver) pair ran in its own process with a hard kill at 1.5 x limit + 30 s.
Times are wall seconds inside the solve call (file reading excluded); for cuOpt they are the time
the CLI reports, which excludes ~0.3-1.5 s of process start and CUDA context creation per call.
"err" is |obj - reference| / max(|reference|, 1) against the published optimal values.

### 2.1 LP, simplex / interior-point solvers (all correct to 1e-6 except where marked)

| instance | highs | highs-ipm | scip | ortools-glop | ortools-clp | glpk | cbc | clarabel | cuopt | cuopt-dualsimplex | cuopt-barrier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| afiro | 0.002 | 0.003 | 0.002 | 0.001 | 0.001 | 0.000 | 0.022 | 0.001 | 0.241 | 0.150 | 1.649 |
| adlittle | 0.003 | 0.005 | 0.004 | 0.002 | 0.001 | 0.001 | 0.007 | 0.001 | 0.304 | 0.150 | 2.392 |
| blend | 0.004 | 0.004 | 0.006 | 0.002 | 0.014 | 0.001 | 0.014 | 0.002 | 0.218 | 0.110 | 1.112 |
| sc50a | 0.002 | 0.003 | 0.003 | 0.002 | 0.001 | 0.000 | 0.006 | 0.001 | 0.119 | 0.170 | 1.062 |
| share2b | 0.004 | 0.011 | 0.017 | 0.002 | 0.001 | 0.001 | 0.009 | 0.002 | 0.449 | 0.190 | 3.054 |
| 25fv47 | 0.147 | 0.104 | 0.425 | 0.108 | 0.154 | 0.139 | 0.584 | 0.080 | 1.256 | 0.920 | 2.545 |
| degen2 | 0.015 | 0.024 | 0.040 | 0.015 | 0.011 | 0.017 | 0.031 | 0.015 | 0.583 | 0.240 | 1.810 |
| pilot4 | 0.028 | 0.044 | 0.112 | 0.033 | 0.032 | 0.147 | 0.143 | 0.090 | 2.767 | 0.220 | 0.961 |
| greenbea | 0.261 | 0.355 | 3.324 | 0.711 | 0.834 | 0.624 | 0.720 | 3.817 (err 1e-03) | 0.551 | 1.100 | 2.592 |
| stocfor1 | 0.006 | 0.048 | 0.554 | 0.062 | 0.059 | 0.013 | 0.007 | 0.018 | 0.135 | 0.100 | 0.413 |

Iteration counts (25fv47, 821 x 1571): HiGHS dual simplex 2583, GLOP 1389, CLP 2706, GLPK 1615,
cuOpt dual simplex 2579, HiGHS IPX 37 IPM + 9 crossover, cuOpt barrier 36, Clarabel 26.

### 2.2 LP, first-order (PDLP / ADMM) solvers and GLPK's interior point

| instance | highs-pdlp | highs-hipdlp | ortools-pdlp | ortools-pdlp-tight | cuopt-pdlp | glpk-interior | osqp | osqp-tight |
|---|---|---|---|---|---|---|---|---|
| afiro | 0.004 | 0.002 | 0.002 | 0.017 | 0.970 (err 6e-05) | 0.000 | 0.000 (err 9e-04) | 0.001 (err 1e-06) |
| adlittle | 0.029 | 0.012 | 0.021 | 0.054 | 1.017 (err 7e-05) | 0.001 | 0.008 (err 5e-04) | IT |
| blend | 0.008 | 0.005 | 0.012 | 0.031 | 1.180 (err 2e-05) | 0.001 | 0.009 (err 4e-04) | IT |
| sc50a | 0.003 | 0.002 | 0.004 (err 2e-06) | 0.012 | 1.389 (err 5e-05) | 0.000 | 0.001 (err 4e-04) | 0.008 |
| share2b | 0.020 | 0.132 | 0.180 | 0.477 | 8.557 (err 1e-04) | 0.001 | inacc | IT |
| 25fv47 | 3.859 | 1.909 | 1.930 (err 2e-06) | 6.999 | 2.456 (err 5e-05) | 0.067 | IT | TL |
| degen2 | 0.105 | 0.039 | 0.058 | 0.289 | 1.766 (err 2e-05) | 0.014 | 0.005 (err 4e-05) | 0.436 |
| pilot4 | TL | TL | TL | TL | 32.1 (err 4e-04) | infeasible | inacc | TL |
| greenbea | TL | TL | TL | TL | TL | infeasible | IT | TL |
| stocfor1 | 0.013 | 0.104 | 0.059 (err 2e-06) | 0.063 | 0.709 (err 4e-06) | 0.003 | inacc | IT |

TL = time limit (60 s; 120 s hard kill for highs-pdlp, see 3.1), IT = iteration limit, inacc =
"solved inaccurate". Tolerances: HiGHS PDLP 1e-7 relative, OR-Tools PDLP 1e-6 (tight: 1e-8),
cuOpt PDLP 1e-4, OSQP 1e-3 (tight: 1e-6), GLPK interior 1e-8.

### 2.3 MIP (correct to the 1e-4 gap tolerance except where marked)

| instance | highs | scip | ortools-scip | ortools-cbc | ortools-cpsat | cbc | glpk | glpk-cuts | cuopt |
|---|---|---|---|---|---|---|---|---|---|
| p0033 | 0.058 | 0.094 | 0.181 | 0.111 | 0.198 | 0.031 | 0.007 | 0.065 | 0.350 |
| p0201 | 1.108 | 1.652 | 1.513 | 1.443 | 2.529 | 1.195 | 0.495 | 2.475 | 1.590 (err 7e-03) |
| p0282 | 0.550 | 0.910 | 0.287 | 0.691 | 0.243 | 0.960 | 0.139 | 0.724 | 0.900 |
| lseu | 0.398 | 1.123 | 0.654 | 0.480 | 0.404 | 0.313 | 0.895 | 4.426 | 1.050 |
| stein27 | 0.786 | 0.343 | 1.421 | 1.048 | 0.102 | 0.595 | 0.294 | 0.878 | 0.960 |
| egout | 0.032 | 0.017 | 0.017 | 0.019 | MODEL_INVALID | 0.025 | 0.082 | 0.167 | 0.470 |
| mod008 | 1.279 | 0.154 | 0.833 | 0.143 | 0.209 | 0.137 | 0.356 | 5.961 | 1.140 |
| bell5 | 0.506 | 0.322 | 0.269 | TL | 0.256 | 0.872 | TL | 2.611 | 0.810 |
| flugpl | 0.116 | 0.050 | 0.111 | 0.039 | 0.156 | 0.085 | 0.005 | 0.009 | 0.250 |
| misc03 | 0.959 | 2.200 | 1.273 | 2.236 | 0.089 | 1.089 | 0.200 | 4.005 | 1.190 |

Threads: HiGHS MIP "Thread count 6 (of 12 threads). Using 1 max workers. Parallel search off";
SCIP and GLPK single-threaded; CP-SAT "Setting number of workers to 12"; cuOpt "Exploring the B&B
tree using 11 threads" plus the GPU; CBC single-threaded. Node counts: p0201 HiGHS 5, SCIP 17,
OR-Tools SCIP 35, OR-Tools CBC 40, CBC 84, cuOpt 1402; stein27 HiGHS 1434, SCIP 191, CBC 655.

### 2.4 QP (Maros-Meszaros, small)

| instance | highs | clarabel | scip | osqp | osqp-tight |
|---|---|---|---|---|---|
| HS21 | 0.002 | 0.000 | 0.063 | 0.000 | 0.000 |
| HS35 | 0.002 | 0.000 | 0.073 | 0.000 (err 2e-06) | 0.000 |
| QAFIRO | 0.002 | 0.001 | 0.054 (err 3e-06) | 0.000 (err 6e-05) | 0.000 (err 2e-06) |
| QADLITTL | 0.006 | 0.001 | 0.176 | 0.005 (err 7e-04) | 0.006 |
| CVXQP1_S | 0.003 | 0.001 | 0.422 | 0.001 (err 1e-04) | 0.002 |
| DUALC1 | 0.003 | 0.002 | 0.074 (err 7e-06) | 0.002 (err 1e-04) | 0.004 |
| PRIMALC1 | 0.004 | 0.003 | 0.162 | IT | "unbounded" (wrong) |
| QSC205 | 0.005 | 0.004 | 0.314 | 0.001 (err 1e-05) | 0.001 |
| QSHARE2B | 0.005 | 0.002 | 0.043 | inacc | IT |
| QPCBLEND | 0.006 | 0.002 | 0.343 | 0.000 (err 1e-03) | 0.007 |

HiGHS QP = active-set (`qpasm`, 239 iterations on QADLITTL); Clarabel = interior point (9-19
iterations everywhere); SCIP treats the QP as a nonlinear program (Ipopt / spatial B&B, hence the
1e-6 level errors); OSQP = ADMM.

### 2.5 Two larger LPs, CPU vs GPU

`qap15` (6330 x 22275, 94950 nz) and `ex10` (69608 x 17680, 1.16 M nz), both from the Mittelmann
LP sets in `data/large/`, 60 s limit:

| solver | qap15 | ex10 |
|---|---|---|
| highs (dual simplex, 6 threads allowed) | TL, 28921 iterations | TL, 12563 iterations |
| highs-ipm (IPX + crossover) | 12.9 s (2585 IPM+crossover its, obj 1040.994041) | TL: "Ipx: IPM optimal" after 16 iterations, then "Crossover reached time limit" at 7999 crossover iterations |
| ortools-pdlp (CPU, 1 thread, 1e-6) | 5.7 s, 8192 its, obj 1040.994156 | 8.5 s, 448 its, obj 100.0000033 |
| cuopt-pdlp (GPU, 1e-4) | 0.57 s, 2800 its, obj 1041.0151 (err 2e-5) | 2.0 s, 460 its, obj 100.005778 (err 6e-5) |
| cuopt concurrent | 0.55 s (PDLP won) | 2.9 s (PDLP won) |
| cuopt-barrier (GPU cuDSS) | 10.5 s, 23 its | 17.7 s, "Suboptimal solution found in 16 iterations" |

This is the regime PDLP was built for: structured, larger, low-accuracy-tolerant LPs where a
simplex times out and an IPM factorisation is expensive. The 10 Netlib instances are the opposite
regime and PDLP loses there on every implementation.

## 3. What each solver actually runs (from the logs)

### 3.1 HiGHS 1.15.1

Default LP path: presolve (log shows the pass-by-pass shrinking, then "Presolve reductions: rows
673(-148); columns 1439(-132); nonzeros 9926(-474)" on 25fv47), then "Using dual simplex solver"
on every one of the 10 LPs (`simplex_strategy = 1` = dual, `presolve = choose`, `solver = choose`
never picked IPM for these sizes). Log shows Ph1 / Du infeasibility counts per report line. Then
"Performed postsolve / Solving the original LP from the solution after postsolve", which is a
clean-up simplex on the original model, usually 0 iterations.

`solver=ipm` runs IPX 1.0 ("IPX model has 673 rows..."), with crossover on by default
(`run_crossover = on`; "Crossover iterations: 9" on 25fv47, 2557 on qap15). `ipm_optimality_tolerance
= 1e-08`, `kkt_tolerance = 1e-07`.

`solver=pdlp` is the CPU build of cuPDLP-C: "Solving with cuPDLP-C / running scaling / - use Ruiz
scaling / - use PC scaling"; `pdlp_scaling_mode = 5` (Ruiz + Pock-Chambolle), `pdlp_ruiz_iterations
= 10`, `pdlp_restart_strategy = 2` (adaptive), `pdlp_step_size_strategy = 1` (adaptive),
`pdlp_optimality_tolerance = 1e-07`. **Defect found**: with any finite `time_limit`, cuPDLP-C stops
~1000x too early and reports "Time limit reached" with objective 0 (25fv47: limit 60 -> stops at
0.09 s after 744 iterations; limit 1000 -> 1.03 s; no limit -> optimal after 63240 iterations,
3.9 s). Reproduced 5 times in-process, so the `highs-pdlp` column above was run with no time limit
and a 120 s hard kill. `solver=hipdlp` is the newer native HiGHS PDLP and behaves correctly (same
tolerance, "adaptive" restarts, 640 iterations on afiro vs 320 for cuPDLP-C).

Presolve rule census (HiGHS, `presolve_rule_logging=true`, whole Netlib set, 90 LPs; run with
`tools/presolve_census.py`): rows 66329 -> 42033 (36.6% removed), columns 173631 -> 137047 (21.1%).
Rows / columns removed per rule, and how many instances the rule fired on:

| rule | rows | cols | instances |
|---|---|---|---|
| Aggregator (free column substitution into a row) | 8060 | 8022 | 68 |
| Doubleton equation | 5630 | 5630 | 66 |
| Forcing row | 1615 | 7022 | 44 |
| Singleton row | 3885 | 2544 | 72 |
| Free col substitution | 2550 | 2550 | 48 |
| Fixed column | 0 | 3914 | 40 |
| Dominated col | 0 | 3392 | 49 |
| Parallel rows and columns | 205 | 3181 | 50 |
| Redundant row | 1396 | 0 | 46 |
| Empty row | 780 | 0 | 34 |
| Empty column | 0 | 250 | 17 |
| Forcing col | 141 | 79 | 7 |
| Dependent equations | 34 | 0 | 6 |

On the 64 MIPLIB 3 instances the same presolve removes 34.8% of rows and 39.2% of columns (HiGHS does
not print the per-rule table for MIPs).

MIP: the log is a single table with a one-letter "Src" column that says which component produced
each new incumbent. Over the 10 MIPLIB runs the incumbent-improving events were: L (sub-MIP, i.e.
RINS/RENS) 14, S (solve LP / rounding of LP solution) 10, J (feasibility jump, always first, before
any LP) 7, R (randomised rounding) 5, C (central rounding) 4, T (node evaluation) 2, B (branching)
2, u (trivial upper) 2, H 1, p 1. Options: `mip_heuristic_effort = 0.05`, feasibility jump / RINS /
RENS / root reduced-cost on, ZI-round and shifting off, `mip_rel_gap = 0.0001`. HiGHS restarts the
root when reduced-cost fixing made enough columns inactive: p0282 "31.2% inactive integer columns,
restarting / Model after restart has 61 rows, 112 cols", then again to 46 x 78 and 25 x 43. It
reports the cut pool size ("Cuts 2947, InLp 39" on p0201) but not cut families. Symmetry detection
runs at the root ("Found 2 generator(s)" on p0201, 8 on stein27). p0201 log excerpt:

```
 J       0       0         0   0.00%   -inf            11340              Large ...      0     0.0s
 R       0       0         0   0.00%   7125            9160              22.22% ...     82     0.0s
 C       0       0         0   0.00%   7210.46345      8530              15.47% ...    222     0.1s
 L       0       0         0   0.00%   7398.961736     7805               5.20% ...    897     0.4s
0.6% inactive integer columns, restarting
```

### 3.2 SCIP 10.0.2 / SoPlex 8.0.2 (PySCIPOpt 6.2.1)

LP: SCIP's own presolve first ("presolving (14 rounds: 14 fast, 4 medium, 4 exhaustive): presolved
problem has 1434 variables ... and 696 constraints" on 25fv47), then SoPlex dual simplex. It is 3-100x
slower than HiGHS on pure LPs (stocfor1 0.55 s vs 0.006 s) because it goes through the full MIP
machinery; not the reference to use for LP timing, but a good reference for MIP algorithm design
because `printStatistics()` breaks everything down.

MIP presolve: the plugins that did work on the 10 instances (sum of the statistics tables) were
`linear` (48 fixed, 40 aggregated, 338 bound changes, 217 constraints deleted, 310 coefficient
changes), `knapsack` (197 constraints deleted, 95 added = knapsack-to-clique reformulation, 394 side
changes), `logicor` (671 side changes), `setppc`, `domcol` (12 fixed on p0201), `probing`,
`components`, `dualfix`, `trivial`. `symmetry` ran on 3 instances and deleted 1 constraint.
Dual presolvers (`dualagg`, `dualinfer`, `dualcomp`) never fired on this set.

Cuts, from the Separators table (cuts added to pool / cuts applied to the LP, summed over the 10):
gomory 12894 / 361 (of which gomorymi 28113 found / 273 applied, strongcg 14374 / 88), aggregation
2274 / 270 (cmir 180, knapsackcover 56, flowcover 34), zerohalf 341 / 131, clique 39 / 20,
impliedbounds 22 / 16, mcf 413 / 4. So SCIP generates ~50x more cuts than it keeps; the cut selector
(`cutselectors: hybrid`) applies 1-3% of the pool. Gomory mixed-integer cuts plus c-MIR are the two
families that actually enter the LP. Root gap closed by cuts (first LP value -> root dual bound):
lseu 927.9 -> 1080.8 with optimum 1120 (80% of the gap), bell5 8951800 -> 8963449 (79%).

Heuristics (Found / Best incumbent over the 10): shifting 131 / 22, strong branching 157 / 9, oneopt
12 / 12, vbounds 8 / 7, rounding 5 / 5, clique 3 / 3, locks 3 / 3, trivial 6 / 3, alns 2 / 2,
farkasdiving 2 / 2, rens 6 / 2, feaspump 2 / 0. The cheap LP-rounding heuristics (`shifting`,
`simplerounding`, `rounding`, `zirounding`, `randrounding`) are called hundreds of times and find most
first incumbents; the expensive sub-MIP heuristics (RENS, ALNS, crossover, RINS) are called once or
twice and produce the final improvements.

Nodes: 6 of the 10 instances were solved at the root or within 5 nodes (p0033, p0282, egout, mod008,
flugpl, misc03); lseu 187, stein27 191, bell5 357 nodes. Default branching is reliability
pseudo-cost branching with strong branching initialisation (visible as "strong branching" finding 157
solutions).

### 3.3 Google OR-Tools 9.15

**GLOP** (primal simplex + the `glop` presolve): the presolve log lists every preprocessor in
order. On greenbea (2392 x 5405): `FixedVariable` (-103 cols), `Singleton`, `ForcingAndImpliedFree`
(-765 cols in one pass), `FreeConstraint`, `ImpliedFree`, `UnconstrainedVariable`,
`DoubletonFreeColumn`, `DoubletonEqualityRow` (-522 rows), then it loops until a fixed point, then
`EmptyConstraint`, `ProportionalColumn` (-131 cols), `SingletonColumnSign`, `Scaling`, ending at
1525 rows, 3456 columns (36% / 36% removed, close to HiGHS's 951 x 2989). Then "Starting basis:
create from scratch", primal simplex with a crash basis ("Crash is set to 2"), and a final
"Final unscaled solution" block with max primal/dual infeasibility after unscaling. GLOP's log on
afiro ends with "Objective error <= 0.000474753", a bound it computes from the perturbations.

**PDLP** (CPU, `ortools.pdlp.python.pdlp`): defaults read from the proto are
`l_inf_ruiz_iterations 5`, `l2_norm_rescaling True`, `restart_strategy ADAPTIVE_HEURISTIC`,
`linesearch_rule ADAPTIVE_LINESEARCH_RULE`, `major_iteration_frequency 64`,
`termination_check_frequency 64`, `primal_weight_update_smoothing 0.5`,
`sufficient_reduction_for_restart 0.1`, `necessary_reduction_for_restart 0.9`, `eps_optimal_absolute
= eps_optimal_relative = 1e-6` in the L2 norm, `num_threads 1`, `presolve_options.use_glop False`,
`use_feasibility_polishing False`. It prints the problem statistics before and after rescaling (25fv47:
matrix range 2e-4..239 becomes 3e-5..1, i.e. rescaling only equalises row/column norms, it does not
compress the dynamic range). Iteration counts are multiples of 64: 384 on afiro, 43968 on 25fv47,
8192 on qap15.

**CP-SAT**: turns the MIP into a pure integer model ("Scaling to pure integer problem"), warns
"31 continuous variable domain with fewer than 1000 values" on afiro, then runs a 12-worker
portfolio. On p0201 the solutions came from `quick_restart_no_lp` (20 of 24), `fj_restart`
(feasibility jump, 2) and `graph_var_lns` (2, including the optimum); over the 10 instances the
solution-producing workers were quick_restart_no_lp 22, core 22, no_lp 14, fj_restart 8, max_lp 6,
graph_var_lns 6. Bounds came from `max_lp`, `pseudo_costs`, `reduced_costs`. It returned
MODEL_INVALID on egout ("Invalid domain in constraint") because the continuous-to-integer scaling
produced an inconsistent domain: CP-SAT is not a general MILP solver for models with continuous
variables. It was the fastest on stein27 (0.10 s, pure set-covering) and misc03 (0.09 s).

**Bundled SCIP / CBC / CLP** are the same libraries as the standalone ones with OR-Tools' MPS reader
and default parameters; OR-Tools CBC timed out on bell5 (standalone CBC 0.87 s), so its bundled CBC
is an older / differently configured build. OR-Tools SCIP is within noise of PySCIPOpt.

### 3.4 CBC (cbcbox 2.935, "CBC devel git:146ce89")

Root cuts, generators and row counts summed over the 10 MIPs: Gomory 9125, MixedIntegerRounding2
1980, ZeroHalf 1365, TwoMirCuts 1099, Probing 729, Knapsack 692, FlowCover 8, Clique 0, OddWheel 0.
CBC also reports the closure per instance: "Cut generation complete - 21 cuts, obj 2819.36 -> 3088.7
in 13 passes" (p0033, optimum 3089: cuts alone closed 99.9% of the root gap), "56 cuts, obj 180000 ->
257641 in 100 passes" (p0282, optimum 258411), "25 cuts, obj 1910 -> 2567.52" (misc03, optimum 3360,
only 45%). Each generator has a "Next run" schedule (every node for Probing/Gomory/Knapsack/MIR/
ZeroHalf, every 1000 nodes for Clique/OddWheel/FlowCover, TwoMir disabled after the root). Root
heuristics run before cuts ("Root node heuristics - best 8115 in 0.045s" on p0201). Nodes: p0033 0,
p0201 84, lseu 655, misc03 306.

### 3.5 GLPK 5.0

Simplex: `presolve` is off by default in `glp_smcp` (turned on here); GLPK's simplex is competitive
with the others on these sizes (25fv47 0.14 s, 1615 iterations; greenbea 0.62 s) thanks to its
"Constructing initial basis... Size of triangular part is 25" crash. Interior point (`glp_interior`,
Mehrotra predictor-corrector, no crossover) fails with "ret=17 ipt_status=3" (numerical instability
declared as infeasible) on pilot4 and greenbea, the two badly scaled LPs. Branch-and-cut: **all cut
generators and heuristics are off by default** (`gmi_cuts`, `mir_cuts`, `cov_cuts`, `clq_cuts`,
`fp_heur`, `ps_heur` = GLP_OFF). Default GLPK still solved 9 of 10 (pure branch and bound with the
dual simplex re-solve, 175 k+ nodes on bell5 before the limit). With all cuts on (`glpk-cuts`) bell5
solves in 2.6 s ("Cuts on level 0: gmi = 5; mir = 28; cov = 25; clq = 4") but lseu, mod008, misc03
get 5-15x slower: cuts are added only at the root and their LP re-solve cost is not offset on small
instances. Its objective is capped at 0.2% gap by `mip_gap` default 0.0 and no tolerance, so it
proves optimality exactly.

### 3.6 Clarabel 0.11.1, OSQP 1.1.3, qpsolvers 4.13

Clarabel (homogeneous embedding interior point, direct LDL via faer, "equilibrate: on, min_scale
1e-4, max_scale 1e4, max iter 10", tolerances 1e-8, "static reg eps1 = 1e-8", "iter refine: on")
solved all 10 LPs and 10 QPs in 8-42 iterations and is the fastest QP solver in the set (QADLITTL
1 ms). Its one failure is instructive: on greenbea, fed without any presolve, it returns status
`Solved` with objective -72462468.49 (true -72555248.13, 1.3e-3 relative error); the returned point
has max equality violation 2.7e-6 and max inequality violation 6.6e-5 with |x| up to 2e6, so its
relative stopping criteria were met in the scaled space while the original-space objective is far
off. Tightening tolerances to 1e-10 gives -72485741 after 162 iterations, still wrong. HiGHS's IPX
solves the same LP correctly after presolve reduced it 60%.

OSQP (ADMM, default `eps_abs = eps_rel = 1e-3`, `max_iter = 4000`, `polishing` off) is 1e-3 to 1e-5 off
on every LP at defaults, hits the iteration limit on 25fv47 and greenbea, and with 1e-6 tolerances
either runs out of 1e6 iterations or (PRIMALC1) declares a bounded QP "dual infeasible". It is a QP
solver for embedded / MPC use, not an LP reference.

qpsolvers is only a uniform wrapper (`solve_qp(P, q, G, h, A, b, lb, ub, solver=...)`); it exposes
clarabel, highs and osqp on this machine. Useful for NIRNAY's QP test harness because it gives one
call signature for three references.

### 3.7 NVIDIA cuOpt 26.8.0 (WSL2, RTX 3050)

LP: default `--method 0` is **concurrent**: it launches PDLP (GPU), dual simplex (CPU) and barrier
(GPU, cuDSS) and takes the first to finish: "Running concurrent (showing only PDLP log) ... Dual
simplex finished in 0.27 seconds / PDLP finished / Barrier finished in 0.38 seconds / Concurrent time:
0.378s / Solved with dual simplex". On all 10 Netlib LPs the CPU dual simplex won; on qap15 and ex10
PDLP won. Presolve: "Using PSLP presolver / PSLP Presolved problem: 718 constraints, 1466 variables"
(25fv47; HiGHS gets 673 x 1439), and for MIPs also "Papilo presolve time: 0.07". Warns "input problem
contains a large range of coefficients: consider reformulating" on 25fv47 (range 2e-4..2e2).

PDLP defaults: `--pdlp-solver-mode 4`, `--absolute/relative-{primal,dual,gap}-tolerance 1e-4`,
`--crossover false`, `--pdlp-precision` native (FP64), and it prints "Objective offset ... scaling
factor". At 1e-4 the objectives are 2e-5 to 4e-4 off (25fv47: 5502.093 vs 5501.846, "Primal
infeasibility (abs/rel): +3.55e-01 / +7.45e-05"). With 1e-8 tolerances (`cuopt-pdlp-tight`) 25fv47
becomes exact but takes 105200 iterations / 12.5 s vs 4500 / 2.5 s, i.e. PDLP's cost grows roughly
linearly in the number of accuracy digits. GPU PDLP on Netlib sizes is 0.5-30 s against
milliseconds for simplex because every iteration is a handful of kernel launches on a 100-10000
nonzero matrix; the GPU only pays off once one SpMV is worth more than the launch latency (ex10:
1.16 M nonzeros, 460 iterations, 2.0 s vs 8.5 s on one CPU thread with OR-Tools PDLP).

Barrier: "Optimal solution found in 36 iterations and 0.618s" on 25fv47 with iterative refinement on
and dual postsolve; on ex10 it stopped at "Suboptimal solution found in 16 iterations" (status other).

MIP: a CPU branch-and-bound ("Exploring the B&B tree using 11 threads") with dual simplex node LPs,
root cuts (summed over 10 instances: MIR 194, Gomory 76, Knapsack 44, Implied Bounds 33, Flow Cover
30, Clique 9, Strong CG 5, Zero-Half 0), symmetry detection, probing ("Probing implied bounds: 180
zero entries, 2592 one entries"), strong branching on 11 threads, and GPU primal heuristics
(incumbent sources D 23, B 8, S 6, H 5 in the log). **Defect found**: on p0201 it reports "Optimal
solution found. Best objective 7.665000e+03, best bound 7.665000e+03, gap 0.00%" in four out of four
runs, while the optimum is 7615 (all other 8 solvers agree; it is the MIPLIB reference value). Its
own log shows a better incumbent 7615 was never found and the bound was cut above the true optimum,
i.e. a wrong dual bound (over-aggressive cut, probing or reduced-cost fixing). bell5 also printed
"Post-solve status: Post solved solution violates constraints. This is most likely due to different
tolerances." seven times and finished at 8966414 (reference 8966406.49, inside the 1e-4 gap). Node
counts are 10-100x those of SCIP/HiGHS (p0201: 1402 vs 17 / 5), and results vary run to run
(p0201: 1001, 1347, 1402, 1661 nodes).

## 4. PDLP tolerance and termination comparison

| implementation | default optimality tolerance | norm / criterion | restart | step size | scaling | check every |
|---|---|---|---|---|---|---|
| OR-Tools PDLP 9.15 | 1e-6 abs + 1e-6 rel | L2, relative to norms of c and b/bounds | adaptive heuristic (KKT-based, sufficient 0.1 / necessary 0.9) | adaptive line search | 5 Ruiz (L-inf) + L2 | 64 iterations |
| HiGHS cuPDLP-C | 1e-7 rel | cuPDLP-C's relative KKT | adaptive (`pdlp_restart_strategy=2`) | adaptive (`=1`) | 10 Ruiz + Pock-Chambolle | per major iteration |
| HiGHS HiPDLP | 1e-7 rel | same family | adaptive (option allows Halpern) | adaptive | same | same |
| cuOpt 26.8 PDLP | 1e-4 abs and rel (primal, dual, gap) | relative primal/dual residual and gap, per-constraint optional | mode-dependent (`--pdlp-solver-mode 4`) | adaptive | on | not printed |

Two cross-implementation facts: (1) every PDLP solved the same 8 of 10 Netlib LPs and failed the same
2 (pilot4, greenbea), the badly scaled ones, so the failure is algorithmic, not implementation
specific; (2) at 1e-4 cuOpt's answers are useless as reference values (4e-4 error on pilot4), at
1e-6 OR-Tools PDLP is at 2e-6, at 1e-8 both are exact but 2-5x slower.

## 5. Defects found in the comparators (so NIRNAY's tests do not trust them blindly)

1. HiGHS 1.15.1 `solver=pdlp` (cuPDLP-C) with a finite `time_limit` stops after limit/1000 seconds.
2. cuOpt 26.8.0 MIP returns a wrong "optimal" 7665 on p0201 (optimum 7615), reproducibly.
3. Clarabel 0.11.1 returns `Solved` at 1.3e-3 objective error on greenbea (no presolve).
4. GLPK 5.0 interior point declares pilot4 and greenbea infeasible.
5. OSQP declares PRIMALC1 dual infeasible at 1e-6 tolerances.
6. OR-Tools CP-SAT rejects egout (continuous variables) as MODEL_INVALID.
7. python-mip / cbcbox binaries are unsigned and blocked by Windows Smart App Control.

For the benchmark harness this means: keep at least two independent references per instance
(HiGHS + SCIP for MIP, HiGHS + GLOP + Clarabel for LP), and compare against the published optimal
value where one exists.

## 6. Presolve: what to build first

Ordered by rows + columns removed on Netlib (HiGHS census, section 3.1) and by what SCIP's and GLOP's
logs show on the MIPs:

1. Singleton rows -> bounds (fires on 72 of 90 LPs) and empty rows / columns.
2. Doubleton equations (66 of 90; 5630 rows) and the general aggregator / free-column substitution
   (8060 rows; 68 of 90): together half of all reductions.
3. Forcing rows (1615 rows but 7022 columns fixed: a forcing row fixes every column in it).
4. Fixed columns, dominated columns (dual argument on costs and signs; 3392 columns).
5. Parallel rows and columns (needs hashing; 50 of 90).
6. For MIPs, SCIP shows the additional payoff of coefficient tightening (310 coefficient changes,
   `linear` presolver), knapsack-to-clique reformulation, probing, and dual fixing; symmetry only
   mattered on stein27 / p0201.
7. Presolve is what makes IPMs robust: Clarabel and GLPK IPM both fail on greenbea without it, IPX
   inside HiGHS succeeds with it.

## 7. Cuts and heuristics: what closes the gap

- Cut families by cuts actually kept in the LP: SCIP gomorymi 273, cmir 180, zerohalf 131, strongcg
  88, knapsack cover 56, flow cover 34, clique 20, implied bounds 16 (sum over 10 MIPLIB 3); CBC by
  cuts generated: Gomory 9125, MIR 1980, ZeroHalf 1365, TwoMir 1099, Probing 729, Knapsack 692;
  cuOpt: MIR 194, Gomory 76, Knapsack 44. Gomory mixed-integer + (c-)MIR is the common core of all
  three; cover / clique / zero-half are the second tier.
- Root cut loops close most of the gap on these instances: CBC 99.9% on p0033, SCIP 80% on lseu and
  bell5; 6 of 10 instances solve at the root in SCIP and HiGHS.
- GLPK's experience (cuts off by default, 5-15x slowdown when forced on for small instances) shows
  that a cut loop without a cut selector / aging policy hurts. SCIP keeps 1-3% of the cuts it
  generates (pool 767 -> 20 applied on p0201).
- First incumbents come from the cheapest heuristics run first: HiGHS feasibility jump (J) before any
  LP on 7 of 10 instances, CP-SAT `fj_restart`, SCIP `shifting` / `locks` / `trivial`. Final
  incumbents come from LP rounding of improved LPs and sub-MIP neighbourhood search (HiGHS L = 14
  improvements, SCIP rens / alns).
- HiGHS's root restart after reduced-cost fixing (p0282: 159 x 199 -> 25 x 43 in three restarts)
  is a cheap trick with a large effect on node LP size.

## 8. What the GPU actually speeds up

- Only PDLP-style first-order methods and dense-ish barrier factorisations; the simplex in cuOpt is
  a CPU code and its branch-and-bound is CPU ("11 threads"), the GPU runs PDLP for root/relaxation
  bounds and the primal heuristics.
- On the 10 Netlib LPs the GPU PDLP is 100-1000x slower than any CPU simplex (kernel-launch bound,
  plus 0.3-1.5 s of CUDA context start per process); cuOpt's own concurrent mode chose the CPU dual
  simplex 10 times out of 10.
- On qap15 (95 k nz) GPU PDLP 0.57 s vs CPU PDLP 5.7 s vs IPX 12.9 s vs dual simplex time-out;
  on ex10 (1.16 M nz) 2.0 s vs 8.5 s vs time-out. That is a 4-10x GPU-vs-one-CPU-thread ratio at
  1e-4 vs 1e-6 tolerance, on a 4 GB laptop GPU.
- cuOpt's concurrent design (race PDLP / simplex / barrier, take the first) is the practical answer
  to "which method": no single method won both regimes.

## 9. Licence and cost

| solver | licence | cost model (primary source) |
|---|---|---|
| HiGHS | MIT | free |
| SCIP / SoPlex / PaPILO | Apache-2.0 (since SCIP 9) | free |
| OR-Tools (GLOP, PDLP, CP-SAT) | Apache-2.0 | free |
| CBC / CLP / CGL | EPL-2.0 (python-mip: EPL-2.0) | free |
| GLPK | GPL-3.0 | free (copyleft: cannot be linked into a non-GPL product) |
| Clarabel | Apache-2.0 | free |
| OSQP | Apache-2.0 | free |
| qpsolvers | LGPL-3.0 | free |
| NVIDIA cuOpt | Apache-2.0 (open source since 2025) | free; NVIDIA GPU required |
| cuPDLP-C | MIT | free |
| IBM CPLEX | proprietary | ibm.com pricing page: Community Edition free, "limited to 1000 variables and 1000 constraints"; Developer Subscription "Starting at $320.00 USD per authorized user per month"; deployment editions "Contact for pricing" |
| Gurobi | proprietary | gurobi.com pricing page: "Commercial licenses are quote-based and billed annually"; no figures published. Free size-limited licence in `pip install gurobipy`: "2,000 variables and 2,000 linear constraints (or 200 variables for models with quadratic terms)"; academic licences free |
| FICO Xpress | proprietary | not published. Community licence free with "the sum of the number of rows and columns restricted to 5000 for linear and mixed-integer problems, and to 200 for quadratic and general nonlinear problems" (Xpress install guide) |

## 10. Lessons for NIRNAY (ordered by expected payoff)

1. **Presolve before anything else.** Singleton rows, doubleton equations, aggregator / free-column
   substitution, forcing rows, fixed and dominated columns, parallel rows/columns, in that order,
   iterated to a fixed point; they remove 21-37% of Netlib and 35-39% of MIPLIB 3 before any
   iteration, and they are the difference between an IPM that fails (Clarabel, GLPK on greenbea)
   and one that works (IPX). Log the per-rule counts like HiGHS does.
2. **Dual simplex is the default LP engine of every winner** (HiGHS, cuOpt concurrent, SCIP/SoPlex
   for node LPs); IPM with crossover is the alternative for large LPs; PDLP is a third engine for
   very large, badly conditioned-for-factorisation LPs at low accuracy. Build them as a portfolio
   and, like cuOpt, run them concurrently and take the first.
3. **Scaling and a crash basis matter measurably**: GLPK's triangular crash and GLOP's "Crash is set
   to 2" give iteration counts 30-50% below a slack basis; PDLPs that only equilibrate norms fail on
   pilot4 / greenbea where the coefficient range is 1e-6..1e4.
4. **Report and test against tolerances explicitly**: the same LP has "optimal" answers spread over
   4e-4 (cuOpt PDLP at 1e-4), 2e-6 (OR-Tools PDLP at 1e-6) and 1e-9 (simplex). NIRNAY's benchmark
   must record the tolerance with every objective and verify KKT residuals in the original,
   unscaled space (Clarabel's greenbea "Solved" shows why).
5. **A finite time limit must be tested** (HiGHS cuPDLP-C stops at limit/1000): every engine gets a
   test that a 60 s limit runs for about 60 s and returns the best available point, never
   objective 0.
6. **MIP root loop = feasibility jump first, then LP, then cheap rounding, then cuts, then sub-MIP
   heuristics, then restart.** That order appears in HiGHS (J, S/R/C, cuts, L, restart), CP-SAT
   (fj_restart, quick_restart_no_lp, lns) and SCIP (trivial/locks/shifting, cuts, rens/alns).
7. **Cut families**: Gomory mixed-integer and (c-)MIR first, then knapsack cover, zero-half, clique
   and implied bounds; keep a pool, apply only cuts that are violated, orthogonal and efficacious
   (SCIP applies 1-3% of what it generates), and age them out. Do not add cuts at every node until
   the root loop is proven to help (GLPK's forced-cuts slowdown).
8. **Restart the root after reduced-cost fixing and presolve again** (HiGHS p0282: 159 x 199 to 25 x
   43 in three restarts); it is cheap because presolve already exists.
9. **Branching**: reliability pseudo-cost branching with strong-branching initialisation is what
   SCIP, CBC, HiGHS and cuOpt all use; the node counts of the solver without a well-tuned version
   (cuOpt, 1402 nodes vs 5-17) show the cost.
10. **Correctness checks in the benchmark, not just speed**: keep two independent references per
    instance and the published optimum, flag any "optimal" that disagrees (this study caught a wrong
    optimum in cuOpt 26.8 and a wrong "Solved" in Clarabel). Determinism is a feature: cuOpt's node
    count varied 1001-1661 across identical runs; HiGHS and SCIP were repeatable.

Secondary observations worth keeping: CP-SAT's pure-integer approach wins on set-covering-like
models (stein27, misc03) and should not be imitated for general MILP; cuOpt's "PSLP" presolve and
HiGHS's presolve end at almost the same size (718 x 1466 vs 673 x 1439 on 25fv47), so a well done
classical presolve is already competitive; every solver prints the coefficient ranges of the model
before solving and warns above ~1e6 dynamic range, which NIRNAY should copy.

## 11. Reproducing

```
python bench/comparators.py data/netlib/afiro.mps.gz --solvers highs,scip,ortools-pdlp,cbc
python bench/comparators.py data/netlib/{afiro,adlittle,...}.mps.gz --solvers highs,highs-ipm,... \
    --time-limit 60 --csv raw_lp.csv --log-dir logs/lp
python tools/merge_comparator_csv.py results/comparators_lp.csv lp raw_lp.csv ...
python tools/presolve_census.py --out census.csv
wsl.exe -d Ubuntu-24.04 -u root -e bash tools/wsl_single.sh cuopt /mnt/c/.../afiro.mps.gz 60 pdlp 1
```

Solver names accepted by `--solvers`: highs, highs-simplex, highs-ipm, highs-pdlp, highs-hipdlp,
scip, ortools-glop, ortools-clp, ortools-pdlp, ortools-pdlp-tight, ortools-scip, ortools-cbc,
ortools-cpsat, glpk, glpk-interior, glpk-cuts, clarabel, osqp, osqp-tight, cbc (WSL), cuopt,
cuopt-pdlp, cuopt-pdlp-tight, cuopt-dualsimplex, cuopt-barrier, cuopt-concurrent (WSL + GPU).
