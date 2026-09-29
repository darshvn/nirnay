"""Fill-reducing orderings for sparse symmetric factorisation.

Minimum degree (Tinney & Walker 1967; George & Liu, SIAM Review 1989): repeatedly eliminate the
node with the fewest neighbours in the elimination graph. Eliminating a node joins its
neighbours into a clique, which is exactly the fill that node creates.

This implementation keeps the elimination graph explicitly and picks nodes from degree buckets,
with two refinements that matter on optimisation matrices:
  * mass elimination: nodes whose closed neighbourhood equals the pivot's are eliminated together
    (they would each be chosen next anyway), which removes most of the cost on block structure;
  * dense-row deferral: nodes with degree above 10*sqrt(n) are ordered last. Interior-point
    normal equations often carry a few almost-dense rows, and leaving them to the end both
    bounds the work here and keeps the dense part of L in the final block.
"""
from __future__ import annotations

import numpy as np


def minimum_degree(n: int, colptr: np.ndarray, rowidx: np.ndarray) -> np.ndarray:
    """Permutation `perm` such that P M P' factors with little fill; perm[k] is the k-th pivot."""
    adj = [set() for _ in range(n)]
    for j in range(n):
        for p in range(colptr[j], colptr[j + 1]):
            i = int(rowidx[p])
            if i != j:
                adj[i].add(j)
                adj[j].add(i)

    dense_cut = max(16, int(10 * np.sqrt(max(n, 1))))
    dense = [v for v in range(n) if len(adj[v]) > dense_cut]
    dense_set = set(dense)
    for v in dense:                     # take dense nodes out of the graph entirely
        for u in adj[v]:
            if u not in dense_set:
                adj[u].discard(v)
        adj[v] = set()

    alive = [v not in dense_set for v in range(n)]
    degree = [len(adj[v]) for v in range(n)]
    maxdeg = n + 1
    buckets = [set() for _ in range(maxdeg)]
    for v in range(n):
        if alive[v]:
            buckets[degree[v]].add(v)

    order: list[int] = []
    mindeg = 0
    remaining = n - len(dense)
    while remaining > 0:
        while not buckets[mindeg]:
            mindeg += 1
        v = buckets[mindeg].pop()
        nbrs = adj[v]
        # mass elimination: neighbours whose closed neighbourhood equals v's
        closed_v = nbrs | {v}
        group = [v]
        for u in list(nbrs):
            if len(adj[u]) == len(nbrs) and (adj[u] | {u}) == closed_v:
                group.append(u)
        group_set = set(group)
        clique = nbrs - group_set
        for u in group:
            if u != v:
                buckets[degree[u]].discard(u)
            alive[u] = False
            order.append(u)
            remaining -= 1
        for u in clique:
            buckets[degree[u]].discard(u)
            au = adj[u]
            au -= group_set
            au |= clique
            au.discard(u)
            degree[u] = len(au)
            buckets[degree[u]].add(u)
            if degree[u] < mindeg:
                mindeg = degree[u]
        for u in group:
            adj[u] = set()

    order.extend(sorted(dense, key=lambda v: 0))
    perm = np.array(order, dtype=np.int64)
    assert len(perm) == n and len(np.unique(perm)) == n
    return perm


def identity(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)
