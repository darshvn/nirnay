"""Check primal and dual feasibility of NIRNAY LP solutions (with presolve) on MPS files.

usage: python tools/dual_check.py data/netlib/*.mps.gz
"""
import sys, numpy as np
from nirnay.io.mps import read_mps
from nirnay import solve
def dual_inf(m, r):
    x=r.x; y=r.y*m.sense; z=m.c-m.A.rmatvec(y); tol=1e-7
    at_l=np.abs(x-m.lb)<=tol*(1+np.abs(m.lb)); at_u=np.abs(x-m.ub)<=tol*(1+np.abs(m.ub))
    zi=np.where(at_l&at_u,0,np.where(at_l,np.maximum(-z,0),np.where(at_u,np.maximum(z,0),np.abs(z))))
    Ax=m.A.matvec(x); rl_=np.abs(Ax-m.rl)<=tol*(1+np.abs(m.rl)); ru_=np.abs(Ax-m.ru)<=tol*(1+np.abs(m.ru))
    yi=np.where(rl_&ru_,0,np.where(rl_,np.maximum(-y,0),np.where(ru_,np.maximum(y,0),np.abs(y))))
    sc=1+np.abs(m.c).max()
    return zi.max(initial=0)/sc, yi.max(initial=0)/sc
bad=0
for f in sys.argv[1:]:
    m=read_mps(f)
    r1=solve(m,method='simplex',presolve=True, time_limit=60)
    if r1.status!='optimal': print(f,'status',r1.status); bad+=1; continue
    a=dual_inf(m,r1); v=m.violation(r1.x)
    flag = max(a)>1e-6 or v['row']>1e-6*(1+np.abs(m.rl[np.isfinite(m.rl)]).max(initial=1))
    if flag: bad+=1
    print(f"{'BAD ' if flag else 'ok  '}{f.split('/')[-1]:22s} zrel {a[0]:.1e} yrel {a[1]:.1e} row {v['row']:.1e} obj {r1.objective:+.10e}")
print('bad',bad)
