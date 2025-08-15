#!/usr/bin/env python
# ===============================================================
#  verify_collision_scaling.py   (centroid-contact version)
#  ---------------------------------------------------------------
#  First-contact time versus grid size  (h⁴ scaling test)
#  • Collision = |⟨x⟩₁ − ⟨y⟩₂| < DIST_TOL
#  • Adds fit  Δt = a·h⁴ + b
# ===============================================================
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import cupy as cp
import cupyx.scipy.sparse as css
import argparse, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- constants ------------------------------------------
La, Lb = -80.0, 40.0            # domain (Bohr radii)
GRIDS  = [ 51, 76, 101, 151, 176, 201, 401]         # coarse → fine
tunits = 2.4188843265857e-17
dt     = 0.01 * 2.4e-3 * 1e-15 / tunits   # ≈ 9.921×10⁻⁴ a.u.
NSTEPS = 30_000


# ---------- finite-difference helpers --------------------------
def laplacian1D(N, dx):
    diag = np.ones(N)
    return sp.spdiags(np.vstack([-diag,16*diag,-30*diag,16*diag,-diag]),
                      [-2,-1,0,1,2], N, N) / (12*dx**2)

def GPUkinterm(N, dx, dt):
    diag  = (-1j)*(dt/2)*(-0.5)*cp.ones(N*N)/(12*dx**2)
    diag1 = diag.copy(); diag1[(N-1)::N] = 0
    diag2 = diag.copy(); diag2[(N-2)::N] = 0; diag2[(N-1)::N] = 0
    diag1m, diag1p = cp.roll(diag1, 0), cp.roll(diag1, 1)
    diag2m, diag2p = cp.roll(diag2, 0), cp.roll(diag2, 2)
    return css.spdiags(
        [-diag,16*diag,-diag2m,16*diag1m,-60*diag,16*diag1p,
         -diag2p,16*diag,-diag],
        [-2*(N-1)-2,-(N-1)-1,-2,-1,0,1,2,(N-1)+1,2*(N-1)+2],
        N**2, N**2, 'dia')

# ---------- wave-packet & 1-D ground state ---------------------
def phiWP(x, alpha, x0, p):
    return ((2*alpha/np.pi)**0.25)*np.exp(-alpha*(x-x0)**2 + 1j*p*(x-x0))

def ground_state_1d(N, La, Lb):
    dx   = (Lb-La)/(N-1)
    x    = np.linspace(La, Lb, N)
    H    = -0.5*laplacian1D(N, dx) - sp.diags(1/np.sqrt((x+10)**2 + 1))
    _, vec = spl.eigsh(H, k=1, which='SA')
    g = vec[:,0]
    return g / np.sqrt((np.abs(g)**2).sum()*dx)
# ---------- collision-time detector ----------------------------
DIST_TOL   = 0.5       # “contact’’ separation |x−y| ≤ 0.5  (Bohr radii)
PROB_TOL   = 1e-4      # declare collision when prob > 1e-4

def contact_time(N, alpha, x0, p, nsteps=NSTEPS):
    dx    = (Lb-La)/(N-1)
    xgrid = cp.linspace(La, Lb, N)

    # operators (unchanged)
    Kin_half = GPUkinterm(N, dx, dt)
    Kprop    = (css.spdiags(cp.ones(N**2), [0], N**2, N**2, 'dia')
                + Kin_half + (Kin_half@Kin_half)/2
                + (Kin_half@Kin_half@Kin_half)/6
                + (Kin_half@Kin_half@Kin_half@Kin_half)/24)

    X, Y  = cp.meshgrid(xgrid, xgrid)
    Vsoft = -(1/cp.sqrt((X+10)**2+1) + 1/cp.sqrt((Y+10)**2+1)
              - 1/cp.sqrt((X-Y)**2+1))
    Vprop = css.spdiags(cp.exp(-1j*dt*Vsoft.ravel()), [0], N**2, N**2)

    # initial ψ
    phi  = cp.asarray(phiWP(cp.asnumpy(xgrid), alpha, x0, p))
    gs   = cp.asarray(ground_state_1d(N, La, Lb))
    psi0 = (cp.outer(phi, gs) + cp.outer(gs, phi)) / cp.sqrt(2)
    psi0 = psi0.ravel(); psi0 /= cp.linalg.norm(psi0)*dx

    # mask for physical contact |x−y| ≤ DIST_TOL
    mask = (cp.abs(X - Y) <= DIST_TOL).astype(cp.float64).ravel()

    psi = psi0.copy()
    for n in range(nsteps+1):
        prob_contact = cp.asnumpy(cp.sum(cp.abs(psi)**2 * mask) * dx * dx)
        if prob_contact > PROB_TOL:
            return n * dt            # first contact in atomic units
        if n < nsteps:
            psi = Kprop @ (Vprop @ (Kprop @ psi))

    return np.nan                    # never collided within nsteps


# ---------- main routine ---------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collision-time scaling plot (Δt ∝ h⁴ + const)")
    parser.add_argument('--nsteps', type=int, default=NSTEPS,
                        help=f'total steps (default {NSTEPS})')
    parser.add_argument('--tol', type=float, default=DIST_TOL,
                        help=f'contact tolerance in Bohr (default {DIST_TOL})')
    args = parser.parse_args()

    DIST_TOL = args.tol        # allow CLI override
    alpha, x0, p = 0.08, 8.0, -1.4

    times, h4_vals = [], []
    print("\nGrid   h         t_contact (a.u.)")
    for N in GRIDS:
        tcol = contact_time(N, alpha, x0, p, nsteps=args.nsteps)
        h    = (Lb-La)/(N-1)
        times.append(tcol); h4_vals.append(h**4)
        print(f"{N:3d}  {h:8.4f}   {tcol:11.4f}")

    # reference = finest grid
    t_ref = times[-1]
    delta_t = np.array(times) - t_ref
    h4 = np.array(h4_vals)

    # ---------- a·h⁴ + b fit (coarse grids only) ----------------
    A = np.vstack([h4[:-1], np.ones_like(h4[:-1])]).T
    a, b = np.linalg.lstsq(A, delta_t[:-1], rcond=None)[0]
    fit_line = a * h4 + b

    # ---------- plot -------------------------------------------
    plt.figure(figsize=(4.5,3.5))
    plt.scatter(h4[:-1], delta_t[:-1],
                label='observed Δt', marker='o', s=60)
    plt.plot(h4, fit_line, '--',
             label=fr'fit Δt = {a:.3e}·h$^4$ + {b:.3e}')
    plt.xlabel('$h^{4}$')
    plt.ylabel(r'$\Delta t_{\mathrm{contact}}$  (a.u.)')
    plt.title('Contact-time shift vs $h^{4}$')
    plt.legend()
    plt.tight_layout()
    out_path = Path('contact_scaling.pdf')
    plt.savefig(out_path, dpi=150)
    print(f"\n[plot saved → {out_path.resolve()}]")

    # ---------- table of shifts --------------------------------
    print("\nΔt_contact relative to finest grid (N=256):")
    for N, dt_shift, h4v in zip(GRIDS[:-1], delta_t[:-1], h4[:-1]):
        print(f"N={N:3d}: Δt = {dt_shift:.5e} a.u.   h⁴ = {h4v:.2e}")
