# ------------------------------------------------------------ imports
import os, sys, argparse
import numpy as np
import cupy as cp
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import cupyx.scipy.sparse as css
import cupyx.scipy.sparse.linalg as cssl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
# ------------------------------------------------------------ CLI
parser = argparse.ArgumentParser(prog='tdse2d_dataset',
    description='Generate paired coarse/fine TDSE trajectories')
parser.add_argument('--outpath', required=True, help='root output folder')
parser.add_argument('--nsteps',  type=int, required=True,
                    help='total number of time steps')
parser.add_argument('--saveplot', action=argparse.BooleanOptionalAction,
                    help='also save coarse|fine contour plots')
parser.add_argument('--saveint', type=int, default=300,
                    help='snapshot interval (default=300)')
args = parser.parse_args()

# ------------------------------------------------------------ helpers
def laplacian1D(N, dx):
    diag = np.ones(N)
    return sp.spdiags(np.vstack([-diag,16*diag,-30*diag,16*diag,-diag]),
                      [-2,-1,0,1,2], N, N) / (12*dx**2)

def GPUlaplacian2D(N, dx):
    diag  = cp.ones([N*N])/(12*dx**2)
    diag1 = diag.copy(); diag1[(N-1)::N] = 0
    diag2 = diag.copy(); diag2[(N-2)::N] = 0; diag2[(N-1)::N] = 0
    diag1m, diag1p = cp.roll(diag1, 0), cp.roll(diag1, 1)
    diag2m, diag2p = cp.roll(diag2, 0), cp.roll(diag2, 2)
    return css.spdiags(
        [-diag,16*diag,-diag2m,16*diag1m,-60*diag,16*diag1p,
         -diag2p,16*diag,-diag],
        [-2*(N-1)-2,-(N-1)-1,-2,-1,0,1,2,(N-1)+1,2*(N-1)+2],
        N**2, N**2, 'dia')

def GPUkinterm(N, dx, dt):
    diag  = (-1j)*(dt/2)*(-0.5)*cp.ones([N*N])/(12*dx**2)
    diag1 = diag.copy(); diag1[(N-1)::N] = 0
    diag2 = diag.copy(); diag2[(N-2)::N] = 0; diag2[(N-1)::N] = 0
    diag1m, diag1p = cp.roll(diag1, 0), cp.roll(diag1, 1)
    diag2m, diag2p = cp.roll(diag2, 0), cp.roll(diag2, 2)
    return css.spdiags(
        [-diag,16*diag,-diag2m,16*diag1m,-60*diag,16*diag1p,
         -diag2p,16*diag,-diag],
        [-2*(N-1)-2,-(N-1)-1,-2,-1,0,1,2,(N-1)+1,2*(N-1)+2],
        N**2, N**2, 'dia')

def phiWP(x, alpha, x0, p):
    return ((2*alpha/np.pi)**0.25)*np.exp(-alpha*(x-x0)**2 + 1j*p*(x-x0))

# ------------------------------------------------------------ constants
tunits = 2.4188843265857e-17              
dt     = 0.01*2.4e-3*1e-15 / tunits       

SAMPLE_INT   = args.saveint               
SAVE_PLOTS   = bool(args.saveplot)
La, Lb       = -80.0, 40.0                 # domain
GRIDS        = [64, 256]                  # [coarse, fine]
NTRAJ        = 20                          # number of trajectories
rng          = np.random.default_rng(666)

# random Gaussian-packet parameters
alphas = rng.uniform(0.05, 0.10, NTRAJ)
x0s    = rng.uniform(5.0,  10.0, NTRAJ)
ps     = rng.uniform(-1.8, -1.0, NTRAJ)

# ------------------------------------------------------------ 1-D ground states
numgs_dict = {}
for N in GRIDS:
    dx = (Lb-La)/(N-1)
    ham1d = -0.5*laplacian1D(N, dx) - sp.spdiags(
        [((np.linspace(La,Lb,N)+10)**2 + 1)**(-0.5)], [0], N, N)
    evals, evecs = spl.eigsh(ham1d, k=1, which='SA')
    numgs_dict[N] = evecs[:,0] * dx**(-0.5)
    print(f"[init] ground-state norm check (N={N}): "
          f"{np.sum(np.abs(numgs_dict[N])**2)*dx:.6f}")

# ------------------------------------------------------------ figure root
fig_root = os.path.join(args.outpath, 'figures')
os.makedirs(fig_root, exist_ok=True)

# ------------------------------------------------------------ propagation
print("\n===== TDSE propagation start =====\n")
for traj in range(NTRAJ):
    alpha, x0_, p_ = alphas[traj], x0s[traj], ps[traj]
    print(f"[traj {traj:04d}] alpha={alpha:.4f}, x0={x0_:5.2f}, p={p_:5.2f}")

    # folder for plots of this trajectory
    fig_dir = os.path.join(
        fig_root,
        f"low_traj{traj:04d}_a{alpha:.3f}_x{x0_:05.2f}_p{p_:05.2f}")
    if SAVE_PLOTS:
        os.makedirs(fig_dir, exist_ok=True)

    coarse_snapshots = {}  
    
    for N in GRIDS: 
        start = time.time()
        dx    = (Lb-La)/(N-1)
        xgrid = cp.linspace(La, Lb, N)
        xmat, ymat = cp.meshgrid(xgrid, xgrid)

        # operators
        kin_half = GPUkinterm(N, dx, dt)
        gpukinprop = (css.spdiags(cp.ones(N**2), [0], N**2, N**2, 'dia')
                      + kin_half
                      + (kin_half@kin_half)/2
                      + (kin_half@kin_half@kin_half)/6
                      + (kin_half@kin_half@kin_half@kin_half)/24)
        vmat = ( -((xmat+10)**2+1)**(-0.5)
                 -((ymat+10)**2+1)**(-0.5)
                 +((xmat-ymat)**2+1)**(-0.5) )
        gpupotprop = css.spdiags(
            [cp.exp((-1j)*dt*vmat.reshape(-1))], [0], N**2, N**2)

        # initial Ψ
        phi_vec = cp.array(phiWP(cp.asnumpy(xgrid), alpha, x0_, p_))
        numgs   = cp.array(numgs_dict[N])
        psi = (cp.outer(phi_vec, numgs) + cp.outer(numgs, phi_vec))/cp.sqrt(2.0)
        psi = psi.ravel()
        psi *= (cp.sum(cp.abs(psi)**2)*dx*dx)**(-0.5)

        # storage for this grid / trajectory
        traj_frames = []

        rootN = os.path.join(args.outpath, f"grid{N}", f"traj{traj:04d}")
        os.makedirs(rootN, exist_ok=True)

        # time loop
        for step in range(args.nsteps+1):        
            if step % SAMPLE_INT == 0:
                psiM = psi.get().reshape(N, N)
                traj_frames.append(psiM)

                if N == GRIDS[0]:               
                    coarse_snapshots[step] = psiM

                if SAVE_PLOTS and N == GRIDS[-1]:
                    coarse = coarse_snapshots[step]
                    fine   = psiM
                    vmax   = max(np.abs(coarse).max(), np.abs(fine).max())

                    plt.figure(figsize=(8,3.5))
                    plt.subplot(1,2,1)
                    plt.contourf(np.abs(coarse), levels=40,
                                 vmin=0, vmax=vmax)
                    plt.title(f'|Ψ| coarse {GRIDS[0]}  (t={step*dt:.2e} a.u.)')
                    plt.axis('off')

                    plt.subplot(1,2,2)
                    plt.contourf(np.abs(fine), levels=40,
                                 vmin=0, vmax=vmax)
                    plt.title(f'|Ψ| fine {GRIDS[1]}')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        fig_dir, f"step{step:06d}.pdf"), dpi=150)
                    plt.close()

            # propagate (skip after last snapshot)
            if step < args.nsteps:
                psi = gpukinprop @ (gpupotprop @ (gpukinprop @ psi))

        # ----- save whole trajectory array
        traj_arr = np.stack(traj_frames, axis=0)   # shape (T,N,N)
        np.save(os.path.join(rootN, 'trajectory.npy'), traj_arr)
        end = time.time()
        print('Total wallclock time: ', (end - start))

        # cleanup GPU
        del kin_half, gpukinprop, gpupotprop, psi
        cp.get_default_memory_pool().free_all_blocks()
    # end grid loop
# end trajectory loop
print("\n===== all trajectories complete =====\n")
