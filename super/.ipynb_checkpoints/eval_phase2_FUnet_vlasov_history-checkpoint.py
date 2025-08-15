##!/usr/bin/env python
# ===============================================================
#  eval_phase2_total_compare.py  (Phase‑2 roll‑out, 1D1V VP)
#  ▸ Adds Lyapunov–style diagnostics + bound figure
# ===============================================================


import os, sys, argparse, csv
import numpy as np
import torch, torch.nn.functional as F
from tqdm import trange
from numpy.fft import (
    rfft2, fftshift, fft2, fftfreq,
    fft, ifft                              # for Poisson/E‑field helpers
)
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from sklearn.decomposition import PCA
import time

# ═══════════════════════════════════════════════════════════════
# 0 ▸  TRAINING OBJECTS & LOW‑LEVEL BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════
from train_phase2_FUnet_vlasov import (
    SuperResUNet, VlasovStepper,
    PATH_FINE_E, PATH_FINE_I, PATH_COARSE_I, PATH_COARSE_E
)
from train_phase1_FNO_2d_vlasov  import FNO2dSR
from train_phase1_EDSR_2d_vlasov import EDSR
from train_phase2_vlasov_FNO_OTP import FNO2dOTP              # NEW

SAVE_DIR = "/pscratch/sd/h/hbassi/models"
PATH_WEIGHTS_FUNET   = os.path.join(SAVE_DIR, "FUnet_phase2_64to256_best_hist_two-stream_no_ion.pth")
PATH_WEIGHTS_FNO     = os.path.join(SAVE_DIR, "FNO_phase2_best_hist_two-stream_no_ion.pth")
PATH_WEIGHTS_EDSR    = os.path.join(SAVE_DIR, "EDSR_phase2_best_hist_two-stream_no_ion.pth")
PATH_WEIGHTS_FNO_OTP = os.path.join(SAVE_DIR, "2d_vlasov_two-stream_FNO_OTP_phase2_best.pth")

PATH_STATS_FUNET = "./data/2d_vlasov_funet_phase1_stats_64to256_two-stream_v1.pt"
PATH_STATS_FNO   = "./data/2d_vlasov_fno_phase2_stats_32to128_two-stream_no_ion_v1.pt"
PATH_STATS_EDSR  = "./data/2d_vlasov_two-stream_phase1_EDSR_stats.pt"
PATH_STATS_FNO_OTP = "./data/2d_vlasov_fno_otp_stats.pt"

# ── physics support (unchanged) ─────────────────────────────────
sys.path.append(os.path.abspath("vlasov_fd"))
from setup_.configs      import UnitsConfiguration, ElcConfiguration, IonConfiguration, DerivativeConfiguration
from setup_.enums        import BCType, FDType
from axis                import Axis
from coord.coord_sys     import Coordinate, CoordinateType
from coord.cartesian     import CartesianCoordinateSpace
from grid                import Grid
from field               import ScalarField
from pde_vlasovES        import VlasovPoisson

# ═══════════════════════════════════════════════════════════════
# ✦  Lyapunov‑diagnostic helpers (added)
# ═══════════════════════════════════════════════════════════════
def spectral_norm_fd(fn, x, n_iter=15, eps=1e-4):
    """
    Finite‑difference power iteration to estimate  σ_max(J_fn(x))
    when fn is not autograd‑friendly (e.g. legacy Fortran stepper).
    """
    v = torch.randn_like(x)
    v /= torch.linalg.vector_norm(v)
    for _ in range(n_iter):
        y1, y0 = fn(x + eps * v), fn(x)
        Jv = (y1 - y0) / eps
        v  = Jv / (torch.linalg.vector_norm(Jv) + 1e-12)
    y1, y0 = fn(x + eps * v), fn(x)
    Jv = (y1 - y0) / eps
    return torch.dot(v.flatten(), Jv.flatten()).abs().sqrt().item()

def spectral_norm_rect(fn, x, n_iter=30):
    """
    Power iteration for rectangular Jacobians exploiting
    reverse‑mode (vjp) and forward‑mode (jvp) products.
    """
    y = fn(x)
    u = torch.randn_like(y); u /= torch.linalg.vector_norm(u)
    for _ in range(n_iter):
        (v,) = torch.autograd.grad(y, x, grad_outputs=u,
                                   retain_graph=True, create_graph=True)
        v /= torch.linalg.vector_norm(v) + 1e-12
        _, Jv = torch.autograd.functional.jvp(fn, (x,), (v,), create_graph=True)
        u = Jv / (torch.linalg.vector_norm(Jv) + 1e-12)
        y = fn(x)
    _, Jv = torch.autograd.functional.jvp(fn, (x,), (v,))
    return torch.linalg.vector_norm(Jv).item()

# ═══════════════════════════════════════════════════════════════
# 1 ▸  NORMALISATION + misc. helpers  (unchanged)
# ═══════════════════════════════════════════════════════════════
ve_lims = (-6.0, 6.0)
eps = 1e-8

def moments_1d(f_slice, q=-1.0):
    Nx, Nv = f_slice.shape
    v  = np.linspace(*ve_lims, Nv, endpoint=False)
    dv = v[1] - v[0]
    rho = q * f_slice.sum(-1)        * dv
    J   = q * (f_slice *  v).sum(-1) * dv
    M  = q * (f_slice * v**2).sum(-1)* dv
    return rho, J, M

def radial_energy_spectrum(u):
    """Parseval‑normalised one‑sided radial energy spectrum."""
    N = u.shape[0]
    u_hat  = fftshift(fft2(u))
    energy = np.abs(u_hat)**2 / N**2
    kx     = fftshift(fftfreq(N)) * N
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    kr     = np.sqrt(KX**2 + KY**2).astype(int)
    E      = np.bincount(kr.ravel(), energy.ravel(),
                         minlength=kr.max()+1)
    return np.arange(len(E)), E

def load_stats(path, device, hist_len):
    st = torch.load(path, map_location="cpu")

    if "edsr" in os.path.basename(path).lower():        # EDSR case
        μ_raw = st["mean"]
        σ_raw = st["std"]
        return μ_raw, σ_raw
    else:                                               # FUnet / FNO
        μ_raw = st["data_mean"].to(device)
        σ_raw = st["data_std"].to(device)

    # Broadcast 1‑D vectors to (1, L) if needed
    if μ_raw.dim() == 1:
        μ_raw = μ_raw.unsqueeze(0)
    if σ_raw.dim() == 1:
        σ_raw = σ_raw.unsqueeze(0)

    μ = μ_raw[:, :hist_len]
    σ = σ_raw[:, :hist_len].clamp_min(1e-8)
    return μ, σ


# ── NEW physics helpers ─────────────────────────────────────────

def poisson_residual_norm(rho, dx):
    """‖∇²φ + ρ‖₂ / ‖ρ‖₂  (periodic Poisson)."""
    Nx = rho.size
    k  = 2*np.pi*fftfreq(Nx, d=dx)
    rho_hat     = fft(rho)
    phi_hat     = np.zeros_like(rho_hat, dtype=complex)
    nz          = k != 0
    phi_hat[nz] = -rho_hat[nz] / (k[nz]**2)
    lap_phi     = np.real(ifft(-(k**2) * phi_hat))
    res         = lap_phi + rho
    return np.linalg.norm(res) / (np.linalg.norm(rho) + 1e-14)

def efield_energy(rho, dx):
    """Return ½∫E² dx with E from periodic Poisson solve."""
    Nx = rho.size
    k  = 2*np.pi*fftfreq(Nx, d=dx)
    rho_hat     = fft(rho)
    phi_hat     = np.zeros_like(rho_hat, dtype=complex)
    nz          = k != 0
    phi_hat[nz] = -rho_hat[nz] / (k[nz]**2)
    E_hat       = -1j * k * phi_hat
    E           = np.real(ifft(E_hat))
    return 0.5 * np.sum(E**2) * dx

def entropy_2d(f, dx, dv):
    """Boltzmann entropy ∫ f log f dx dv."""
    return np.sum(f * np.log(np.clip(f, 1e-14, None))) * dx * dv

# ═══════════════════════════════════════════════════════════════
# 2 ▸  FINE‑GRID (128×128) PHYSICS STEPPER   (unchanged)
# ═══════════════════════════════════════════════════════════════
NXF = NVF = 256
DT   = 0.010
#run42_eps0.005_beta0.175
class FineVlasovStepper:
    """Single Vlasov–Poisson step on a 128×128 phase‑space grid."""
    def __init__(self, dt=DT):
        self.dt       = dt
        self.te_order = 344
        self.fd_order = 1
        units = UnitsConfiguration()
        n0_e = n0_i = 1.0
        T_e = T_i = 1.0
        mass_ratio = 25.0
        me, mi = 1.0, mass_ratio
        self.elc = ElcConfiguration(n0=n0_e, mass=me, T=T_e, units_config=units)
        self.ion = IonConfiguration(n0=n0_i, mass=mi, T=T_i, units_config=units)
        vth_e = self.elc.vth; lamD  = self.elc.lamD; k0 = 0.175 / lamD
        X_BOUNDS  = (-np.pi / k0, np.pi / k0)
        VE_BOUNDS = (-6.0*vth_e, 6.0*vth_e)
        X = Coordinate("X", CoordinateType.X)
        V = Coordinate("V", CoordinateType.X)
        ax_x  = Axis(NXF, coordinate=X, xpts=np.linspace(*X_BOUNDS,  NXF, endpoint=False))
        ax_ve = Axis(NVF, coordinate=V, xpts=np.linspace(*VE_BOUNDS, NVF, endpoint=False))
        ax_vi = Axis(NVF, coordinate=V, xpts=np.linspace(*VE_BOUNDS, NVF, endpoint=False))
        self.ax_x, self.ax_ve, self.ax_vi = ax_x, ax_ve, ax_vi
        self.grid_X = Grid("X",   (ax_x,))
        self.grid_e = Grid("XVe", (ax_x, ax_ve))
        self.grid_i = Grid("XVi", (ax_x, ax_vi))
        self.deriv_x  = DerivativeConfiguration(BCType.PERIODIC,    order=1, fd_type=FDType.CENTER)
        self.deriv_ve = DerivativeConfiguration(BCType.ZEROGRADIENT,order=1, fd_type=FDType.CENTER)
        self.lapl_mpo = self.grid_X.laplacian_mpo({ax_x: self.deriv_x})
        bc00          = np.zeros((1,NXF)); bc00[0,-1] = 1.0
        self.bc       = (bc00, None)
        self.coords_x  = CartesianCoordinateSpace("X",  ax_x)
        self.coords_ve = CartesianCoordinateSpace("Ve", ax_ve)
        self.coords_vi = CartesianCoordinateSpace("Vi", ax_vi)

    def step(self, fe, fi):
        fe_gtn = self.grid_e.make_gridTN(fe.numpy().ravel()); fe_gtn.ax_deriv_configs = {self.ax_x:self.deriv_x, self.ax_ve:self.deriv_ve}
        fi_gtn = self.grid_i.make_gridTN(fi.numpy().ravel()); fi_gtn.ax_deriv_configs = {self.ax_x:self.deriv_x, self.ax_vi:self.deriv_ve}
        ρe = fe_gtn.integrate(integ_axes=[self.ax_ve], new_grid=self.grid_X); ρe.scalar_multiply(-1.0, inplace=True)
        ρi = fi_gtn.integrate(integ_axes=[self.ax_vi], new_grid=self.grid_X)
        ρ  = ρi.add(ρe); ρ.ax_deriv_configs = {self.ax_x:self.deriv_x}
        V_gtn = ρ.solve(self.lapl_mpo)
        vp = VlasovPoisson(
            ScalarField("fe", self.grid_e, data=fe_gtn, is_sqrt=False),
            ScalarField("fi", self.grid_i, data=fi_gtn, is_sqrt=False),
            ScalarField("ϕ",  self.grid_X, data=V_gtn,  is_sqrt=False),
            coords_x=self.coords_x, coords_ve=self.coords_ve, coords_vi=self.coords_vi,
            elc_params=self.elc, ion_params=self.ion,
            upwind=True, potential_boundary_conditions=self.bc,
            te_order=self.te_order, evolve_ion=False
        ).next_time_step(self.dt, is_first_time_step=False)
        return (torch.from_numpy(vp.fe.get_comp_data()).float(),
                torch.from_numpy(vp.fi.get_comp_data()).float())

# ═══════════════════════════════════════════════════════════════
# 3 ▸  LOAD INITIAL HISTORY   (unchanged)
# ═══════════════════════════════════════════════════════════════

def load_initial(traj_idx, hist_len):
    fe_full = np.load(PATH_FINE_E, mmap_mode="r")[traj_idx]
    fi_full = np.load(PATH_FINE_I, mmap_mode="r")[traj_idx]
    return (torch.from_numpy(fe_full[:hist_len].copy()).float(),
            torch.from_numpy(fi_full[:hist_len].copy()).float())

# ═══════════════════════════════════════════════════════════════
# 4 ▸  MAIN ROLL‑OUT  •  now with Lyapunov analysis added
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def rollout(traj_idx=42, hist_len=101, n_future=1000,
            out_dir="./eval_frames_compare_all"):

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- instantiate + load networks ----------
    net_map = {}
    net_funet = SuperResUNet(in_channels=hist_len, final_scale=4).to(device)
    net_funet.load_state_dict(torch.load(PATH_WEIGHTS_FUNET, map_location=device))
    net_funet.eval()
    net_map["FUnet"] = {"net": net_funet,
                        "μσ": load_stats(PATH_STATS_FUNET, device, hist_len)}

    # net_fnosr = FNO2dSR(in_ch=hist_len, width=64, modes1=16,
    #                     modes2=16, upscale_factor=4).to(device)
    # net_fnosr.load_state_dict(torch.load(PATH_WEIGHTS_FNO, map_location=device))
    # net_fnosr.eval()
    # net_map["FNO-SR"] = {"net": net_fnosr,
    #                      "μσ": load_stats(PATH_STATS_FNO, device, hist_len)}

    # μ_otp, σ_otp = load_stats(PATH_STATS_FNO_OTP, device, hist_len)
    # net_fnootp = FNO2dOTP(in_ch=hist_len, out_ch=1,
    #                       modes1=16, modes2=16, width=64).to(device)
    # net_fnootp.load_state_dict(torch.load(PATH_WEIGHTS_FNO_OTP, map_location=device))
    # net_fnootp.eval()
    # net_map["FNO-OTP"] = {"net": net_fnootp, "μσ": (μ_otp, σ_otp)}

    # μ_edsr, σ_edsr = load_stats(PATH_STATS_EDSR, device, hist_len)
    # net_edsr = EDSR(in_ch=hist_len, n_feats=64, n_res_blocks=4,
    #                 upscale_factor=4, mean=μ_edsr, std=σ_edsr).to(device)
    # net_edsr.load_state_dict(torch.load(PATH_WEIGHTS_EDSR, map_location=device))
    # net_edsr.eval()
    # net_map["EDSR"] = {"net": net_edsr, "μσ": (μ_edsr, σ_edsr)}

    # ---------- steppers & history ----------
    coarse_stepper = VlasovStepper()
    fine_stepper   = FineVlasovStepper()
    fe_hist, fi_hist = load_initial(traj_idx, hist_len)            # fine history (128×128)
    fine_hist = fe_hist.clone()                                    # for OTP auto‑regression
    fe_curr, fi_curr = fe_hist[-1].clone(), fi_hist[-1].clone()
    coarse_e_hist = fe_hist[:, ::4, ::4].unsqueeze(0)
    coarse_i_hist = fi_hist[:, ::4, ::4].unsqueeze(0)

    ce_full = np.load(PATH_COARSE_E, mmap_mode="r")[traj_idx]
    ci_full = np.load(PATH_COARSE_I, mmap_mode="r")[traj_idx]
    coarse_true_e_hist = torch.from_numpy(ce_full[:hist_len]).float().unsqueeze(0)
    coarse_true_i_hist = torch.from_numpy(ci_full[:hist_len]).float().unsqueeze(0)

    #     # ---------- Lyapunov constants (FUnet) ----------
    # print("Estimating Lyapunov constants for FUnet …")
    
    # ce0 = coarse_e_hist.clone().to(device)        # shape (1, L, 32, 32)
    # ci0 = coarse_i_hist.clone().to(device)
    
    # def coarse_step_fn(ce):                       # Jacobian of coarse stepper
    #     return coarse_stepper.step_block(ce, ci0)
    
    # κ_c = spectral_norm_fd(coarse_step_fn, ce0)   # finite‑difference ⇒ no‑grad OK
    
    # # -- switch grads ON only for this part --------------------------
    # with torch.enable_grad():
    #     ce_pred = coarse_stepper.step_block(ce0, ci0)          # (1, L, 32, 32)
    #     μ, σ = net_map["FUnet"]["μσ"]
    #     normed = ((ce_pred - μ) / σ).requires_grad_(True)      # ✅ needs gradients
    
    #     L_NN = spectral_norm_rect(
    #         lambda z: net_map["FUnet"]["net"](z),              # FUnet forward
    #         normed                                             # input with grad
    #     )
    
    # ρ = κ_c * L_NN
    # print(f"κ_c ≈ {κ_c:.3f}   L_NN ≈ {L_NN:.3f}   ρ ≈ {ρ:.3f}")
    # if ρ >= 1.0:
    #     print("   ⚠️  Predictor–corrector not strictly contractive (ρ ≥ 1).")

    # ---------- metric containers (identical to your version) ----------
    tags = list(net_map.keys()) + ["Upsampled"]
    flow_rel = {t: [] for t in tags}

    # ---------- metric containers ----------
    tags=list(net_map.keys())+["Upsampled"]
    rel_mom={t:{"ρ":[], "J":[], "M":[]} for t in tags}
    flow_rel,flow_ssim,flow_psnr,flow_spec,flow_phys=({t:[] for t in tags} for _ in range(5))
    radial_rel={t:[] for t in tags}; poisson_rel,neg_pen=({t:[] for t in tags} for _ in range(2))
    entropy_err,efield_err=({t:[] for t in tags} for _ in range(2))
    fine_stack=[]; stack_map={t:[] for t in tags}; spec_accum={t:[] for t in tags}

    eps=1e-8
    x_fine=fine_stepper.ax_x.xpts; dx=x_fine[1]-x_fine[0]
    v_fine=fine_stepper.ax_ve.xpts; dv=v_fine[1]-v_fine[0]

    # ========= ROLL‑OUT LOOP =========
    for step in trange(n_future+1,desc="rollout"):
        # --- coarse propagation
        next_step=coarse_stepper.step_block(coarse_e_hist,coarse_i_hist)[-1]
        coarse_e_hist=torch.cat([coarse_e_hist[1:], next_step.unsqueeze(0)],dim=0)
        next_true=coarse_stepper.step_block(coarse_true_e_hist,coarse_true_i_hist)[-1]
        coarse_true_e_hist=torch.cat([coarse_true_e_hist[1:], next_true.unsqueeze(0)],dim=0)

        # --- predictions
        preds={}
        for tag,info in net_map.items():
            μ,σ=info["μσ"]
            if tag=="EDSR":                          # EDSR handles norm inside
                x_in=coarse_e_hist.to(device)
                pred=(info["net"](x_in)).cpu()[0,-1]
            elif tag=="FNO-OTP":                    # OTP uses fine‑grid history autoregressively
                x_in=((fine_hist.unsqueeze(0).to(device)-μ)/σ)
                pred=(info["net"](x_in)*σ[:, :1]+μ[:, :1]).cpu()[0,0]
                # update fine history for next step
                fine_hist=torch.cat([fine_hist[1:], pred.unsqueeze(0)],dim=0)
            else:                                   # FUnet & SR‑FNO
                x_in=((coarse_e_hist.to(device)-μ)/σ)
                pred=(info["net"](x_in)*σ+μ).cpu()[0,-1]
            preds[tag]=pred
        # Upsampled baseline
        bicubic=F.interpolate(next_true.unsqueeze(0)[:, -1:],scale_factor=4,mode="bicubic",align_corners=False)[0,0].numpy()

        # --- ground‑truth fine step
        fe_next,fi_next=fine_stepper.step(fe_curr,fi_curr)

        # --- metrics --------------------------------------------------
        true_np=fe_next.numpy(); spec_true=rfft2(true_np); mass_true=true_np.sum()*dx*dv
        preds_np={t:preds[t].numpy() for t in net_map}; preds_np["Upsampled"]=bicubic
        for tag,pred_np in preds_np.items():
            flow_rel[tag].append(np.linalg.norm(pred_np-true_np)/(np.linalg.norm(true_np)+eps))
            dr=true_np.ptp()
            flow_ssim[tag].append(1.0 if dr==0 else ssim(pred_np,true_np,data_range=dr))
            flow_psnr[tag].append(300.0 if dr==0 else psnr(true_np,pred_np,data_range=dr))
            flow_spec[tag].append(np.mean(np.abs(rfft2(pred_np)-spec_true)**2))
            mass_pred=pred_np.sum()*dx*dv
            flow_phys[tag].append(abs(mass_pred-mass_true)/(abs(mass_true)+eps))
            k_r,E_true=radial_energy_spectrum(true_np); _,E_pred=radial_energy_spectrum(pred_np)
            radial_rel[tag].append(np.linalg.norm(E_pred-E_true)/(np.linalg.norm(E_true)+eps))
            spec_accum[tag].append(E_pred)

            rho_t,_,_=moments_1d(true_np); rho_p,_,_=moments_1d(pred_np)
            res_t=poisson_residual_norm(rho_t+1.0,dx); res_p=poisson_residual_norm(rho_p+1.0,dx)
            poisson_rel[tag].append(res_p/(res_t+eps))
            neg_pen[tag].append(np.abs(pred_np[pred_np<0]).sum()*dx*dv)
            S_t=entropy_2d(true_np,dx,dv); S_p=entropy_2d(pred_np,dx,dv)
            entropy_err[tag].append(abs(S_p-S_t)/(abs(S_t)+eps))
            E_t=efield_energy(rho_t+1.0,dx); E_p=efield_energy(rho_p+1.0,dx)
            efield_err[tag].append(abs(E_p-E_t)/(E_t+eps))

            for tru,pr,lbl in [(rho_t,rho_p,"ρ"),(moments_1d(true_np)[1],moments_1d(pred_np)[1],"J"),
                               (moments_1d(true_np)[2],moments_1d(pred_np)[2],"M")]:
                rel_mom[tag][lbl].append(np.linalg.norm(pr-tru)/(np.linalg.norm(tru)+eps))

        # --- store / plot every 50 ------------------------------------
        if step%50==0:
            fine_stack.append(true_np)
            for t in tags: stack_map[t].append(preds_np[t])
            import matplotlib; matplotlib.use("Agg",force=True)
            import matplotlib.pyplot as plt
            #imgs_order=[fe_next]+[preds[p] for p in ["FUnet","FNO-SR","FNO-OTP","EDSR"]]+[torch.from_numpy(bicubic)]
            imgs_order=[fe_next]+[preds[p] for p in ["FUnet"]]+[torch.from_numpy(bicubic)]
            titles=["True","FUnet","FNO‑SR","FNO‑OTP","EDSR","Upsampled"]
            plt.figure(figsize=(18,3.5))
            for j,img in enumerate(imgs_order):
            
                ax=plt.subplot(1,6,j+1); 
                if img.numpy().max() <= 0:
                    ax.axis('off')
                else:
                    ax.imshow(img.numpy(),cmap="viridis",vmin=0,origin="lower")
                ax.set_title(titles[j],fontsize=9,fontweight='bold'); ax.set_xlabel("$x$"); ax.set_ylabel("$v$")
            plt.tight_layout(); fname=os.path.join(out_dir,f"step_{step:03d}.pdf")
            plt.savefig(fname,dpi=300,bbox_inches="tight"); plt.close()

            plt.figure(figsize=(15,3))
            colors={"FUnet":"C3","FNO-SR":"C2","FNO-OTP":"C4","EDSR":"C1","Upsampled":"0.5"}
            for idx,lbl in enumerate(["ρ","J","M"]):
                axm=plt.subplot(1,3,idx+1); tru=moments_1d(true_np)[idx]
                axm.plot(x_fine,tru,'k-',lw=2,label="GT")
                for tag in ["FUnet"]:#,"FNO-SR","EDSR","Upsampled"]:
                    pr=moments_1d(preds_np[tag])[idx] if tag!="Upsampled" else moments_1d(bicubic)[idx]
                    style='--' if tag!="Upsampled" else ':'
                    axm.plot(x_fine,pr,style,color=colors[tag],lw=1.2,label=tag)
                axm.set_title(rf"${lbl}$"); axm.set_xlabel("$x$")
                if idx==0: axm.set_ylabel("value"); axm.legend(fontsize=6)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"step_{step:03d}_mom.pdf"),dpi=200); plt.close()

            plt.figure(figsize=(5,3.5))
            plt.loglog(k_r,E_true,'k-',label="GT")
            styles={"FUnet":"--","FNO-SR":"-.","FNO-OTP":"-","EDSR":"-."}
            for tag in ["FUnet"]:#,"FNO-SR","EDSR"]:
                plt.loglog(k_r,radial_energy_spectrum(preds_np[tag])[1],styles[tag],label=tag)
            plt.loglog(k_r,radial_energy_spectrum(bicubic)[1],':',color='0.5',label="Upsampled")
            plt.xlabel("$k$"); plt.ylabel("$E(k)$"); plt.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"step_{step:03d}_spec.pdf"),dpi=200); plt.close()

        # --- advance history ----
        new_ci=fi_next[::4,::4].unsqueeze(0).unsqueeze(0)
        coarse_i_hist=torch.cat([coarse_i_hist[:,1:],new_ci],dim=1)
        coarse_true_i_hist=torch.cat([coarse_true_i_hist[:,1:],new_ci],dim=1)
        fe_curr,fi_curr=fe_next.clone(),fi_next.clone()

    # ═══════════════════════════════════════════════════════
    #  SUMMARY + CSV
    # ═══════════════════════════════════════════════════════
    csv_path=os.path.join(out_dir,"2d_vlasov_two-stream_phase2_metrics_summary.csv")
    with open(csv_path,"w",newline='') as fcsv:
        writer=csv.writer(fcsv); writer.writerow(["Metric","Model","Mean","Std"])
        for lbl in ["ρ","J","M"]:
            for t in tags:
                μ,σ=np.mean(rel_mom[t][lbl]),np.std(rel_mom[t][lbl])
                writer.writerow([f"rel_L2_{lbl}",t,f"{μ:.4e}",f"{σ:.1e}"])
        metric_map={"flow_rel_L2":flow_rel,"SSIM":flow_ssim,"PSNR":flow_psnr,"spectral_MSE":flow_spec,
                    "mass_err":flow_phys,"radial_L2":radial_rel,"Poisson_res":poisson_rel,
                    "neg_mass":neg_pen,"entropy_err":entropy_err,"Efield_err":efield_err}
        for m,mdict in metric_map.items():
            for t in tags:
                μ,σ=np.mean(mdict[t]),np.std(mdict[t])
                writer.writerow([m,t,f"{μ:.4e}",f"{σ:.1e}"])
    print(f"CSV summary saved to {csv_path}")
    # # ---------- Lyapunov‑style error envelope plot ----------
    # deltas = np.array(flow_rel["FUnet"])         # relative L2 per step
    # steps  = np.arange(len(deltas))
    # C_OTP  = deltas[0]
    # if len(deltas) > 1:
    #     r_emp = np.mean(deltas[1:11] / (deltas[:10] + eps))
    # else:
    #     r_emp = ρ

    # plt.figure(figsize=(6, 3.5))
    # plt.plot(steps, np.log10(deltas), 'o-', label="empirical error")
    # plt.plot(steps,
    #          np.log10(C_OTP * (r_emp ** steps)),
    #          'k--', lw=2,
    #          label=r"$C_{\mathrm{OTP}}\tilde r^{\,n}$")
    # if ρ < 1.0:
    #     plt.plot(steps,
    #              np.log10(C_OTP * (ρ ** steps)),
    #              'r:', lw=2,
    #              label=r"$C_{\mathrm{OTP}}\rho^{n}$")
    # plt.xlabel("roll‑out step $n$")
    # plt.ylabel(r"$\log_{10}\|e_n\|_2$")
    # plt.title("Lyapunov bound vs. empirical error (FUnet)")
    # plt.grid(True); plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, "lyapunov_errors.pdf"), dpi=200)
    # plt.close()
    #  # ---------- NEW: relative L2‑error vs. time plot ----------
    # import matplotlib.pyplot as plt
    # steps = np.arange(len(flow_rel["FUnet"]))            # x‑axis
    # plt.figure(figsize=(6.5, 3.5))
    # for tag in tags:
    #     plt.plot(steps, flow_rel[tag], label=tag)
    # plt.xlabel("Roll‑out step")
    # plt.ylabel(r"relative $L_2$ error")
    # plt.title("Evolution of flow‑level relative $L_2$ error")
    # plt.legend(fontsize=8)
    # plt.tight_layout()
    # plt.ylim([1e-2, 1e0])
    # plt.savefig(os.path.join(out_dir, "rel_L2_over_time.pdf"), dpi=200)
    # plt.close()
    # ----------------------------------------------------------
    # ---------- POD analysis ----------
    def fit_pod(stack,n=6):
        arr=np.array(stack).reshape(len(stack),-1); pca=PCA(n_components=n); pca.fit(arr); return pca
    pca_map={"Fine":fit_pod(fine_stack)}
    for t in ["FUnet","FNO-SR","EDSR","Upsampled"]: pca_map[t]=fit_pod(stack_map[t])
    energy=pca_map["Fine"].explained_variance_ratio_; modes=len(energy)
    cos=lambda a,b:np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    score={t:{"cos":[abs(cos(pca_map[t].components_[k],pca_map["Fine"].components_[k])) for k in range(modes)]}
           for t in ["FUnet","FNO-SR","EDSR","Upsampled"]}
    for t in ["FUnet","FNO-SR","EDSR","Upsampled"]:
        score[t]["weighted"]=float(np.sum(energy*np.array(score[t]["cos"])))
    import matplotlib.pyplot as plt
    H,W=fine_stack[0].shape
    tag_plot=["Fine","FUnet","FNO-SR","EDSR","Upsampled"]
    fig,ax=plt.subplots(modes,len(tag_plot),figsize=(2.6*len(tag_plot),2.6*modes))
    for k in range(modes):
        comp_ref=pca_map["Fine"].components_[k]; vmax=np.max(np.abs(comp_ref))
        ax[k,0].text(-0.10,0.5,f"Mode {k+1}\n({energy[k]*100:.1f}%)",
                     transform=ax[k,0].transAxes,ha='right',va='center',fontsize=10,fontweight='bold',rotation=90)
        for j,t in enumerate(tag_plot):
            comp=pca_map[t].components_[k].reshape(H,W)
            comp=np.sign(np.dot(comp.reshape(-1),comp_ref))*comp/vmax
            ax[k,j].imshow(comp,cmap="seismic",vmin=-1,vmax=1); ax[k,j].axis("off")
            if k==0: ax[k,j].set_title(t,fontsize=12,fontweight='bold')
            if t!="Fine":
                ax[k,j].text(0.5,0.05,rf"$\cos={score[t]['cos'][k]:.2f}$",
                             transform=ax[k,j].transAxes,ha='center',va='bottom',
                             fontsize=8,color='white',
                             bbox=dict(boxstyle="round,pad=0.2",fc="black",alpha=0.55))
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"POD_mode_comparison.pdf"),dpi=300,bbox_inches="tight"); plt.close()

    # # ---------- averaged radial spectra ----------
    # k_r,_=radial_energy_spectrum(fine_stack[0])
    # E_true_mean=np.mean([radial_energy_spectrum(f)[1] for f in fine_stack],axis=0)
    # plt.figure(figsize=(5,3.5))
    # plt.loglog(k_r,E_true_mean,'k-',label="GT")
    # style_map={"FUnet":"--","FNO-SR":"-.","FNO-OTP":"-","EDSR":"-."}
    # for t,s in style_map.items():
    #     plt.loglog(k_r,np.mean(np.vstack(spec_accum[t]),axis=0),s,label=t)
    # plt.loglog(k_r,np.mean(np.vstack(spec_accum["Upsampled"]),axis=0),':',color='0.5',label="Upsampled")
    # plt.xlabel("$k$"); plt.ylabel(r"$\langle E(k)\rangle_t$"); plt.legend(fontsize=7)
    # plt.tight_layout(); plt.savefig(os.path.join(out_dir,"radial_spectrum_avg.pdf"),dpi=200); plt.close()

# ═══════════════════════════════════════════════════════════════
# 5 ▸  CLI
# ═══════════════════════════════════════════════════════════════
if __name__=="__main__":
    ap=argparse.ArgumentParser("Phase‑2 multi‑model evaluation roll‑out (1D1V)")
    ap.add_argument("--traj",type=int,default=42)
    ap.add_argument("--history",type=int,default=200)
    ap.add_argument("--future",type=int,default=1000)
    ap.add_argument("--out",type=str,default="./figures/2d_vlasov_two-stream_test-high-res")
    args=ap.parse_args()
    rollout(traj_idx=args.traj,hist_len=args.history,n_future=args.future,out_dir=args.out)