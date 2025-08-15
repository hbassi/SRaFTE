#!/usr/bin/env python
# ===============================================================
#  train_phase2_baseline_unet_vlasov.py
#  ---------------------------------------------------------------
#  Phase‑2 training for the baseline U‑Net super‑resolution model:
#  (i) coarse electron/ion snapshots → VlasovStepper → next‑step
#  coarse electron slice (32×32);
#  (ii) baseline U‑Net refines that coarse slice to a fine
#  prediction (128×128);
#  (iii) L1 loss to the true fine grid at the same time.
#
#  Uses the mean / std and pre‑trained weights from the Phase‑1
#  baseline script (single‑channel input, ×4 bilinear upsample).
# ===============================================================

import os, sys, math, contextlib, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange

# ────────────────────────────────────────────────────────── paths
CONFIGS = [(8, 8)]

PATH_COARSE_E = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_coarse_32_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy"
PATH_COARSE_I = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_coarse_32_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy"
PATH_FINE_E   = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_fine_128_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy"

PATH_STATS   = "./logs/sr_stats.pt"                    # scalar mean / std from Phase‑1
PATH_WEIGHTS = "/pscratch/sd/h/hbassi/models/baseline_unet_sr_best.pth"
SAVE_DIR     = "/pscratch/sd/h/hbassi/models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ───────────────────────────────────────────────────── hyper‑params
HIST_LEN     = 200          # time slices per trajectory
BATCH_SIZE   = 1            # heavy Vlasov step ⇒ small batch
NUM_EPOCHS   = 1000
LR_INIT      = 3e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED         = 42
DT       = 0.005
NX = NV  = 32
torch.manual_seed(SEED); np.random.seed(SEED)

# ===============================================================
# 1 ▸ Dataset loader
# ===============================================================
def load_dataset():
    """
    Returns
    -------
    ds : TensorDataset
         (coarse_e, coarse_i, fine_target) with shapes
         (B, HIST_LEN, 32, 32), (B, HIST_LEN, 32, 32),
         (B, HIST_LEN, 128,128)
    """
    coarse_e = np.load(PATH_COARSE_E)   # (N,Tc,32,32)
    coarse_i = np.load(PATH_COARSE_I)   # (N,Tc,32,32)
    fine_e   = np.load(PATH_FINE_E)     # (N,Tf,128,128)

    # truncate / align lengths
    coarse_e = coarse_e[:, :HIST_LEN]
    coarse_i = coarse_i[:, :HIST_LEN]
    fine_e   = fine_e[:, 1:HIST_LEN+1]  # target is t+Δt

    # convert to torch
    ce  = torch.from_numpy(coarse_e).float()
    ci  = torch.from_numpy(coarse_i).float()
    tgt = torch.from_numpy(fine_e).float()

    return TensorDataset(ce, ci, tgt)

# -----------------------------------------------------------------------
# project-specific imports
# -----------------------------------------------------------------------
sys.path.append(os.path.abspath("vlasov_fd"))
from setup_.configs import *
from setup_.enums  import BCType, FDType
from axis          import Axis
from coord.coord_sys      import Coordinate, CoordinateType
from coord.cartesian      import CartesianCoordinateSpace
from grid          import Grid
from field         import ScalarField
from pde_vlasovES  import VlasovPoisson
# -----------------------------------------------------------------------


class VlasovStepper:
    """
    Phase-2 coarse-→fine time-advancer that now mirrors the solver
    settings used for the Buneman two-stream instability (“System A”).
    """

    # -------------------------------------------------------------------
    # initialisation
    # -------------------------------------------------------------------
    def __init__(self):

        # ------------- integration parameters --------------------------
        self.dt       = 0.005      # fixed as in the updated script
        self.te_order = 344         # time-expansion / Strang splitting
        self.fd_order = 1         # spatial FD order (centre diff.)

        # ------------- species / plasma constants ----------------------
        plasma_cfg = UnitsConfiguration()

        charge       = plasma_cfg.e
        eV           = plasma_cfg.eV
        permittivity = plasma_cfg.eps0

        n0_e = 1.0
        n0_i = n0_e
        T_e  = 1.0
        T_i  = T_e

        mass_ratio = 25.0
        mass_e = 1.0
        mass_i = mass_ratio * mass_e

        self.elc_cfg = ElcConfiguration(n0=n0_e, mass=mass_e,
                                        T=T_e, units_config=plasma_cfg)
        self.ion_cfg = IonConfiguration(n0=n0_i, mass=mass_i,
                                        T=T_i, units_config=plasma_cfg)

        # ------------- grid extents / resolutions ----------------------
        # These still come from the global constants NX / NV so that the
        # shape expected by the Phase-2 dataset is unchanged.
        vth_e = self.elc_cfg.vth
        lamD  = self.elc_cfg.lamD
        k     = 0.10 / lamD        # matches updated solver

        X_BOUNDS  = (-np.pi/k, np.pi/k)
        VE_BOUNDS = (-6.0 * vth_e,  6.0 * vth_e)
        VI_BOUNDS = (-6.0 * vth_e,  6.0 * vth_e)

        # coordinates and axes
        X = Coordinate('X', CoordinateType.X)
        V = Coordinate('V', CoordinateType.X)

        x_pts  = np.linspace(*X_BOUNDS,  NX, endpoint=False)
        ve_pts = np.linspace(*VE_BOUNDS, NV, endpoint=False)
        vi_pts = np.linspace(*VI_BOUNDS, NV, endpoint=False)

        self.ax_x  = Axis(NX, coordinate=X, xpts=x_pts)
        self.ax_ve = Axis(NV, coordinate=V, xpts=ve_pts)
        self.ax_vi = Axis(NV, coordinate=V, xpts=vi_pts)

        # coordinate spaces
        self.coords_x  = CartesianCoordinateSpace('X',  self.ax_x)
        self.coords_ve = CartesianCoordinateSpace('Ve', self.ax_ve)
        self.coords_vi = CartesianCoordinateSpace('Vi', self.ax_vi)

        # grids
        self.grid_X = Grid('X',   (self.ax_x,))
        self.grid_e = Grid('XVe', (self.ax_x, self.ax_ve))
        self.grid_i = Grid('XVi', (self.ax_x, self.ax_vi))

        # derivative configs (1st-order centred)
        self.deriv_x = DerivativeConfiguration(
            left_bc=BCType.PERIODIC, order=self.fd_order, fd_type=FDType.CENTER
        )
        self.deriv_ve = DerivativeConfiguration(
            left_bc=BCType.ZEROGRADIENT, order=self.fd_order, fd_type=FDType.CENTER
        )
        self.deriv_vi = DerivativeConfiguration(
            left_bc=BCType.ZEROGRADIENT, order=self.fd_order, fd_type=FDType.CENTER
        )

        # Poisson Laplacian MPO
        self.lapl_mpo = self.grid_X.laplacian_mpo(
            ax_deriv_configs={self.ax_x: self.deriv_x}
        )

        # Dirichlet-style BC vector (same trick as `bc00` in the paper)
        bc00 = np.zeros((1, self.ax_x.npts))
        bc00[0, -1] = 1.0
        self.bc = (bc00, None)

    # -------------------------------------------------------------------
    # batched one-step advance
    # -------------------------------------------------------------------
    def step_block(self, ce, ci):
        """
        Advance a (B,T,NX,NV) batch of electron/ion phase-space slices
        by **one** Δt.  Returns the updated electron tensor.

        Parameters
        ----------
        ce, ci : torch.Tensor
            Shapes (B,T,NX,NV).  Assumed real-valued.

        Returns
        -------
        ce_pred : torch.Tensor
            Same shape as `ce`, containing fₑ at t+Δt.
        """
        B, T, NX_, NV_ = ce.shape
        assert NX_ == self.ax_x.npts and NV_ == self.ax_ve.npts, \
            "Incoming tensors have incompatible grid shape."

        BT = B * T
        ce_flat  = ce.view(BT, NX_, NV_)
        ci_flat  = ci.view(BT, NX_, NV_)
        pred_flat = torch.empty_like(ce_flat)

        for m in trange(BT, desc="Vlasov stepping", leave=False):
            with open(os.devnull, 'w') as fnull, \
                 contextlib.redirect_stdout(fnull), \
                 contextlib.redirect_stderr(fnull):

                fe_np = ce_flat[m].cpu().numpy().ravel()
                fi_np = ci_flat[m].cpu().numpy().ravel()

                # ------------------------------------------------------------------
                # pack into GridTN objects with correct derivative metadata
                # ------------------------------------------------------------------
                fe_gtn = self.grid_e.make_gridTN(fe_np)
                fe_gtn.ax_deriv_configs = {self.ax_x: self.deriv_x,
                                           self.ax_ve: self.deriv_ve}
                fi_gtn = self.grid_i.make_gridTN(fi_np)
                fi_gtn.ax_deriv_configs = {self.ax_x: self.deriv_x,
                                           self.ax_vi: self.deriv_vi}

                # ------------------------------------------------------------------
                # charge density ρ = ∫ fi − ∫ fe
                # ------------------------------------------------------------------
                charge_e = fe_gtn.integrate(integ_axes=[self.ax_ve],
                                            is_sqrt=False,
                                            new_grid=self.grid_X)
                charge_e.scalar_multiply(-1.0, inplace=True)

                charge_i = fi_gtn.integrate(integ_axes=[self.ax_vi],
                                            is_sqrt=False,
                                            new_grid=self.grid_X)

                rho_gtn = charge_i.add(charge_e)
                rho_gtn.ax_deriv_configs = {self.ax_x: self.deriv_x}

                # Poisson solve (∇²ϕ = −ρ)
                V_gtn = rho_gtn.solve(self.lapl_mpo)
                V_field = ScalarField('V', self.grid_X, data=V_gtn, is_sqrt=False)

                # wrap fields
                fe_field = ScalarField('fe', self.grid_e, data=fe_gtn, is_sqrt=False)
                fi_field = ScalarField('fi', self.grid_i, data=fi_gtn, is_sqrt=False)

                # --------------------------------------------------------------
                # main Vlasov–Poisson step
                # --------------------------------------------------------------
                vp_sys = VlasovPoisson(
                    fe_field, fi_field, V_field,
                    coords_x=self.coords_x,
                    coords_ve=self.coords_ve,
                    coords_vi=self.coords_vi,
                    elc_params=self.elc_cfg,
                    ion_params=self.ion_cfg,
                    upwind=True,                       # <--– changed
                    potential_boundary_conditions=self.bc,
                    te_order=self.te_order,
                    evolve_ion=False
                )

                vp_sys = vp_sys.next_time_step(
                    self.dt,
                    compress_level=1,
                    is_first_time_step=(m < B)  # first T slice in each batch
                )

                # gather electron distribution at t+Δt
                pred_flat[m] = torch.from_numpy(
                    vp_sys.fe.get_comp_data()
                ).to(ce.device, dtype=ce.dtype)

        # restore original (B,T,NX,NV) shape
        return pred_flat.view(B, T, NX_, NV_)
# ===============================================================
# 3 ▸  Baseline U‑Net (same as Phase‑1)
# ===============================================================
def conv_block(in_ch, out_ch, k=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=k//2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU()
    )

class UNetSR(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, upscale_factor=4):
        super().__init__()
        self.upscale_factor = upscale_factor
        # Encoder
        self.enc1 = nn.Sequential(conv_block(in_ch, base_ch),
                                  conv_block(base_ch, base_ch))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(conv_block(base_ch, base_ch*2),
                                  conv_block(base_ch*2, base_ch*2))
        self.pool2 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = nn.Sequential(conv_block(base_ch*2, base_ch*4),
                                         conv_block(base_ch*4, base_ch*4))
        # Decoder
        self.up2  = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = nn.Sequential(conv_block(base_ch*4, base_ch*2),
                                  conv_block(base_ch*2, base_ch*2))
        self.up1  = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = nn.Sequential(conv_block(base_ch*2, base_ch),
                                  conv_block(base_ch, base_ch))
        self.out_head = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_head(d1)
        out = F.interpolate(out, scale_factor=self.upscale_factor,
                            mode='bilinear', align_corners=False)
        return out

# ===============================================================
# 4 ▸  Data loaders
# ===============================================================
def get_loaders():
    ds = load_dataset()
    n_tr = int(0.9 * len(ds))
    tr, va = random_split(ds, [n_tr, len(ds) - n_tr])
    return (DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True),
            DataLoader(va, batch_size=BATCH_SIZE, pin_memory=True))

# ===============================================================
# 5 ▸  Phase‑2 training loop
# ===============================================================
def train_phase2():
    stats = torch.load(PATH_STATS, map_location=DEVICE)
    mean = stats['mean'].view(1, 1, 1, 1).to(DEVICE)
    std  = stats['std'].view(1, 1, 1, 1).to(DEVICE)

    model = UNetSR(in_ch=1, base_ch=64, upscale_factor=4).to(DEVICE)
    model.load_state_dict(torch.load(PATH_WEIGHTS, map_location=DEVICE), strict=True)

    stepper = VlasovStepper()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=NUM_EPOCHS,
                                                     eta_min=1e-6)

    tr_loader, va_loader = get_loaders()
    train_losses, val_losses = [], []
    best_val = float('inf')

    for epoch in trange(NUM_EPOCHS, desc="Epochs", ascii=True):
        # ── training ──────────────────────────────────────────────
        model.train()
        running = 0.0
        for ce, ci, tgt in tr_loader:
            ce, ci, tgt = ce.to(DEVICE), ci.to(DEVICE), tgt.to(DEVICE)

            # Coarse step → next coarse prediction
            ce_pred = stepper.step_block(ce, ci)              # (B,T,32,32)
            x_in = ce_pred[:, -1].unsqueeze(1)                # (B,1,32,32)

            x_nrm = (x_in - mean) / std
            out = model(x_nrm)                                # (B,1,128,128)
            out = out * std + mean                            # undo normalisation

            loss = criterion(out, tgt[:, -1])                 # compare final slice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        tr_loss = running / len(tr_loader)
        train_losses.append(tr_loss)

        # ── validation every 25 epochs ───────────────────────────
        if epoch % 25 == 0:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for ce, ci, tgt in va_loader:
                    ce, ci, tgt = ce.to(DEVICE), ci.to(DEVICE), tgt.to(DEVICE)
                    ce_pred = stepper.step_block(ce, ci)
                    x_in = ce_pred[:, -1].unsqueeze(1)
                    x_nrm = (x_in - mean) / std
                    pred = model(x_nrm)
                    pred = pred * std + mean
                    val_running += criterion(pred, tgt[:, -1]).item()
            val_loss = val_running / len(va_loader)
            val_losses.append(val_loss)
            print(f"[{epoch:4d}] train={tr_loss:.6e} | val={val_loss:.6e}")

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': tr_loss
            }
            torch.save(ckpt, os.path.join(SAVE_DIR,
                       f"baseline_unet_phase2_ep{epoch:04d}.pth"))

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(),
                           os.path.join(SAVE_DIR,
                           "baseline_unet_phase2_best.pth"))
                print("   ↳ new best model saved")

    # Save loss curves
    torch.save({'train_losses': train_losses,
                'val_losses': val_losses},
               os.path.join(SAVE_DIR, "baseline_unet_phase2_loss_hist.pt"))

if __name__ == "__main__":
    train_phase2()
