# ================================================================
#  train_phase2_FUnet_vlasov_history.py
# ================================================================
import os, sys, math
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
import contextlib
CONFIGS  = [(8, 8)]


# PATH_FINE_E   = "/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_fine_256_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
# PATH_COARSE_E = "/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_coarse_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
# PATH_COARSE_I = "/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_coarse_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
# PATH_FINE_I   = "/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_fine_256_fixed_timestep_buneman_phase1_training_data_no_ion.npy"

# PATH_STATS    = "./data/2d_vlasov_funet_phase1_stats_32to128_buneman_no_ion_v1.pt"
# PATH_WEIGHTS  = "/pscratch/sd/h/hbassi/models/2d_vlasov_buneman_FUnet_best_PS_FT_32to128_t=200_no_ion.pth"
# SAVE_DIR      = "/pscratch/sd/h/hbassi/models"

# PATH_FINE_E   = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_fine_256_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy"
# PATH_COARSE_E = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_coarse_64_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy"
# PATH_COARSE_I = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_coarse_64_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy"
# PATH_FINE_I   = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_fine_256_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy"
PATH_FINE_E   = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_fine_256_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
PATH_COARSE_E = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_coarse_64_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
PATH_COARSE_I = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_coarse_64_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
PATH_FINE_I   = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_fine_256_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
PATH_STATS    = "./data/2d_vlasov_funet_phase1_stats_64to256_two-stream_v1.pt"
PATH_WEIGHTS  = "/pscratch/sd/h/hbassi/models/2d_vlasov_two-stream_FUnet_best_PS_FT_32to128_t=200_no_ion.pth"
SAVE_DIR      = "/pscratch/sd/h/hbassi/models"

# -----------------------------------------------------------------
# ⚙️  NUMERICAL CONSTANTS  – must match test_vlasov-random.py
# -----------------------------------------------------------------
# DT       = 0.010
# NX = NV  = 32
# X_BOUNDS = (-math.pi,  math.pi)
# V_BOUNDS = (-6.0,      6.0)
# TE_ORDER = 4                     # RK4 coarse integrator
# N_CH     = 101  DT       = 0.010
DT       = 0.010
NX = NV  = 64
#X_BOUNDS = (-math.pi,  math.pi)
#V_BOUNDS = (-6.0,      6.0)
TE_ORDER = 344                     # RK4 coarse integrator
N_CH     = 200                  # history length (channels)
# -----------------------------------------------------------------
# 2 ▸  DATASET  (shift target by +1 slice)
# -----------------------------------------------------------------
def load_dataset():
    """
    Returns  (coarse_e_full , coarse_i_full , fine_target_shift)
      shapes (B,102,32,32) (B,102,32,32)  (B,101,128,128)
    """
    fine_e   = np.load(PATH_FINE_E  )  # (N,T,128,128)
    coarse_e = np.load(PATH_COARSE_E)  # (N,T,32 ,32 )
    coarse_i = np.load(PATH_FINE_I)    # (N,T,32 ,32 )
    #import pdb; pdb.set_trace()
    # → tensors, keep channel dim
    ce  = torch.from_numpy(fine_e[[42, 42], :200]).float()
    ci  = torch.from_numpy(coarse_i[[42,42], :200, ::4, ::4]).float()
    tgt = torch.from_numpy(fine_e[[42, 42], 1:201]).float()  
    

    print("shapes:", ce.shape, ci.shape, tgt.shape)

    data_mean = ce.mean(dim=(0, 2, 3), keepdim=True)
    data_std  = ce.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)
    torch.save({'data_mean': data_mean, 'data_std': data_std},
               './data/2d_vlasov_funet_phase2_stats_64to256_two-stream_no_ion_v1.pt')

    return TensorDataset(ce, ci, tgt), data_mean, data_std
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
        self.dt       = 0.010      # fixed as in the updated script
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
        #k     = 0.175 / lamD
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
                    is_first_time_step=False # first T slice in each batch
                )

                # gather electron distribution at t+Δt
                pred_flat[m] = torch.from_numpy(
                    vp_sys.fe.get_comp_data()
                ).to(ce.device, dtype=ce.dtype)

        # restore original (B,T,NX,NV) shape
        return pred_flat.view(B, T, NX_, NV_)


# ================================================================
# Coordinate → Fourier features
# ================================================================
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords):                               # (B,H,W,2)
        proj = 2 * math.pi * torch.matmul(coords, self.B)    # (B,H,W,mapping_size)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return ff.permute(0, 3, 1, 2)                        # (B,2*mapping_size,H,W)

# ---------------------------------------------------------------
def get_coord_grid(batch, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid                                              # (B,H,W,2)

# ================================================================
# Fourier Neural Operator 2-D spectral layer
# ================================================================
class FourierLayer(nn.Module):
    def __init__(self, in_ch, out_ch, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        self.weight = nn.Parameter(
            torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
            / (in_ch * out_ch)
        )

    @staticmethod
    def compl_mul2d(inp, w):                                 # (B,IC,H,W) × (IC,OC,H,W)
        return torch.einsum('bixy,ioxy->boxy', inp, w)

    def forward(self, x):                                    # (B,C,H,W)  real
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.size(-1))                 # W_freq = W//2+1

        out_ft = torch.zeros(
            B, self.weight.size(1), H, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])

# ---------------------------------------------------------------
class ConvBlock(nn.Module):
    """[Conv → GELU] × 2 (keeps H×W)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU()
        )
    def forward(self, x): return self.block(x)

# ================================================================
# ↓↓↓ PixelShuffle-based up-sample block ↓↓↓
# ================================================================
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (upscale ** 2), 3, padding=1)
        self.pix  = nn.PixelShuffle(upscale)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.pix(self.conv(x)))

# ================================================================
# U-Net with Fourier bottleneck + PixelShuffle up-sampling
# NOTE: -- No pooling inside the bottleneck (only in encoder) --
# ================================================================
class SuperResUNet(nn.Module):
    def __init__(
        self,
        in_channels=101,
        lift_dim=128,
        mapping_size=64,
        mapping_scale=5.0,
        final_scale=2        # ← auto-detected from data
    ):
        super().__init__()

        # -------- lift ---------------
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)

        # -------- encoder ------------
        self.enc1 = ConvBlock(lift_dim,        lift_dim)        # keep  (Hc)
        self.enc2 = ConvBlock(lift_dim,        lift_dim * 2)    # pool → (Hc/2)
        self.pool = nn.MaxPool2d(2)

        # -------- bottleneck ---------
        # ⚠️ Removed extra pooling here
        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=32, modes2=32),
            nn.GELU()
        )

        # -------- decoder ------------
        # up1 keeps spatial dims (upscale=1) so it matches e2
        self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, upscale=1)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)                    # cat(up1,e2)

        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim)             # ×2  (Hc/2 → Hc)
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)               # cat(up2,e1)

        self.dec0 = nn.Sequential(                                       # Hc → Hc×final_scale
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, upscale=final_scale),
            ConvBlock(lift_dim // 4, lift_dim // 4)
        )

        # -------- output head --------
        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim // 4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1)                      # 11-channel output
        )

    # -----------------------------------------------------------
    def forward(self, x):                                 # (B,11,Hc,Wc) normalised
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W, x.device)
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)   # lift
        x = self.lift(x)

        e1 = self.enc1(x)               # Hc
        e2 = self.enc2(self.pool(e1))   # Hc/2

        # ---- bottleneck (no extra pooling) ----
        b  = self.bottleneck(e2)        # Hc/2

        # ---- decoder ----
        d2 = self.up1(b)                             # Hc/2  (spatially matches e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up2(d2)                            # Hc
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = self.dec0(d1)                           # Hf
        return self.out_head(d0)                     # (B,11,Hf,Wf)  normalised
# -----------------------------------------------------------------
# 5 ▸  TRAIN / VAL SPLIT
# -----------------------------------------------------------------
def get_loaders(batch=2):
    ds, mean, std = load_dataset()
    #import pdb; pdb.set_trace()
    ntr = int(0.9 * len(ds))
    tr, va = random_split(ds, [ntr, len(ds) - ntr])
    return (DataLoader(tr, batch, shuffle=True, pin_memory=True),
            DataLoader(va, batch, pin_memory=True),
            mean, std)

# -----------------------------------------------------------------
# 6 ▸  MAIN TRAIN LOOP
# -----------------------------------------------------------------
def train_phase2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr_loader, va_loader, mean, std = get_loaders(batch=1)
    mean = mean[:, :200].to(device)
    std  = std[:, :200].to(device)

    model = SuperResUNet(in_channels=200, final_scale=4).to(device)
    model.load_state_dict(torch.load(PATH_WEIGHTS, map_location=device))

    stepper = VlasovStepper()
    crit    = nn.L1Loss()
    opt     = optim.AdamW(model.parameters(), lr=5e-4)
    sch     = optim.lr_scheduler.CosineAnnealingLR(opt, 2000, 1e-6)

    train_losses = []
    val_losses   = []
    best_val     = float('inf')
    os.makedirs(SAVE_DIR, exist_ok=True)

    for ep in trange(2001):
        # ----------- train -----------------------------------------
        model.train()
        epoch_train_loss = 0.0
        for ce, ci, tgt in tr_loader:
            ce, ci, tgt = ce.to(device), ci.to(device), tgt.to(device)
            
            ce = ce[:, :, ::4, ::4]
            ce_pred = stepper.step_block(ce, ci)
            x       = (ce_pred - mean) / std
            fine_pred = model(x)
            fine_pred = fine_pred * std + mean
            loss = crit(fine_pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(tr_loader)
        train_losses.append(epoch_train_loss)

        # ----------- validation every epoch -----------------------
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for ce, ci, tgt in va_loader:
                ce, ci, tgt = ce.to(device), ci.to(device), tgt.to(device)
                ce = ce[:, :, ::4, ::4]
                ce_pred = stepper.step_block(ce, ci)
                x       = (ce_pred - mean) / std
                pred    = model(x)
                pred    = pred * std + mean
                epoch_val_loss += crit(pred, tgt).item()
        epoch_val_loss /= len(va_loader)
        val_losses.append(epoch_val_loss)

        # log & checkpoint
        if ep % 25 == 0:
            print(f"[{ep:4d}] train={epoch_train_loss:.6e}, val={epoch_val_loss:.6e}")
        # if epoch_val_loss < best_val:
        #     best_val = epoch_val_loss
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "FUnet_phase2_64to256_best_hist_two-stream_no_ion.pth"))
            print("   ↳ saved new best")

    # ----------- save loss curves -------------------------------
    losses_path = os.path.join(SAVE_DIR, "phase2_loss_history_64to256_two-stream_no_ion.pt")
    torch.save({'train_losses': train_losses, 'val_losses': val_losses},
               losses_path)
    print(f"Saved loss history to {losses_path}")

if __name__ == "__main__":
    train_phase2()
