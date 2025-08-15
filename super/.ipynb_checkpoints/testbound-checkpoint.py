# ================================================================
#  train_phase2_FUnet_vlasov_history_Lyap.py   (Lyapunov diagnostics added)
# ================================================================
import os, sys, math
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import contextlib

# -----------------------------------------------------------------
# original imports / paths  .......................................
# -----------------------------------------------------------------
CONFIGS  = [(8, 8)]
PATH_FINE_E   = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_fine_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
PATH_COARSE_E = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_electron_coarse_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
PATH_COARSE_I = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_coarse_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
PATH_FINE_I   = f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_fine_128_fixed_timestep_buneman_phase1_training_data_no_ion.npy"
PATH_STATS    = "./data/2d_vlasov_funet_phase1_stats_32to128_buneman_no_ion_v1.pt"
PATH_WEIGHTS  = "/pscratch/sd/h/hbassi/models/2d_vlasov_buneman_FUnet_best_PS_FT_32to128_t=200_no_ion.pth"
SAVE_DIR      = "/pscratch/sd/h/hbassi/models"

# -----------------------------------------------------------------
# 🔧 logging / diagnostic helpers
# -----------------------------------------------------------------
LOG_EVERY = 100
rollout_errors = []      # [(epoch, [δ0, δ1, …])]

# -----------------------------------------------------------------
# ⚙️  NUMERICAL CONSTANTS (unchanged) .............................
# -----------------------------------------------------------------
DT       = 0.010
NX = NV  = 32
TE_ORDER = 344
N_CH     = 200     # history length (channels)

# -----------------------------------------------------------------
# 💡  POWER–ITERATION‑BASED SPECTRAL‑NORM ESTIMATORS
# -----------------------------------------------------------------
N_POWER_ITER = 20      # adjust for accuracy / speed trade‑off


def spectral_norm_fd(fn, x, n_iter=15, eps=1e-4):
    """
    Finite‑difference power iteration to estimate ‖J_fn(x)‖₂
    when fn is *not* differentiable by autograd.
    """
    v = torch.randn_like(x)
    v /= torch.linalg.vector_norm(v)

    for _ in range(n_iter):
        y1 = fn(x + eps * v)
        y0 = fn(x)
        Jv = (y1 - y0) / eps
        v  = Jv / (torch.linalg.vector_norm(Jv) + 1e-12)

    # one last Rayleigh quotient
    y1 = fn(x + eps * v)
    y0 = fn(x)
    Jv = (y1 - y0) / eps
    sigma = torch.dot(v.flatten(), Jv.flatten()).abs().sqrt()
    return sigma.item()


def spectral_norm_fn(fn, x):
    """
    Autograd‑based power iteration for square Jacobians.
    """
    x = x.clone().detach().requires_grad_(True)
    v = torch.randn_like(x)
    v /= torch.linalg.vector_norm(v)

    for _ in range(N_POWER_ITER):
        y = fn(x)
        (Jv,) = torch.autograd.grad(y, x, grad_outputs=v, retain_graph=True)
        v = Jv / (torch.linalg.vector_norm(Jv) + 1e-12)

    y = fn(x)
    (Jv,) = torch.autograd.grad(y, x, grad_outputs=v)
    return torch.dot(v.flatten(), Jv.flatten()).abs().sqrt().item()


def spectral_norm_rect(fn, x, n_iter=50):
    """
    Power iteration for *rectangular* Jacobians using
    mixed forward/reverse‑mode products.
    """
    y = fn(x)
    u = torch.randn_like(y)
    u /= torch.linalg.vector_norm(u)

    for _ in range(n_iter):
        (v,) = torch.autograd.grad(y, x, grad_outputs=u,
                                   retain_graph=True, create_graph=True)
        v = v / (torch.linalg.vector_norm(v) + 1e-12)

        _, Jv = torch.autograd.functional.jvp(fn, (x,), (v,), create_graph=True)
        u = Jv / (torch.linalg.vector_norm(Jv) + 1e-12)
        y = fn(x)

    _, Jv = torch.autograd.functional.jvp(fn, (x,), (v,))
    return torch.linalg.vector_norm(Jv).item()


def coarse_step_fn(stepper, ci):
    def _fn(ce):
        return stepper.step_block(ce, ci)
    return _fn

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
    Phase‑2 coarse→fine Vlasov‑Poisson time‑advancer.
    """

    def __init__(self):
        self.dt       = 0.010
        self.te_order = 344
        self.fd_order = 1

        plasma_cfg = UnitsConfiguration()
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

        vth_e = self.elc_cfg.vth
        lamD  = self.elc_cfg.lamD
        k     = 0.10 / lamD
        X_BOUNDS  = (-np.pi/k, np.pi/k)
        VE_BOUNDS = (-6.0 * vth_e,  6.0 * vth_e)
        VI_BOUNDS = (-6.0 * vth_e,  6.0 * vth_e)

        X = Coordinate('X', CoordinateType.X)
        V = Coordinate('V', CoordinateType.X)

        x_pts  = np.linspace(*X_BOUNDS,  NX, endpoint=False)
        ve_pts = np.linspace(*VE_BOUNDS, NV, endpoint=False)
        vi_pts = np.linspace(*VI_BOUNDS, NV, endpoint=False)

        self.ax_x  = Axis(NX, coordinate=X, xpts=x_pts)
        self.ax_ve = Axis(NV, coordinate=V, xpts=ve_pts)
        self.ax_vi = Axis(NV, coordinate=V, xpts=vi_pts)

        self.coords_x  = CartesianCoordinateSpace('X',  self.ax_x)
        self.coords_ve = CartesianCoordinateSpace('Ve', self.ax_ve)
        self.coords_vi = CartesianCoordinateSpace('Vi', self.ax_vi)

        self.grid_X = Grid('X',   (self.ax_x,))
        self.grid_e = Grid('XVe', (self.ax_x, self.ax_ve))
        self.grid_i = Grid('XVi', (self.ax_x, self.ax_vi))

        self.deriv_x = DerivativeConfiguration(
            left_bc=BCType.PERIODIC, order=self.fd_order, fd_type=FDType.CENTER)
        self.deriv_ve = DerivativeConfiguration(
            left_bc=BCType.ZEROGRADIENT, order=self.fd_order, fd_type=FDType.CENTER)
        self.deriv_vi = DerivativeConfiguration(
            left_bc=BCType.ZEROGRADIENT, order=self.fd_order, fd_type=FDType.CENTER)

        self.lapl_mpo = self.grid_X.laplacian_mpo(
            ax_deriv_configs={self.ax_x: self.deriv_x})

        bc00 = np.zeros((1, self.ax_x.npts))
        bc00[0, -1] = 1.0
        self.bc = (bc00, None)

    def step_block(self, ce, ci):
        """
        Advance a (B,T,NX,NV) batch by one Δt.
        """
        B, T, NX_, NV_ = ce.shape
        BT = B * T
        ce_flat  = ce.view(BT, NX_, NV_)
        ci_flat  = ci.view(BT, NX_, NV_)
        pred_flat = torch.empty_like(ce_flat)

        for m in trange(BT, desc="Vlasov stepping", leave=False):
            with open(os.devnull, 'w') as fnull, \
                 contextlib.redirect_stdout(fnull), \
                 contextlib.redirect_stderr(fnull):

                fe_np = ce_flat[m].detach().cpu().numpy().ravel()
                fi_np = ci_flat[m].detach().cpu().numpy().ravel()

                fe_gtn = self.grid_e.make_gridTN(fe_np)
                fe_gtn.ax_deriv_configs = {self.ax_x: self.deriv_x,
                                           self.ax_ve: self.deriv_ve}
                fi_gtn = self.grid_i.make_gridTN(fi_np)
                fi_gtn.ax_deriv_configs = {self.ax_x: self.deriv_x,
                                           self.ax_vi: self.deriv_vi}

                charge_e = fe_gtn.integrate(integ_axes=[self.ax_ve],
                                            is_sqrt=False, new_grid=self.grid_X)
                charge_e.scalar_multiply(-1.0, inplace=True)

                charge_i = fi_gtn.integrate(integ_axes=[self.ax_vi],
                                            is_sqrt=False, new_grid=self.grid_X)

                rho_gtn = charge_i.add(charge_e)
                rho_gtn.ax_deriv_configs = {self.ax_x: self.deriv_x}

                V_gtn = rho_gtn.solve(self.lapl_mpo)
                V_field = ScalarField('V', self.grid_X, data=V_gtn, is_sqrt=False)

                fe_field = ScalarField('fe', self.grid_e, data=fe_gtn, is_sqrt=False)
                fi_field = ScalarField('fi', self.grid_i, data=fi_gtn, is_sqrt=False)

                vp_sys = VlasovPoisson(
                    fe_field, fi_field, V_field,
                    coords_x=self.coords_x,
                    coords_ve=self.coords_ve,
                    coords_vi=self.coords_vi,
                    elc_params=self.elc_cfg,
                    ion_params=self.ion_cfg,
                    upwind=True,
                    potential_boundary_conditions=self.bc,
                    te_order=self.te_order,
                    evolve_ion=False
                )

                vp_sys = vp_sys.next_time_step(
                    self.dt, compress_level=1, is_first_time_step=False)

                pred_flat[m] = torch.from_numpy(
                    vp_sys.fe.get_comp_data()
                ).to(ce.device, dtype=ce.dtype)

        return pred_flat.view(B, T, NX_, NV_)

# ================================================================
# Coordinate → Fourier features
# ================================================================
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim=2, mapping_size=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)

    def forward(self, coords):
        proj = 2 * math.pi * torch.matmul(coords, self.B)
        ff   = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return ff.permute(0, 3, 1, 2)

def get_coord_grid(batch, h, w, device):
    xs = torch.linspace(0, 1, w, device=device)
    ys = torch.linspace(0, 1, h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid

# ================================================================
# Fourier Neural Operator 2‑D spectral layer
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
    def compl_mul2d(inp, w):
        return torch.einsum('bixy,ioxy->boxy', inp, w)

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.size(-1))

        out_ft = torch.zeros(
            B, self.weight.size(1), H, x_ft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weight[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:])

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU()
        )
    def forward(self, x):
        return self.block(x)

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, upscale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (upscale ** 2), 3, padding=1)
        self.pix  = nn.PixelShuffle(upscale)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.pix(self.conv(x)))

class SuperResUNet(nn.Module):
    def __init__(self, in_channels=101, lift_dim=128,
                 mapping_size=64, mapping_scale=5.0, final_scale=2):
        super().__init__()
        self.fourier_mapping = FourierFeatureMapping(2, mapping_size, mapping_scale)
        lifted_ch = in_channels + 2 * mapping_size
        self.lift = nn.Conv2d(lifted_ch, lift_dim, kernel_size=1)

        self.enc1 = ConvBlock(lift_dim,        lift_dim)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(lift_dim,        lift_dim * 2)

        self.bottleneck = nn.Sequential(
            ConvBlock(lift_dim * 2, lift_dim * 2),
            FourierLayer(lift_dim * 2, lift_dim * 2, modes1=32, modes2=32),
            nn.GELU()
        )

        self.up1  = PixelShuffleUpsample(lift_dim * 2, lift_dim * 2, upscale=1)
        self.dec2 = ConvBlock(lift_dim * 4, lift_dim)

        self.up2  = PixelShuffleUpsample(lift_dim, lift_dim)
        self.dec1 = ConvBlock(lift_dim * 2, lift_dim // 2)

        self.dec0 = nn.Sequential(
            PixelShuffleUpsample(lift_dim // 2, lift_dim // 4, upscale=final_scale),
            ConvBlock(lift_dim // 4, lift_dim // 4)
        )

        self.out_head = nn.Sequential(
            nn.Conv2d(lift_dim // 4, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        coords = get_coord_grid(B, H, W, x.device)
        x = torch.cat([x, self.fourier_mapping(coords)], dim=1)
        x = self.lift(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        b  = self.bottleneck(e2)

        d2 = self.up1(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up2(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = self.dec0(d1)
        return self.out_head(d0)

# -----------------------------------------------------------------
#  DATASET
# -----------------------------------------------------------------
def load_dataset():
    fine_e   = np.load(PATH_FINE_E)
    coarse_i = np.load(PATH_FINE_I)

    ce  = torch.from_numpy(fine_e[[42], :200]).float()
    ci  = torch.from_numpy(coarse_i[[42], :200, ::4, ::4]).float()
    tgt = torch.from_numpy(fine_e[[42], 1:201]).float()

    data_mean = ce.mean(dim=(0, 2, 3), keepdim=True)
    data_std  = ce.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-8)

    ds = TensorDataset(ce, ci, tgt)
    return ds, data_mean, data_std

def get_loaders(batch=1):
    ds, m, s = load_dataset()
    return DataLoader(ds, batch), DataLoader(ds, batch), m, s

# -----------------------------------------------------------------
#  MAIN TRAIN LOOP  + Lyapunov checks
# -----------------------------------------------------------------
def train_phase2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr_loader, va_loader, mean, std = get_loaders(batch=1)
    mean, std = mean.to(device), std.to(device)

    model = SuperResUNet(in_channels=N_CH, final_scale=4).to(device)
    model.load_state_dict(torch.load(PATH_WEIGHTS, map_location=device))
    model.train()

    stepper = VlasovStepper()

    with torch.no_grad():
        ce0, ci0, _ = next(iter(tr_loader))
        ce0, ci0 = ce0[:, :, ::4, ::4].to(device), ci0.to(device)

    print("Estimating κ_c …")
    kappa_c = spectral_norm_fd(coarse_step_fn(stepper, ci0), ce0)

    print("Estimating L_NN …")
    with torch.no_grad():
        ce_pred = stepper.step_block(ce0, ci0)
    normed = ((ce_pred - mean) / std).requires_grad_(True)
    L_NN = spectral_norm_rect(lambda z: model(z), normed)

    rho = kappa_c * L_NN
    print(f"κ_c ≈ {kappa_c:.3f}   L_NN ≈ {L_NN:.3f}   ρ = κ_c·L_NN ≈ {rho:.3f}")
    if rho >= 1.0:
        print("⚠️  Predictor–corrector NOT strictly contractive (ρ ≥ 1); continuing anyway.")

    crit = nn.L1Loss()
    opt  = optim.AdamW(model.parameters(), lr=5e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, 2000, 1e-6)

    best_val = float("inf")
    for ep in trange(1001, desc="Epoch"):
        ce, ci, tgt = next(iter(tr_loader))
        ce, ci, tgt = ce.to(device), ci.to(device), tgt.to(device)
        ce_c = ce[:, :, ::4, ::4]

        ce_pred = stepper.step_block(ce_c, ci)
        fine_pred = model((ce_pred - mean) / std) * std + mean

        loss = crit(fine_pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

        model.eval()
        with torch.no_grad():
            ce_val, ci_val, tgt_val = next(iter(va_loader))
            ce_val, ci_val, tgt_val = ce_val.to(device), ci_val.to(device), tgt_val.to(device)
            ce_val_c = ce_val[:, :, ::4, ::4]

            ce_pred_val = stepper.step_block(ce_val_c, ci_val)
            fine_pred_val = model((ce_pred_val - mean)/std) * std + mean
            C_OTP = (fine_pred_val - tgt_val).flatten(1).norm(2, dim=1).max().item()

            deltas = []
            u = tgt_val.clone()
            for _ in range(2):
                u_hat = model(((stepper.step_block(u[:, :, ::4, ::4], ci_val) - mean)/std))*std
                deltas.append(torch.linalg.vector_norm(u_hat - u).item())
                u = u_hat
            ratios = np.array(deltas[1:]) / np.array(deltas[:-1])
            r_emp  = ratios.mean()

        model.train()

        if ep % LOG_EVERY == 0:
            rollout_errors.append((ep, deltas.copy()))
            print(f"[{ep:03d}]  L1={loss.item():.4e} |  C_OTP={C_OTP:.2e} | r̄={r_emp:.3f}")
            if r_emp > rho + 1e-3:
                print("⚠️  Empirical decay r̄ exceeds ρ – model may be unstable!")

        if ep % 100 == 0 and loss.item() < best_val:
            best_val = loss.item()
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, "FUnet_phase2_best_with_lyap.pth"))

    if rollout_errors:
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        epochs, runs = zip(*rollout_errors)
        steps = list(range(len(runs[0])))

        fig, ax = plt.subplots()
        for ep, deltas in rollout_errors:
            ax.plot(steps, np.log10(deltas), marker='o', label=f"epoch {ep}")

        C_env = C_OTP
        r_env = runs[-1][1] / runs[-1][0]
        env = [C_env * (r_env ** k) for k in steps]
        ax.plot(steps, np.log10(env), linestyle='--', linewidth=2,
                label=r"analytic $C_{\mathrm{OTP}}\,r^{n}$")

        ax.set_xlabel("roll‑out step $n$")
        ax.set_ylabel(r"$\log_{10}\|e_n\|_2$")
        ax.set_title("Error decay per step (Phase‑2 predictor–corrector)")
        ax.legend(); ax.grid(True)
        plt.tight_layout()
        plt.savefig('lyapunov_errors.pdf')

    print("Training complete.")

# -----------------------------------------------------------------
if __name__ == "__main__":
    train_phase2()
