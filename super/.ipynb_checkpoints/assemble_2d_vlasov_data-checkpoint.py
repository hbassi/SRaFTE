import pickle, numpy as np, os, pathlib

# ------------------------------------------------------------------
# fixed run-specific pieces
BASE_DIR = "/pscratch/sd/h/hbassi/data250506/VP1-buneman-SystemA_test"
CONFIGS  = [(8, 8)]  # (mx, my)
SEEDS    = list(range(1001, 1021))#list(range(1000,1020))#[42, 666]#list(range(1000, 1021))  # list of seeds
# patterns for coarse electron/ion and fine electron/ion
PATTERN_FE = ("fe_seed{seed}_k0.1_mr25.0_"
              "Ls{d:02d},{d:02d},{d:02d}_o1_dt0.010_T050_te344.pkl")
PATTERN_FI = ("fi_seed{seed}_k0.1_mr25.0_"
              "Ls{d:02d},{d:02d},{d:02d}_o1_dt0.010_T050_te344.pkl")
# ------------------------------------------------------------------

cg_list, ci_list, fg_list, fi_list = [], [], [], []

for mx, my in CONFIGS:
    cfg_dir = pathlib.Path(BASE_DIR) / f"mx{mx}my{my}"
    for seed in SEEDS:
        skip_seed = False
        trajs = {}  # hold all four kinds

        # -------- load COARSE (dim=5) -------------------------------
        for patt, label in [(PATTERN_FE, "cg"), (PATTERN_FI, "ci")]:
            file_name = patt.format(seed=seed, d=5)
            path = cfg_dir / file_name
            if not path.is_file():
                raise FileNotFoundError(f"missing: {path}")

            with open(path, "rb") as f:
                traj = pickle.load(f)[-1]  # full trajectory
            if traj.shape[0] != 202:
                print(f"  → skipping seed {seed}: got {traj.shape[0]} frames")
                skip_seed = True
                break
            trajs[label] = traj.real.astype(np.float32)

        if skip_seed:
            continue

        # -------- load FINE ELECTRON (dim=7) ------------------------
        path_fe = cfg_dir / PATTERN_FE.format(seed=seed, d=7)
        with open(path_fe, "rb") as f:
            traj_fe = pickle.load(f)[-1]
        if traj_fe.shape[0] != 202:
            print(f"  → skipping seed {seed}: fine electron traj mismatch")
            continue
        trajs["fg"] = traj_fe.real.astype(np.float32)

        # -------- load FINE ION (dim=7) -----------------------------
        path_fi = cfg_dir / PATTERN_FI.format(seed=seed, d=7)
        with open(path_fi, "rb") as f:
            traj_fi = pickle.load(f)[-1]
        if traj_fi.shape[0] != 202:
            print(f"  → skipping seed {seed}: fine ion traj mismatch")
            continue
        trajs["fi"] = traj_fi.real.astype(np.float32)

        # -------- append to master lists ----------------------------
        cg_list.append(trajs["cg"])
        ci_list.append(trajs["ci"])
        fg_list.append(trajs["fg"])
        fi_list.append(trajs["fi"])

print(f"Loaded  cg: {len(cg_list)}  ci: {len(ci_list)}  fg: {len(fg_list)}  fi: {len(fi_list)}")

# stack and save
cg_data = np.stack(cg_list)
ci_data = np.stack(ci_list)
fg_data = np.stack(fg_list)
fi_data = np.stack(fi_list)

np.save(f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_coarse_32_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy", cg_data)
np.save(f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_coarse_32_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy", ci_data)
np.save(f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_fine_128_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy", fg_data)
np.save(f"/pscratch/sd/h/hbassi/2d_vlasov_multi_traj_ion_fine_128_fixed_timestep_mx={CONFIGS[0][0]}_my={CONFIGS[0][1]}_no_ion_phase1_training_data.npy", fi_data)

print("saved shapes:", cg_data.shape, ci_data.shape, fg_data.shape, fi_data.shape)
