"""Robustness check: refit the selected models on all targets up to 2017 (training + validation years) and forecast 2018-2024 with the same fixed-origin design as the
main study. No test data are used for any choice.

Procedure
- Neural models (STGNN, GRU and LSTM, hidden size 16, weight decay 1e-3; the configurations of Table 4), seeds 42, 1, 2, 3, 4.
  Step A: the original run (targets <= 2014, validation 2015-2017, 300 epochs, best validation epoch) is repeated to record the best epoch; its test forecasts must
          equal the saved run (check).
  Step B: the model is retrained from the same seed on targets 2002-2017 for exactly (best epoch of step A + 1) epochs with the same optimiser, batch scheme and
          normalisation rule (training-period statistics, now of input years 1996-2016; training targets winsorised at the 1st/99th percentile of targets up to 2017).
          No checkpoint selection (no validation years remain).
- Random forest (36-input window, configuration chosen on validation in 19_random_forest_seeds.py, 300 trees, 5 seeds) and the pooled AR(1): refitted on targets 2002-2017
  with the same preprocessing rules; the targets <= 2014 versions are recomputed as a check.
Outputs: outputs/runs/REFIT17_<model>_s<seed>.npz/.json, outputs/baselines/REFIT17_RandomForest_s<seed>.npz, REFIT17_AR1_pooled.npz, outputs/refit2017_log.txt (via stdout).
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[k] = "1"
import json, subprocess, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs"; RUNS = OUT / "runs"; BASE = OUT / "baselines"
SEEDS = (42, 1, 2, 3, 4)
CFG = {"STGNN": (16, 1e-3), "GRU": (16, 1e-3), "LSTM": (16, 1e-3)}


def _refit_build():
    import panel_data as V
    V.FOLDS["refit17"] = dict(train_end=2016, val_end=2016, last_t=2023)
    return V.build("base", fold="refit17"), V.build("base")


def neural(model_name, seed):
    tag = f"REFIT17_{model_name}_base_h{CFG[model_name][0]}_wd1e-3_s{seed}"
    if (RUNS / f"{tag}.json").exists(): return tag
    import torch, torch.nn.functional as F
    torch.set_num_threads(1)
    import train_neural; train_neural.EDGE_DIM = 13
    from gdp_stgnn.config import stmodels_hyperparams, temporalonly_hyperparams
    from gdp_stgnn.data.graph_data import unpack_graphs
    from gdp_stgnn.data.sequence_data import build_gru_dataset_from_stgnn
    from gdp_stgnn.training import train_lstm, train_stmodel
    from gdp_stgnn.utils import set_seed
    h, wd = CFG[model_name]; t0 = time.time()
    dR, d0 = _refit_build()
    Fdim = d0["train"][0][0][0].x.size(1)
    temporal = model_name in train_neural.TEMPORAL
    set_seed(seed); m = train_neural.build_model(model_name, Fdim, h)
    if temporal:
        seq = [build_gru_dataset_from_stgnn(d0[s], 6, 1) for s in ("train", "val", "test")]
        hp = dict(temporalonly_hyperparams, n_epochs=300, print_interval=10 ** 6, weight_decay=wd)
        m, losses = train_lstm(m, tag + "_A", hp, seq[0][0], seq[0][1], seq[1][0], seq[1][1]); m.eval()
        with torch.no_grad(): tpA = m(torch.FloatTensor(seq[2][0])).numpy().ravel()
    else:
        hp = dict(stmodels_hyperparams, n_epochs=300, print_interval=10 ** 6, weight_decay=wd)
        m, losses = train_stmodel(m, tag + "_A", hp, d0["train"], d0["val"]); tpA, _ = train_neural.predict_st(m, d0["test"])
    vm = [l[4] for l in losses]; best = int(losses[int(np.argmin(vm))][0])
    orig = np.load(RUNS / f"{model_name}_base_h{h}_wd1e-3_s{seed}.npz")["test_pred"]
    check = float(np.abs(orig - tpA).max())
    set_seed(seed); m = train_neural.build_model(model_name, Fdim, h)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=wd)
    if temporal:
        from torch.utils.data import DataLoader, TensorDataset
        Xtr, Ytr = build_gru_dataset_from_stgnn(dR["train"], 6, 1); Xte, Yte = build_gru_dataset_from_stgnn(dR["test"], 6, 1)
        loader = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Ytr)), batch_size=32, shuffle=True)
        for ep in range(best + 1):
            m.train()
            for bx, by in loader:
                opt.zero_grad(); loss = F.mse_loss(m(bx), by); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0); opt.step()
        m.eval()
        with torch.no_grad(): tp = m(torch.FloatTensor(Xte)).numpy().ravel()
        tt = Yte.ravel()
    else:
        for ep in range(best + 1):
            m.train()
            for graphs, y in dR["train"]:
                xs, ei, ea = unpack_graphs(graphs); ea = None if ea[0] is None else ea
                opt.zero_grad(); loss = F.mse_loss(m(xs, ei, ea), y); loss.backward(); opt.step()
        tp, tt = train_neural.predict_st(m, dR["test"])
    np.savez(RUNS / f"{tag}.npz", test_pred=tp, test_true=tt)
    json.dump({"test_MAE": float(np.abs(tp - tt).mean()), "best_epoch_A": best, "epochs_refit": best + 1, "check_stepA_vs_saved_maxabs": check,
               "testA_MAE": float(np.abs(tpA - tt).mean()), "sec": time.time() - t0}, open(RUNS / f"{tag}.json", "w"))
    return tag


def tabular():
    import panel_data as V, tabwin as W
    from sklearn.ensemble import RandomForestRegressor
    P, iso = V.load(); order = iso.iso3.tolist()
    Pw = {y: P[P.year == y].set_index("iso3").loc[order] for y in range(1996, 2025)}; G = P.pivot(index="iso3", columns="year", values="g").loc[order]
    test = range(2017, 2024); msg = []
    for end, lab in ((2013, "CHECK"), (2016, "REFIT17")):
        tr = range(2001, end + 1)
        X = {k: np.concatenate([W.window(Pw, t) for t in v]) for k, v in (("train", tr), ("test", test))}
        Y = {k: np.concatenate([G[t + 1].values for t in v]) for k, v in (("train", tr), ("test", test))}
        lo, hi = np.percentile(Y["train"], [1, 99]); yc = np.clip(Y["train"], lo, hi); z = W.scaler(X["train"])
        rf = []
        for s in SEEDS:
            p = RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=10, max_features=0.6, n_jobs=1, random_state=s).fit(z(X["train"]), yc).predict(z(X["test"]))
            rf.append(p)
            if lab == "REFIT17": np.savez(BASE / f"REFIT17_RandomForest_s{s}.npz", test_pred=p, test_true=Y["test"])
            else: msg.append(f"RF check seed {s}: max abs diff to saved {np.abs(np.load(BASE / f'RandomForest_win_s{s}.npz')['test_pred'] - p).max():.2e}")
        A = lambda ts: np.nan_to_num(np.column_stack([np.concatenate([G[t].values for t in ts]), np.ones(len(ts) * len(order))]))
        b = np.linalg.lstsq(A(tr), yc, rcond=None)[0]; ar = A(test) @ b
        if lab == "REFIT17": np.savez(BASE / "REFIT17_AR1_pooled.npz", test_pred=ar, test_true=Y["test"], coef=b)
        else: msg.append(f"AR1 check: max abs diff to saved {np.abs(np.load(BASE / 'AR1_pooled.npz')['test_pred'] - ar).max():.2e}")
        msg.append(f"{lab}: RF test MAE {np.mean([np.abs(p - Y['test']).mean() for p in rf]):.5f}; AR1 test MAE {np.abs(ar - Y['test']).mean():.5f}; AR1 coef {b.round(4).tolist()}")
    return "; ".join(msg)


def script(name):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / name)], capture_output=True, text=True)
    (OUT / f"{name.replace('.py', '')}_stdout.txt").write_text(r.stdout + "\n" + r.stderr[-3000:], encoding="utf-8")
    return f"{name} exit {r.returncode}"


def run(job):
    t = time.time()
    try:
        kind, args = job
        out = {"neural": neural, "tabular": tabular, "script": script}[kind](*args)
        return f"{job} -> {out} ({time.time() - t:.0f}s)"
    except Exception as e:
        import traceback; return f"{job} FAILED {e!r}\n{traceback.format_exc()}"


if __name__ == "__main__":
    jobs = [("script", ("44_conformal_prediction.py",)), ("script", ("20_tabular_eleven_features.py",)), ("tabular", ())]
    jobs += [("neural", ("STGNN", s)) for s in SEEDS] + [("neural", ("GRU", s)) for s in SEEDS] + [("neural", ("LSTM", s)) for s in SEEDS]
    sel = sys.argv[1:]
    if sel: jobs = [j for j in jobs if j[0] in sel]
    with Pool(3, maxtasksperchild=3) as p:
        for m in p.imap_unordered(run, jobs): print(time.strftime("%H:%M:%S"), m, flush=True)
    print("ALL DONE", flush=True)
