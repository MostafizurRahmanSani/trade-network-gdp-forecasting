"""Training-stability and checkpoint-sensitivity study for the selected STGNN: gradient clipping, cosine learning-rate schedule, epoch budget, and test
performance under different checkpoint rules (best-validation epoch, mean over +-5 epochs around it, average of the weights of the last epochs, last epoch).
Five seeds. Test curves are recorded for reporting only; no choice depends on them.
"""
import copy, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "outputs" / "stability"
CONFIGS = {"e100": dict(epochs=100), "e300": dict(epochs=300), "e300_clip": dict(epochs=300, clip=1.0), "e300_cos": dict(epochs=300, sched="cos"),
           "e300_clip_cos": dict(epochs=300, clip=1.0, sched="cos"), "e1000": dict(epochs=1000), "e1000_clip_cos": dict(epochs=1000, clip=1.0, sched="cos")}
K = 10


def run(args):
    cname, seed = args; f = OUT / f"{cname}_s{seed}.json"
    if f.exists(): return f.name
    import torch, torch.nn.functional as F
    torch.set_num_threads(1)
    import train_neural, panel_data
    from gdp_stgnn.data.graph_data import unpack_graphs
    from gdp_stgnn.utils import set_seed
    import selection
    nm, feat, hid, wd = selection.final_config("STGNN"); lr0 = float(nm.split("#lr")[1]) if "#lr" in nm else 1e-3
    cfg = CONFIGS[cname]; d = panel_data.build(feat); train_neural.EDGE_DIM = 13; set_seed(seed)
    model = train_neural.build_model("STGNN", d["train"][0][0][0].x.size(1), hid)
    opt = torch.optim.Adam(model.parameters(), lr=lr0, weight_decay=float(wd))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"]) if cfg.get("sched") == "cos" else None
    prep = lambda data: [(unpack_graphs(gs), y) for gs, y in data]
    tr, va, te = prep(d["train"]), prep(d["val"]), prep(d["test"])

    def mae(data):
        model.eval(); e = []
        with torch.no_grad():
            for (xs, ei, ea), y in data: e.append(torch.abs(model(xs, ei, ea) - y).numpy().ravel())
        return float(np.concatenate(e).mean())
    vcurve, tcurve, states = [], [], []
    for ep in range(cfg["epochs"]):
        model.train()
        for (xs, ei, ea), y in tr:
            opt.zero_grad(); F.mse_loss(model(xs, ei, ea), y).backward()
            if cfg.get("clip"): torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
        if sched: sched.step()
        vcurve.append(mae(va)); tcurve.append(mae(te))
        if ep >= cfg["epochs"] - K: states.append(copy.deepcopy(model.state_dict()))
    v, t = np.array(vcurve), np.array(tcurve); b = int(v.argmin())
    avg = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}; model.load_state_dict({k: avg[k].to(states[0][k].dtype) for k in avg})
    res = {"config": cname, "seed": seed, "best_epoch": b, "val_best": float(v[b]), "test_at_best_val": float(t[b]),
           "test_mean_pm5_around_best": float(t[max(0, b - 5): b + 6].mean()), "val_lastK_avgweights": mae(va), "test_lastK_avgweights": mae(te),
           "test_last_epoch": float(t[-1]), "val_last_epoch": float(v[-1]), "test_sd_last50": float(t[-50:].std()), "val_sd_last50": float(v[-50:].std()),
           "val_range_last100": [float(v[-100:].min()), float(v[-100:].max())], "val_curve_every10": [float(x) for x in v[::10]], "test_curve_every10": [float(x) for x in t[::10]]}
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(f, "w")); return f.name


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, s) for c in CONFIGS for s in (42, 1, 2, 3, 4)]
    with Pool(4) as p:
        for r in p.imap_unordered(run, jobs): print("done", r, flush=True)
