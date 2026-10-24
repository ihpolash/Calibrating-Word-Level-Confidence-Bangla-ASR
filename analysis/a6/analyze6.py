# Review fixes: true WER, signed confidence gap, Brier, 1000-rep utterance bootstrap CIs, review baselines, reliability figure. CPU only.
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0); UNSEEN = ["comilla", "noakhali"]; NB = 1000
FEATS = ["mean_p", "min_p", "first_p", "last_p", "mean_logp", "mean_ent", "max_ent", "mean_margin", "n_tok", "relpos", "nhyp", "wlen"]
NAMES = {"std": "Whisper-med (Std)", "reg": "Whisper-med (Reg)", "large": "Whisper-large-v3", "xlsr": "XLS-R (CTC)"}
def sel(d, m):
    cv = d[d.region == "standard"]; dia = d[d.region != "standard"]; dia = dia[(dia.split == "test") | dia.region.isin(UNSEEN)]
    if m == "reg": dia = dia[dia.region != "rangpur"]
    return cv, dia
def load(m):
    d = pd.concat(map(pd.read_csv, glob.glob(f"/kaggle/input/**/feat_{m}/words_*.csv", recursive=True)), ignore_index=True)
    d["relpos"] = d.pos / d.nhyp.clip(lower=1); d["wlen"] = d.word.astype(str).str.len(); d["strict"] = d.err
    d = d.fillna({"mean_ent": 0, "max_ent": 0, "mean_margin": 0})
    u = pd.concat(map(pd.read_csv, glob.glob(f"/kaggle/input/**/feat_{m}/utt_*.csv", recursive=True)), ignore_index=True)
    cv, dia = sel(d, m); ucv, udia = sel(u, m); return cv, dia.reset_index(drop=True), ucv, udia
def ece(c, y, bins=15):
    c, y = np.asarray(c), np.asarray(y, float); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        k = (c > lo) & (c <= lo + 1 / bins)
        if k.any(): e += k.mean() * abs(y[k].mean() - c[k].mean())
    return e
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def gbm(tr): 
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=0).fit(tr[FEATS], 1 - tr.strict.values); return lambda te: m.predict_proba(te[FEATS])[:, 1]
def platt(tr):
    m = LogisticRegression(C=1e6).fit(lg(tr.mean_p.values)[:, None], 1 - tr.strict.values); return lambda te: m.predict_proba(lg(te.mean_p.values)[:, None])[:, 1]
def resid20(conf, err): o = np.argsort(conf); return err[o[int(.2 * len(err)):]].mean()
S, CI, REL, RV = [], [], [], []
fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
for ax, m in zip(axs, ["std", "reg", "large", "xlsr"]):
    cv, dia, ucv, udia = load(m); regs = sorted(dia.region.unique()); y = 1 - dia.strict.values; print(m, len(dia), flush=True)
    P = {"raw mean-p": dia.mean_p.values, "raw min-p": dia.min_p.values, "Std-trained GBM": gbm(cv)(dia)}
    pl, gl = np.zeros(len(dia)), np.zeros(len(dia))
    for r in regs:
        te = (dia.region == r).values; tr = dia[~te]; pl[te] = platt(tr)(dia[te]); gl[te] = gbm(tr)(dia[te])
    P["Platt (other dialects)"] = pl; P["GBM (other dialects)"] = gl
    # --- metric definitions: true WER vs hypothesis-word error share, per speech type
    S.append(dict(model=m, speech="dialects", WER=udia.ed.sum() / udia.nref.sum(), hyp_err_share=dia.strict.mean(), n_clips=len(udia), n_words=len(dia), mean_conf=dia.mean_p.mean()))
    S.append(dict(model=m, speech="Standard (CV)", WER=ucv.ed.sum() / ucv.nref.sum(), hyp_err_share=cv.strict.mean(), n_clips=len(ucv), n_words=len(cv), mean_conf=cv.mean_p.mean()))
    # --- signed gap (mean predicted P(correct) - accuracy) and Brier, dialect words pooled and per-region-weighted
    for k, p in P.items():
        gaps = [(p[(dia.region == r).values].mean() - y[(dia.region == r).values].mean(), (dia.region == r).sum()) for r in regs]
        brier = brier_score_loss(y, p)
        RV.append(dict(model=m, estimator=k, mean_conf=p.mean(), accuracy=y.mean(), signed_gap_pooled=p.mean() - y.mean(), signed_gap_region_min=min(g for g, _ in gaps), signed_gap_region_max=max(g for g, _ in gaps), brier=brier,
                       resid20=resid20(p, 1 - y)))
    RV.append(dict(model=m, estimator="random review", resid20=1 - y.mean()))
    # --- reliability curves (10 equal bins, dialect words pooled)
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    for k, st in [("raw mean-p", "o-"), ("GBM (other dialects)", "s-"), ("Std-trained GBM", "^-")]:
        b = np.minimum((P[k] * 10).astype(int), 9); xs = [P[k][b == i].mean() for i in range(10) if (b == i).sum() >= 30]; ys = [y[b == i].mean() for i in range(10) if (b == i).sum() >= 30]
        ax.plot(xs, ys, st, ms=3, lw=1, label=k)
    ax.set_title(NAMES[m], fontsize=9); ax.set_xlabel("predicted P(correct)", fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelsize=7)
    if m == "std": ax.set_ylabel("observed accuracy", fontsize=8); ax.legend(fontsize=6, loc="upper left")
    # --- utterance bootstrap: resample clips within each region, per-region-weighted ECE/AUROC; pooled resid@20
    groups = {r: [g.values for _, g in dia[dia.region == r].groupby("file").groups.items()] if False else [np.asarray(v) for v in dia[dia.region == r].groupby("file").indices.values()] for r in regs}
    base = {r: np.where((dia.region == r).values)[0] for r in regs}
    keys = ["raw mean-p", "raw min-p", "Std-trained GBM", "GBM (other dialects)", "Platt (other dialects)"]; acc = {k: {"ECE": [], "AUROC": []} for k in keys}; diff, rs = [], {"GBM (other dialects)": [], "raw min-p": [], "random": []}
    for b in range(NB):
        idxs = {}
        for r in regs:
            arrs = [base[r][a] for a in groups[r]]; pick = rng.integers(0, len(arrs), len(arrs)); idxs[r] = np.concatenate([arrs[j] for j in pick])
        n = np.array([len(idxs[r]) for r in regs], float); er = {}
        for k in keys:
            e = np.array([ece(P[k][idxs[r]], y[idxs[r]]) for r in regs]); a = np.array([roc_auc_score(1 - y[idxs[r]], 1 - P[k][idxs[r]]) for r in regs])
            acc[k]["ECE"].append((e * n).sum() / n.sum()); acc[k]["AUROC"].append((a * n).sum() / n.sum()); er[k] = (acc[k]["ECE"][-1], acc[k]["AUROC"][-1])
        diff.append((er["Std-trained GBM"][0] - er["GBM (other dialects)"][0], er["GBM (other dialects)"][1] - er["Std-trained GBM"][1]))
        allx = np.concatenate([idxs[r] for r in regs]); ee = 1 - y[allx]
        rs["GBM (other dialects)"].append(resid20(P["GBM (other dialects)"][allx], ee)); rs["raw min-p"].append(resid20(P["raw min-p"][allx], ee)); rs["random"].append(ee.mean())
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
    full = {k: (np.average([ece(P[k][base[r]], y[base[r]]) for r in regs], weights=[len(base[r]) for r in regs]), np.average([roc_auc_score(1 - y[base[r]], 1 - P[k][base[r]]) for r in regs], weights=[len(base[r]) for r in regs])) for k in keys}
    for k in keys: CI.append(dict(model=m, estimator=k, ECE=full[k][0], ECE_lo=q(acc[k]["ECE"])[0], ECE_hi=q(acc[k]["ECE"])[1], AUROC=full[k][1], AUROC_lo=q(acc[k]["AUROC"])[0], AUROC_hi=q(acc[k]["AUROC"])[1]))
    d = np.array(diff); CI.append(dict(model=m, estimator="DIFF Std-trained minus other-dialects GBM", ECE=full["Std-trained GBM"][0] - full["GBM (other dialects)"][0], ECE_lo=q(d[:, 0])[0], ECE_hi=q(d[:, 0])[1],
                                       AUROC=full["GBM (other dialects)"][1] - full["Std-trained GBM"][1], AUROC_lo=q(d[:, 1])[0], AUROC_hi=q(d[:, 1])[1]))
    for k, v in rs.items(): CI.append(dict(model=m, estimator=f"resid@20 {k}", ECE=np.mean(v), ECE_lo=q(v)[0], ECE_hi=q(v)[1]))
plt.tight_layout(); plt.savefig("/kaggle/working/fig_reliability.png", dpi=170)
pd.DataFrame(S).to_csv("/kaggle/working/metrics_wer.csv", index=False); pd.DataFrame(RV).to_csv("/kaggle/working/signed_brier.csv", index=False); pd.DataFrame(CI).to_csv("/kaggle/working/bootstrap_ci.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(S).round(3).to_string()); print(pd.DataFrame(RV).round(3).to_string()); print(pd.DataFrame(CI).round(3).to_string())
