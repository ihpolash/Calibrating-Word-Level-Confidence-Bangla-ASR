# Cross-dialect calibration analysis on cached per-word features (CPU only).
import glob, os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
rng = np.random.default_rng(0)
UNSEEN = ["comilla", "noakhali"]; NEAR = 0.34
FEATS = ["mean_p", "min_p", "first_p", "last_p", "mean_logp", "mean_ent", "max_ent", "mean_margin", "n_tok", "relpos", "nhyp", "wlen"]
BUDGETS = [0.1, 0.2, 0.3, 0.5]

def load(m):
    fs = glob.glob(f"/kaggle/input/**/feat_{m}/words_*.csv", recursive=True)
    if not fs: return None
    d = pd.concat(map(pd.read_csv, fs), ignore_index=True)
    d = d[(d.split == "test") | d.region.isin(UNSEEN)].copy()
    if m == "reg": d = d[d.region != "rangpur"]            # test text overlaps Ben-10 train (94%)
    d["relpos"] = d.pos / d.nhyp.clip(lower=1); d["wlen"] = d.word.astype(str).str.len()
    d["strict"] = d.err; d["lenient"] = ((d.err == 1) & (d.lev_ratio > NEAR)).astype(int)
    return d.fillna({"mean_ent": 0, "max_ent": 0, "mean_margin": 0})

def ece(c, y, bins=15):
    c, y = np.asarray(c), np.asarray(y, float); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        k = (c > lo) & (c <= lo + 1 / bins)
        if k.any(): e += k.mean() * abs(y[k].mean() - c[k].mean())
    return e

def metrics(conf, err):
    """conf = P(word correct); err = 1 if wrong"""
    conf, err = np.asarray(conf), np.asarray(err)
    out = dict(n=len(err), err_rate=err.mean(), ECE=ece(conf, 1 - err), Brier=brier_score_loss(1 - err, conf),
               AUROC=roc_auc_score(err, 1 - conf) if 0 < err.mean() < 1 else np.nan)
    o = np.argsort(conf)                       # lowest confidence reviewed first
    for b in BUDGETS:
        k = int(round(b * len(err))); keep = err[o[k:]]
        out[f"resid@{int(b*100)}"] = keep.mean() if len(keep) else np.nan   # error rate left after human review of b of words
    return out

def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def fit_predict(method, tr, te, lab):
    ytr = 1 - tr[lab].values
    if method == "raw_mean": return te.mean_p.values
    if method == "raw_min": return te.min_p.values
    if method == "platt_mean":
        m = LogisticRegression(C=1e6).fit(lg(tr.mean_p.values)[:, None], ytr); return m.predict_proba(lg(te.mean_p.values)[:, None])[:, 1]
    if method == "gbm":
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=0, early_stopping=False).fit(tr[FEATS], ytr)
        return m.predict_proba(te[FEATS])[:, 1]
METHODS = ["raw_mean", "raw_min", "platt_mean", "gbm"]
rows, boot = [], []
for m in ["std", "reg", "large", "xlsr"]:
    D = load(m)
    if D is None: print("missing", m); continue
    regs = sorted(D.region.unique()); print(m, len(D), "words;", {r: round(D[D.region == r].strict.mean(), 2) for r in regs}, flush=True)
    for lab in ["strict", "lenient"]:
        preds = {k: pd.Series(index=D.index, dtype=float) for k in METHODS}
        # leave-one-dialect-out: calibrator never sees the target dialect
        for r in regs:
            te, tr = D[D.region == r], D[D.region != r]
            for k in METHODS: preds[k].loc[te.index] = fit_predict(k, tr, te, lab)
        # in-distribution control: random utterance-level 5-fold (calibrator has seen every dialect)
        files = D.file.unique(); fold = dict(zip(files, rng.integers(0, 5, len(files)))); D["fold"] = D.file.map(fold)
        ind = {k: pd.Series(index=D.index, dtype=float) for k in METHODS}
        for f in range(5):
            te, tr = D[D.fold == f], D[D.fold != f]
            for k in METHODS: ind[k].loc[te.index] = fit_predict(k, tr, te, lab)
        for k in METHODS:
            for setting, P in [("LODO", preds[k]), ("in-dist", ind[k])]:
                rows.append(dict(model=m, label=lab, method=k, setting=setting, region="ALL", **metrics(P.values, D[lab].values)))
            for r in regs:
                i = D.region == r
                rows.append(dict(model=m, label=lab, method=k, setting="LODO", region=r, unseen=r in UNSEEN, **metrics(preds[k][i].values, D[lab][i].values)))
        # few-shot recalibration: Platt on raw mean_p fitted from n labelled words of the target dialect only
        for r in regs:
            te = D[D.region == r]
            for n in [25, 100, 400]:
                for rep in range(5):
                    s = te.sample(min(n, len(te) // 2), random_state=rep); rest = te.drop(s.index)
                    if s[lab].nunique() < 2: continue
                    mod = LogisticRegression(C=1e6).fit(lg(s.mean_p.values)[:, None], 1 - s[lab].values)
                    p = mod.predict_proba(lg(rest.mean_p.values)[:, None])[:, 1]
                    rows.append(dict(model=m, label=lab, method="fewshot_platt", setting=f"n={n}", region=r, **metrics(p, rest[lab].values)))
        # utterance-level bootstrap CIs of pooled LODO AUROC / ECE (gbm, platt, raw)
        for k in ["raw_mean", "platt_mean", "gbm"]:
            P = preds[k]; fl = D.file.values; uf = D.file.unique()
            for b in range(100):
                pick = rng.choice(uf, len(uf)); sel = D.file.isin(pick)
                mm = metrics(P[sel].values, D[lab][sel].values); boot.append(dict(model=m, label=lab, method=k, ECE=mm["ECE"], AUROC=mm["AUROC"]))
R = pd.DataFrame(rows); R.to_csv("/kaggle/working/results.csv", index=False)
B = pd.DataFrame(boot).groupby(["model", "label", "method"]).agg(ECE_lo=("ECE", lambda x: x.quantile(.025)), ECE_hi=("ECE", lambda x: x.quantile(.975)),
                                                                  AUC_lo=("AUROC", lambda x: x.quantile(.025)), AUC_hi=("AUROC", lambda x: x.quantile(.975))).reset_index()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
cols = ["model", "label", "method", "setting", "n", "err_rate", "ECE", "Brier", "AUROC", "resid@10", "resid@20", "resid@30", "resid@50"]
print("\n=== POOLED (LODO vs in-distribution)"); print(R[R.region == "ALL"][cols].round(3).to_string(index=False))
print("\n=== BOOTSTRAP 95% CI (pooled LODO)"); print(B.round(3).to_string(index=False))
print("\n=== UNSEEN-ASR dialects (comilla, noakhali), LODO, strict")
print(R[(R.setting == "LODO") & R.region.isin(UNSEEN) & (R.label == "strict")][["model", "region"] + cols[2:3] + ["n", "err_rate", "ECE", "AUROC", "resid@20"]].round(3).to_string(index=False))
print("\n=== per-region LODO gbm AUROC / platt ECE (strict)")
x = R[(R.setting == "LODO") & (R.region != "ALL") & (R.label == "strict")]
print(x[x.method == "gbm"].pivot(index="region", columns="model", values="AUROC").round(3).to_string())
print(x[x.method == "platt_mean"].pivot(index="region", columns="model", values="ECE").round(3).to_string())
print(x[x.method == "raw_mean"].pivot(index="region", columns="model", values="ECE").round(3).to_string())
print("\n=== few-shot Platt recalibration (mean over regions/reps), strict: ECE by n")
f = R[(R.method == "fewshot_platt") & (R.label == "strict")].groupby(["model", "setting"])[["ECE", "AUROC"]].mean().round(3); print(f.to_string())
print("\n=== Spearman(ECE, error rate) across regions, raw_mean strict:")
from scipy.stats import spearmanr
for m, g in x[x.method == "raw_mean"].groupby("model"): print(m, round(spearmanr(g.ECE, g.err_rate)[0], 3), "| AUROC vs err_rate", round(spearmanr(g.AUROC, g.err_rate)[0], 3))
