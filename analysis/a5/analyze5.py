# Hidden-state confidence estimator vs 12-feature GBM (Whisper-medium Std/Reg). CPU only.
import glob, os, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
UNSEEN = ["comilla", "noakhali"]; NEAR = 0.34
FEATS = ["mean_p", "min_p", "first_p", "last_p", "mean_logp", "mean_ent", "max_ent", "mean_margin", "n_tok", "relpos", "nhyp", "wlen"]
def load(m, kind):
    dirs = glob.glob(f"/kaggle/input/**/{'bn-calib-feat-h-cv' if kind=='cv' else 'bn-calib-feat-h-dialect'}/feat_{m}", recursive=True) or glob.glob(f"/kaggle/input/**/feat_{m}", recursive=True)
    D, H = [], []
    for d in dirs:
        has_cv = "cv" in d.lower()
        if (kind == "cv") != has_cv: continue
        for f in sorted(glob.glob(d + "/words_*.csv")):
            D.append(pd.read_csv(f)); H.append(np.load(f.replace("words_", "hid_").replace(".csv", ".npy")).astype(np.float32))
    df = pd.concat(D, ignore_index=True); h = np.concatenate(H); assert len(df) == len(h), (len(df), len(h))
    df["relpos"] = df.pos / df.nhyp.clip(lower=1); df["wlen"] = df.word.astype(str).str.len(); df["strict"] = df.err
    df = df.fillna({"mean_ent": 0, "max_ent": 0, "mean_margin": 0}); return df, h
def ece(c, y, bins=15):
    c, y = np.asarray(c), np.asarray(y, float); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        k = (c > lo) & (c <= lo + 1 / bins)
        if k.any(): e += k.mean() * abs(y[k].mean() - c[k].mean())
    return e
def make(kind, tr_df, tr_h, te_df, te_h):
    y = 1 - tr_df.strict.values
    out = {}
    g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=0, early_stopping=False).fit(tr_df[FEATS], y); out["features"] = g.predict_proba(te_df[FEATS])[:, 1]
    pca = PCA(64, random_state=0).fit(tr_h[::3]); Xtr = np.hstack([tr_df[FEATS].values, pca.transform(tr_h)]); Xte = np.hstack([te_df[FEATS].values, pca.transform(te_h)])
    g2 = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=0, early_stopping=False).fit(Xtr, y); out["features+hidden"] = g2.predict_proba(Xte)[:, 1]
    return out
rows = []
for m in ["std", "reg"]:
    dia, hd = load(m, "dialect"); cv, hc = load(m, "cv")
    keep = ((dia.split == "test") | dia.region.isin(UNSEEN)).values & (~(dia.region == "rangpur").values if m == "reg" else True)
    dia, hd = dia[keep].reset_index(drop=True), hd[keep]; print(m, len(dia), len(cv), flush=True)
    regs = sorted(dia.region.unique())
    for r in regs:                                           # leave-one-dialect-out
        te = (dia.region == r).values; P = make("lodo", dia[~te], hd[~te], dia[te], hd[te])
        for k, p in P.items(): rows.append(dict(model=m, setting="LODO", est=k, region=r, n=te.sum(), ECE=ece(p, 1 - dia.strict[te]), AUROC=roc_auc_score(dia.strict[te], 1 - p)))
    P = make("std2dia", cv, hc, dia, hd)                      # Standard (Common Voice) -> dialects
    for r in regs:
        te = (dia.region == r).values
        for k, p in P.items(): rows.append(dict(model=m, setting="Standard->dialect", est=k, region=r, n=te.sum(), ECE=ece(p[te], 1 - dia.strict[te]), AUROC=roc_auc_score(dia.strict[te], 1 - p[te])))
R = pd.DataFrame(rows); R.to_csv("/kaggle/working/hidden_baseline.csv", index=False)
w = lambda g: pd.Series(dict(ECE=np.average(g.ECE, weights=g.n), AUROC=np.average(g.AUROC, weights=g.n)))
print(R.groupby(["model", "setting", "est"]).apply(w).round(3).to_string())
