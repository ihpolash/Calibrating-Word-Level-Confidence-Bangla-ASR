# (1) Is the Standard->dialect calibration failure just base-rate (label) shift?  (2) Few-shot / label-free recalibration.  CPU only.
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
rng = np.random.default_rng(0); UNSEEN = ["comilla", "noakhali"]; NEAR = 0.34
FEATS = ["mean_p", "min_p", "first_p", "last_p", "mean_logp", "mean_ent", "max_ent", "mean_margin", "n_tok", "relpos", "nhyp", "wlen"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

def load_all(m):
    d = pd.concat(map(pd.read_csv, glob.glob(f"/kaggle/input/**/feat_{m}/words_*.csv", recursive=True)), ignore_index=True)
    d["relpos"] = d.pos / d.nhyp.clip(lower=1); d["wlen"] = d.word.astype(str).str.len()
    d["strict"] = d.err; d["lenient"] = ((d.err == 1) & (d.lev_ratio > NEAR)).astype(int)
    d = d.fillna({"mean_ent": 0, "max_ent": 0, "mean_margin": 0})
    cv = d[d.region == "standard"]; dia = d[d.region != "standard"]; dia = dia[(dia.split == "test") | dia.region.isin(UNSEEN)]
    if m == "reg": dia = dia[dia.region != "rangpur"]
    return cv, dia
def ece(c, y, bins=15):
    c, y = np.asarray(c), np.asarray(y, float); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        k = (c > lo) & (c <= lo + 1 / bins)
        if k.any(): e += k.mean() * abs(y[k].mean() - c[k].mean())
    return e
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def sg(z): return 1 / (1 + np.exp(-z))
def shift(p, pi_s, pi_t): return sg(lg(p) + lg(np.array(pi_t)) - lg(np.array(pi_s)))
def gbm(tr, lab): 
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=0, early_stopping=False).fit(tr[FEATS], 1 - tr[lab].values); return lambda te: m.predict_proba(te[FEATS])[:, 1]
def platt(tr, lab):
    m = LogisticRegression(C=1e6).fit(lg(tr.mean_p.values)[:, None], 1 - tr[lab].values); return lambda te: m.predict_proba(lg(te.mean_p.values)[:, None])[:, 1]
def em_prior(p, pi_s, it=100):
    pi = pi_s
    for _ in range(it): pi = float(np.clip(shift(p, pi_s, pi).mean(), 1e-3, 1 - 1e-3))
    return pi


FS = []
for m in ["std", "reg", "xlsr", "large"]:
    cv, dia = load_all(m); regs = sorted(dia.region.unique()); print(m, flush=True)
    for lab in ["strict", "lenient"]:
        f = gbm(cv, lab); pis = 1 - cv[lab].mean()
        for t in regs:
            T = dia[dia.region == t].reset_index(drop=True); p = f(T); y = 1 - T[lab].values; pit = y.mean(); files = T.file.unique()
            for n in [25, 50, 100, 200, 400]:
                for rep in range(20):
                    order = rng.permutation(files); take, cnt = [], 0
                    for fl_ in order:                                           # whole utterances until n labelled words
                        take.append(fl_); cnt += (T.file == fl_).sum()
                        if cnt >= n: break
                    ia = np.where(T.file.isin(take))[0][:n]; ib = np.where(~T.file.isin(take))[0]
                    if len(ib) < 200 or y[ia].sum() == 0 or y[ia].sum() == len(ia): continue
                    pl = LogisticRegression(C=1e6).fit(lg(p[ia])[:, None], y[ia])
                    FS.append(dict(model=m, label=lab, tgt=t, method="Platt on n labels (utterance split)", n=n, ECE=ece(pl.predict_proba(lg(p[ib])[:, None])[:, 1], y[ib])))
                    FS.append(dict(model=m, label=lab, tgt=t, method="none", n=n, ECE=ece(p[ib], y[ib])))
pd.DataFrame(FS).to_csv("/kaggle/working/fewshot_utt.csv", index=False)
g = pd.DataFrame(FS).groupby(["label", "method", "n", "model"]).ECE.mean().unstack("model").round(3); print(g.to_string())
