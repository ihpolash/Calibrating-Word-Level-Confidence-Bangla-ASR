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

PAIR, FS = [], []
for m in ["std", "reg", "xlsr", "large"]:
    cv, dia = load_all(m); regs = sorted(dia.region.unique()); print(m, len(cv), len(dia), flush=True)
    for lab in ["strict", "lenient"]:
        srcs = {"CV": cv, **{r: dia[dia.region == r] for r in regs}}
        for sname, S in srcs.items():
            f = gbm(S, lab); pis = 1 - S[lab].mean()
            for t in regs:
                if t == sname: continue
                T = dia[dia.region == t]; p = f(T); pit = 1 - T[lab].mean(); y = 1 - T[lab].values
                PAIR.append(dict(model=m, label=lab, src=sname, src_type="CV" if sname == "CV" else "dialect", tgt=t, gap=abs(pis - pit),
                                 ECE_raw=ece(p, y), ECE_prior=ece(shift(p, pis, pit), y)))
        # ---- recalibration of the Standard(CV)-trained gbm on each dialect
        f = gbm(cv, lab); pis = 1 - cv[lab].mean()
        for t in regs:
            T = dia[dia.region == t]; p = f(T); y = 1 - T[lab].values; pit = y.mean()
            em = em_prior(p, pis)
            FS.append(dict(model=m, label=lab, tgt=t, method="none", n=0, ECE=ece(p, y), pi_err=abs(pis - pit)))
            FS.append(dict(model=m, label=lab, tgt=t, method="EM-prior (label-free)", n=0, ECE=ece(shift(p, pis, em), y), pi_err=abs(em - pit)))
            FS.append(dict(model=m, label=lab, tgt=t, method="mean-conf prior (label-free)", n=0, ECE=ece(shift(p, pis, np.clip(p.mean(), 1e-3, 1 - 1e-3)), y), pi_err=abs(p.mean() - pit)))
            FS.append(dict(model=m, label=lab, tgt=t, method="oracle prior", n=0, ECE=ece(shift(p, pis, pit), y), pi_err=0.0))
            for n in [25, 50, 100, 200, 400]:
                for rep in range(8):
                    idx = rng.permutation(len(T)); a, b = idx[:n], idx[n:]
                    if len(b) < 200: continue
                    pa = (y[a].sum() + 1) / (n + 2)                                    # labelled-sample prior (Laplace)
                    FS.append(dict(model=m, label=lab, tgt=t, method="prior from n labels", n=n, ECE=ece(shift(p[b], pis, pa), y[b]), pi_err=abs(pa - pit)))
                    if 0 < y[a].sum() < n:
                        pl = LogisticRegression(C=1e6).fit(lg(p[a])[:, None], y[a]); FS.append(dict(model=m, label=lab, tgt=t, method="Platt on n labels", n=n, ECE=ece(pl.predict_proba(lg(p[b])[:, None])[:, 1], y[b]), pi_err=np.nan))
P = pd.DataFrame(PAIR); F = pd.DataFrame(FS); P.to_csv("/kaggle/working/pairs.csv", index=False); F.to_csv("/kaggle/working/fewshot.csv", index=False)

print("\n=== 1. Pairwise transfer ECE vs base-rate gap (strict). Raw = transferred calibrator; Prior = after oracle base-rate correction")
P["gap_bin"] = pd.cut(P.gap, [0, .1, .2, .3, .4, .6, 1.0])
for m, g in P[P.label == "strict"].groupby("model"):
    print(m, " Spearman(gap, ECE_raw)=%.3f  Spearman(gap, ECE_prior)=%.3f" % (spearmanr(g.gap, g.ECE_raw)[0], spearmanr(g.gap, g.ECE_prior)[0]))
    print(g.groupby(["gap_bin", "src_type"], observed=True)[["ECE_raw", "ECE_prior"]].agg(["mean", "count"]).round(3).to_string())
print("\n=== 1b. Residual ECE after prior correction, CV source vs dialect sources (strict, all pairs)")
print(P[P.label == "strict"].groupby(["model", "src_type"])[["gap", "ECE_raw", "ECE_prior"]].mean().round(3).to_string())
print("\n=== 2. Recalibrating the Standard-trained calibrator on dialects: mean per-region ECE (strict)")
g = F[F.label == "strict"].groupby(["model", "method", "n"])[["ECE", "pi_err"]].mean().round(3).reset_index()
print(g.pivot_table(index=["method", "n"], columns="model", values="ECE").round(3).to_string())
print("\nbase-rate estimation error |pi_hat - pi_true| (strict)")
print(g.pivot_table(index=["method", "n"], columns="model", values="pi_err").round(3).to_string())
print("\n=== 2b. same, lenient labels"); g = F[F.label == "lenient"].groupby(["model", "method", "n"])[["ECE"]].mean().round(3).reset_index()
print(g.pivot_table(index=["method", "n"], columns="model", values="ECE").round(3).to_string())
