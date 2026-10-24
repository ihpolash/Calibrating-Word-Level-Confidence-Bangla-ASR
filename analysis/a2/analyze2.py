# Standard-Bangla -> dialect calibration transfer, single-dialect transfer, prior shift, accept-side coverage. CPU only.
import glob, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr
rng = np.random.default_rng(0); UNSEEN = ["comilla", "noakhali"]; NEAR = 0.34
FEATS = ["mean_p", "min_p", "first_p", "last_p", "mean_logp", "mean_ent", "max_ent", "mean_margin", "n_tok", "relpos", "nhyp", "wlen"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

def load_all(m):
    fs = glob.glob(f"/kaggle/input/**/feat_{m}/words_*.csv", recursive=True)
    d = pd.concat([pd.read_csv(f).assign(src=f.split("/")[3]) for f in fs], ignore_index=True)
    d["relpos"] = d.pos / d.nhyp.clip(lower=1); d["wlen"] = d.word.astype(str).str.len()
    d["strict"] = d.err; d["lenient"] = ((d.err == 1) & (d.lev_ratio > NEAR)).astype(int)
    d = d.fillna({"mean_ent": 0, "max_ent": 0, "mean_margin": 0})
    cv = d[d.region == "standard"]; dia = d[d.region != "standard"]
    dia = dia[(dia.split == "test") | dia.region.isin(UNSEEN)]
    if m == "reg": dia = dia[dia.region != "rangpur"]
    return cv, dia

def ece(c, y, bins=15):
    c, y = np.asarray(c), np.asarray(y, float); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        k = (c > lo) & (c <= lo + 1 / bins)
        if k.any(): e += k.mean() * abs(y[k].mean() - c[k].mean())
    return e
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def met(conf, err):
    conf, err = np.asarray(conf), np.asarray(err); o = np.argsort(conf)
    return dict(n=len(err), err=err.mean(), ECE=ece(conf, 1 - err), AUROC=roc_auc_score(err, 1 - conf) if 0 < err.mean() < 1 else np.nan,
                resid20=err[o[int(.2 * len(err)):]].mean())
def fit(method, tr, lab):
    y = 1 - tr[lab].values
    if method == "platt": 
        m = LogisticRegression(C=1e6).fit(lg(tr.mean_p.values)[:, None], y); return lambda te: m.predict_proba(lg(te.mean_p.values)[:, None])[:, 1]
    if method == "gbm":
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=0).fit(tr[FEATS], y); return lambda te: m.predict_proba(te[FEATS])[:, 1]
    if method == "raw": return lambda te: te.mean_p.values
def prior_shift(p, pi_s, pi_t):   # p = P(correct); pi = P(correct) base rates
    return 1 / (1 + np.exp(-(lg(p) + np.log(pi_t / (1 - pi_t)) - np.log(pi_s / (1 - pi_s)))))

R, M1, ACC = [], [], []
for m in ["std", "reg", "xlsr", "large"]:
    try: cv, dia = load_all(m)
    except Exception as e: print("skip", m, e); continue
    print(m, "standard words:", len(cv), "err=%.3f" % cv.strict.mean(), "| dialect words:", len(dia), "err=%.3f" % dia.strict.mean(), flush=True)
    if len(cv) < 1000: continue
    regs = sorted(dia.region.unique())
    for lab in ["strict", "lenient"]:
        # A. calibrator trained ONLY on Standard Bangla, applied to each dialect (and to dialects pooled)
        for meth in ["raw", "platt", "gbm", "platt+prior", "gbm+prior"]:
            base = meth.split("+")[0]; f = fit(base, cv, lab); pis = 1 - cv[lab].mean()
            P = pd.Series(index=dia.index, dtype=float)
            for r in regs:
                te = dia[dia.region == r]; p = f(te)
                if meth.endswith("+prior"): p = prior_shift(p, pis, 1 - te[lab].mean())     # oracle target base rate (upper bound)
                P.loc[te.index] = p; R.append(dict(model=m, label=lab, train="Standard(CV)", method=meth, region=r, **met(p, te[lab].values)))
            R.append(dict(model=m, label=lab, train="Standard(CV)", method=meth, region="ALL", **met(P.values, dia[lab].values)))
        # reverse: train on all dialects -> Standard
        for meth in ["platt", "gbm"]:
            p = fit(meth, dia, lab)(cv); R.append(dict(model=m, label=lab, train="Dialects", method=meth, region="Standard(CV)", **met(p, cv[lab].values)))
        # reference: leave-one-dialect-out gbm
        for r in regs:
            te = dia[dia.region == r]; p = fit("gbm", dia[dia.region != r], lab)(te)
            R.append(dict(model=m, label=lab, train="LODO", method="gbm", region=r, **met(p, te[lab].values)))
        # bootstrap CI (utterance resampling) of pooled Standard->dialect gbm & platt & raw
        if lab == "strict":
            fs = {k: fit(k, cv, lab) for k in ["raw", "platt", "gbm"]}; Pp = {k: pd.Series(fs[k](dia), index=dia.index) for k in fs}; uf = dia.file.unique()
            for b in range(100):
                sel = dia.file.isin(rng.choice(uf, len(uf)))
                for k in fs:
                    mm = met(Pp[k][sel].values, dia.strict[sel].values); M1.append(dict(model=m, method=k, ECE=mm["ECE"], AUROC=mm["AUROC"]))
    # B. single-dialect training: train on region s only, test on every other region (strict, gbm + platt)
    for meth in ["platt", "gbm"]:
        for s in regs:
            f = fit(meth, dia[dia.region == s], "strict")
            for t in regs:
                if t == s: continue
                te = dia[dia.region == t]; p = f(te)
                mm = met(p, te.strict.values); R.append(dict(model=m, label="strict", train=f"one:{s}", method=meth, region=t, src_err=1 - (dia[dia.region == s].strict.mean()), **mm))
    # C. accept-side: coverage / precision of words with calibrated P(correct) >= thr (LODO gbm), pooled and low-error dialects
    P = pd.Series(index=dia.index, dtype=float)
    for r in regs: P.loc[dia.region == r] = fit("gbm", dia[dia.region != r], "strict")(dia[dia.region == r])
    for name, sub in [("ALL", dia.index), ("tangail", dia.index[dia.region == "tangail"]), ("comilla", dia.index[dia.region == "comilla"])]:
        for thr in [0.5, 0.7, 0.9]:
            k = P.loc[sub] >= thr; ACC.append(dict(model=m, subset=name, thr=thr, coverage=k.mean(), precision=1 - dia.strict.loc[sub][k].mean() if k.any() else np.nan, base_acc=1 - dia.strict.loc[sub].mean()))

R = pd.DataFrame(R); R.to_csv("/kaggle/working/results2.csv", index=False)
cols = ["model", "label", "train", "method", "region", "n", "err", "ECE", "AUROC", "resid20"]
print("\n=== A. Calibrator trained on Standard Bangla (Common Voice) -> dialects, POOLED"); print(R[(R.train == "Standard(CV)") & (R.region == "ALL")][cols].round(3).to_string(index=False))
print("\n=== A'. Reference LODO gbm pooled (mean over regions weighted by n)")
x = R[(R.train == "LODO")]; 
print(x.groupby(["model", "label"]).apply(lambda g: pd.Series(dict(ECE=np.average(g.ECE, weights=g.n), AUROC=np.average(g.AUROC, weights=g.n)))).round(3).to_string())
print("\n=== A''. dialects -> Standard(CV)"); print(R[R.train == "Dialects"][cols].round(3).to_string(index=False))
print("\n=== A bootstrap 95% CI (Standard->dialect pooled, strict)")
B = pd.DataFrame(M1).groupby(["model", "method"]).agg(ECE_lo=("ECE", lambda v: v.quantile(.025)), ECE_hi=("ECE", lambda v: v.quantile(.975)), AUC_lo=("AUROC", lambda v: v.quantile(.025)), AUC_hi=("AUROC", lambda v: v.quantile(.975))).round(3); print(B.to_string())
print("\n=== A per-region (strict) ECE, Standard-trained gbm vs platt vs raw")
for k in ["raw", "platt", "gbm"]:
    print(k); print(R[(R.train == "Standard(CV)") & (R.label == "strict") & (R.method == k) & (R.region != "ALL")].pivot(index="region", columns="model", values="ECE").round(3).to_string())
print("\n=== B. Single-dialect-trained calibrator, mean over (source,target) pairs, strict")
b = R[R.train.str.startswith("one:")]; print(b.groupby(["model", "method"])[["ECE", "AUROC"]].agg(["mean", "max"]).round(3).to_string())
print("worst 5 source->target pairs by ECE (gbm):"); b2 = b[b.method == "gbm"].sort_values("ECE", ascending=False).head(5); print(b2[["model", "train", "region", "err", "ECE", "AUROC"]].round(3).to_string(index=False))
print("Spearman(|base-rate gap|, ECE) for single-dialect platt:")
for m, g in b[b.method == "platt"].groupby("model"):
    print(m, round(spearmanr(np.abs(g.src_err - (1 - g.err)), g.ECE)[0], 3))
print("\n=== C. Accept-side: words auto-accepted at calibrated P(correct)>=thr (LODO gbm)"); print(pd.DataFrame(ACC).round(3).to_string(index=False))
