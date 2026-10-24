"""Generate every results table in the draft from the stored CSVs (run: python3 -I make_tables.py > tables.md)."""
import csv, statistics as st, collections
D = "data/"
rd = lambda f: list(csv.DictReader(open(D + f)))
fl = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else None
MODELS = [("std", "Whisper-med (Std)"), ("reg", "Whisper-med (Reg)"), ("large", "Whisper-large-v3"), ("xlsr", "XLS-R (CTC)")]
wavg = lambda rows, k: sum(fl(r, "n") * fl(r, k) for r in rows) / sum(fl(r, "n") for r in rows)

L = rd("results_lodo.csv"); S2 = rd("results_std2dialect.csv"); P = rd("pairs.csv"); F = rd("fewshot.csv")

W = {(r["model"], r["speech"]): r for r in rd("metrics_wer.csv")}
print("### Table 1. Error rates by speech type and raw confidence. WER = (substitutions + deletions + insertions) / reference words. Error share = wrong hypothesis words / hypothesis words, the quantity the calibrators predict. Share = error share; dial. = dialects; Conf. = mean raw token-probability confidence on dialect words; Gap = Conf. minus observed accuracy (positive = overconfident); ECE and AUROC use the raw mean (mean) or minimum (min) token probability, AUROC for the mean.\n")
print("| Model | WER CV | WER dial. | Share CV | Share dial. | Conf. | Gap | ECE mean | ECE min | AUROC |\n|---|---|---|---|---|---|---|---|---|---|")
for m, name in MODELS:
    dia = [r for r in L if r["model"] == m and r["label"] == "strict" and r["method"] == "raw_mean" and r["setting"] == "LODO" and r["region"] != "ALL"]
    mn = [r for r in L if r["model"] == m and r["label"] == "strict" and r["method"] == "raw_min" and r["setting"] == "LODO" and r["region"] != "ALL"]
    d, c = W[(m, "dialects")], W[(m, "Standard (CV)")]
    gap = fl(d, "mean_conf") - (1 - fl(d, "hyp_err_share"))
    print(f"| {name} | {fl(c,'WER'):.2f} | {fl(d,'WER'):.2f} | {fl(c,'hyp_err_share'):.2f} | {fl(d,'hyp_err_share'):.2f} | {fl(d,'mean_conf'):.2f} | {gap:+.2f} | {wavg(dia,'ECE'):.2f} | {wavg(mn,'ECE'):.2f} | {wavg(dia,'AUROC'):.3f} |")

print("\n### Table 2. Calibrators trained on the other dialects (leave-one-dialect-out), per-region ECE / AUROC (n-weighted over 12 regions, strict labels)\n")
meths = [("raw_mean", "raw mean-p"), ("raw_min", "raw min-p"), ("platt_mean", "Platt (mean-p)"), ("gbm", "GBM (12 features)")]
print("| Model | " + " | ".join(n for _, n in meths) + " |\n|---|" + "---|" * len(meths))
for m, name in MODELS:
    cells = []
    for k, _ in meths:
        x = [r for r in L if r["model"] == m and r["label"] == "strict" and r["method"] == k and r["setting"] == "LODO" and r["region"] != "ALL"]
        cells.append(f"{wavg(x,'ECE'):.3f} / {wavg(x,'AUROC'):.3f}")
    print(f"| {name} | " + " | ".join(cells) + " |")
print("\nPooled (single ECE over all held-out words) GBM LODO vs in-distribution 5-fold control (ECE / AUROC):\n")
print("| Model | LODO pooled | in-distribution |\n|---|---|---|")
for m, name in MODELS:
    a = [r for r in L if r["model"] == m and r["label"] == "strict" and r["method"] == "gbm" and r["region"] == "ALL"]
    d = {r["setting"]: r for r in a}
    print(f"| {name} | {fl(d['LODO'],'ECE'):.3f} / {fl(d['LODO'],'AUROC'):.3f} | {fl(d['in-dist'],'ECE'):.3f} / {fl(d['in-dist'],'AUROC'):.3f} |")

print("\n### Table 3. Calibrator trained on Standard Bangla (Common Voice) applied to the dialects (per-region ECE / AUROC, n-weighted over regions, strict)\n")
mm = [("raw", "raw mean-p"), ("platt", "Platt"), ("gbm", "GBM"), ("gbm+prior", "GBM + oracle shift")]
print("| Model | " + " | ".join(n for _, n in mm) + " | GBM, dialect-trained |\n|---|" + "---|" * (len(mm) + 1))
for m, name in MODELS:
    cells = []
    for k, _ in mm:
        x = [r for r in S2 if r["model"] == m and r["train"] == "Standard(CV)" and r["method"] == k and r["region"] != "ALL" and r["label"] == "strict"]
        cells.append(f"{wavg(x,'ECE'):.3f} / {wavg(x,'AUROC'):.3f}")
    x = [r for r in S2 if r["model"] == m and r["train"] == "LODO" and r["label"] == "strict"]
    print(f"| {name} | " + " | ".join(cells) + f" | {wavg(x,'ECE'):.3f} / {wavg(x,'AUROC'):.3f} |")

print("\n### Table 4. Transfer ECE by error-rate gap between source and target (strict, GBM, all models)\n")
bins = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .6), (.6, 1.0)]
print("| Gap in accuracy | Source = Common Voice: n, ECE | Source = another dialect: n, ECE |\n|---|---|---|")
for lo, hi in bins:
    c = []
    for t in ["CV", "dialect"]:
        y = [fl(r, "ECE_raw") for r in P if r["label"] == "strict" and r["src_type"] == t and lo < fl(r, "gap") <= hi]
        c.append(f"{len(y)}, {st.mean(y):.3f}" if y else "0, -")
    print(f"| ({lo:.1f}, {hi:.1f}] | {c[0]} | {c[1]} |")
rho = []
def spear(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for i, j in enumerate(o): r[j] = i
        return r
    a, b = rk(x), rk(y); ma, mb = st.mean(a), st.mean(b)
    return sum((i - ma) * (j - mb) for i, j in zip(a, b)) / (sum((i - ma) ** 2 for i in a) * sum((j - mb) ** 2 for j in b)) ** .5
print("\nSpearman(gap, transfer ECE) per model: " + ", ".join(f"{name} {spear([fl(r,'gap') for r in P if r['model']==m and r['label']=='strict'],[fl(r,'ECE_raw') for r in P if r['model']==m and r['label']=='strict']):.2f}" for m, name in MODELS))

U = rd("fewshot_utt.csv")
print("\n### Table 5. Adapting the Standard-trained GBM to a target dialect: unweighted mean over regions of per-region ECE (strict; Table 3 weights regions by word count, hence 0.336 against 0.320). Platt rows: mean (SD across region-by-repetition runs, 20 repetitions per region). Labelled words are whole utterances held out from the evaluation words.\n")
print("| Adaptation | " + " | ".join(n for _, n in MODELS) + " |\n|---|" + "---|" * len(MODELS))
def row(lab, rows, n=None):
    c = []
    for m, _ in MODELS:
        y = [fl(r, "ECE") for r in rows if r["model"] == m and r["label"] == "strict" and (n is None or int(r["n"]) == n)]
        c.append(f"{st.mean(y):.3f}")
    print(f"| {lab} | " + " | ".join(c) + " |")
row("none", [r for r in F if r["method"] == "none"])
row("label-free EM error-rate estimate", [r for r in F if r["method"] == "EM-prior (label-free)"])
row("true target error rate (oracle)", [r for r in F if r["method"] == "oracle prior"])
for n in (25, 50, 100, 200, 400):
    c = []
    for m, _ in MODELS:
        y = [fl(r, "ECE") for r in U if r["method"].startswith("Platt") and r["model"] == m and r["label"] == "strict" and int(r["n"]) == n]
        c.append(f"{st.mean(y):.3f} ({st.pstdev(y):.3f})")
    print(f"| Platt scaling, {n} labelled words | " + " | ".join(c) + " |")

C = rd("clip_counts.csv")
print("\n### Table 0. Clips per region used for evaluation (test split plus, for Comilla and Noakhali, all training and validation clips)\n")
print("| Region | Test clips | Added from train/valid | Total | Used for regional checkpoint |\n|---|---|---|---|---|")
for r in C:
    tot = int(r["test"]) + int(r["added_from_train_valid"]); print(f"| {r['region']} | {r['test']} | {r['added_from_train_valid']} | {tot} | {'no (test text overlaps Ben-10)' if r['region']=='rangpur' else 'yes'} |")
print(f"| Total | {sum(int(r['test']) for r in C)} | {sum(int(r['added_from_train_valid']) for r in C)} | {sum(int(r['test'])+int(r['added_from_train_valid']) for r in C)} | |")

B = rd("bootstrap_ci.csv"); SB = rd("signed_brier.csv")
print("\n### Table 7. Review of the 20% least confident words: hypothesis-word error share left (leave-one-dialect-out, strict; 95% bootstrap interval over clips, 1000 replicates)\n")
print("| Model | Random review | Raw minimum-token probability | GBM (other dialects) |\n|---|---|---|---|")
for m, name in MODELS:
    c = []
    for k in ("random", "raw min-p", "GBM (other dialects)"):
        r = [r for r in B if r["model"] == m and r["estimator"] == f"resid@20 {k}"][0]
        c.append(f"{fl(r,'ECE'):.3f} [{fl(r,'ECE_lo'):.3f}, {fl(r,'ECE_hi'):.3f}]")
    print(f"| {name} | " + " | ".join(c) + " |")
print("\n### Table 8. Per-region ECE and AUROC with 95% bootstrap intervals (clips resampled within region, 1000 replicates; regions weighted by word count). The last row is the paired difference, Standard-trained minus other-dialect-trained GBM (ECE) and the reverse (AUROC). Percentile intervals; ECE is biased upward under resampling, so a point estimate can sit near the lower bound.\n")
print("| Model | Estimator | ECE [95% CI] | AUROC [95% CI] |\n|---|---|---|---|")
for m, name in MODELS:
    for k, lab in (("raw mean-p", "raw mean-p"), ("Platt (other dialects)", "Platt, other dialects"), ("GBM (other dialects)", "GBM, other dialects"), ("Std-trained GBM", "GBM, Standard-trained"), ("DIFF Std-trained minus other-dialects GBM", "paired difference")):
        r = [r for r in B if r["model"] == m and r["estimator"] == k][0]
        print(f"| {name} | {lab} | {fl(r,'ECE'):.3f} [{fl(r,'ECE_lo'):.3f}, {fl(r,'ECE_hi'):.3f}] | {fl(r,'AUROC'):.3f} [{fl(r,'AUROC_lo'):.3f}, {fl(r,'AUROC_hi'):.3f}] |")
print("\n### Table 9. Signed gap (mean predicted probability of correct minus accuracy; pooled over dialect words, with the range over regions) and Brier score\n")
print("| Model | Estimator | Signed gap (region range) | Brier |\n|---|---|---|---|")
for m, name in MODELS:
    for k, lab in (("raw mean-p", "raw mean-p"), ("raw min-p", "raw min-p"), ("GBM (other dialects)", "GBM, other dialects"), ("Std-trained GBM", "GBM, Standard-trained")):
        r = [r for r in SB if r["model"] == m and r["estimator"] == k][0]
        print(f"| {name} | {lab} | {fl(r,'signed_gap_pooled'):+.3f} ({fl(r,'signed_gap_region_min'):+.2f} to {fl(r,'signed_gap_region_max'):+.2f}) | {fl(r,'brier'):.3f} |")
import random
random.seed(0); print("\nBootstrap 95% interval of the gap-ECE Spearman correlation (resampling source-target pairs, 1000 replicates; pairs share clips, so the interval is optimistic): ", end="")
out = []
for m, name in MODELS:
    x = [(fl(r, "gap"), fl(r, "ECE_raw")) for r in P if r["model"] == m and r["label"] == "strict"]; bs = []
    for _ in range(1000):
        s = [random.choice(x) for _ in x]; bs.append(spear([a for a, _ in s], [b for _, b in s]))
    bs.sort(); out.append(f"{name} [{bs[24]:.2f}, {bs[974]:.2f}]")
print("; ".join(out))

print("\n### Table 3b. Calibrator trained on all dialects, applied to Standard Bangla (Common Voice): ECE / AUROC (strict)\n")
print("| Model | Platt | GBM |\n|---|---|---|")
for m, name in MODELS:
    c = []
    for k in ("platt", "gbm"):
        r = [r for r in S2 if r["model"] == m and r["train"] == "Dialects" and r["method"] == k and r["label"] == "strict"][0]
        c.append(f"{fl(r,'ECE'):.3f} / {fl(r,'AUROC'):.3f}")
    print(f"| {name} | " + " | ".join(c) + " |")
H = rd("hidden_baseline.csv")
print("\n### Table 6. Adding the decoder hidden state to the 12 features (GBM; per-region ECE / AUROC, strict)\n")
print("| Model | Training speech | 12 features | + hidden state (PCA 64) |\n|---|---|---|---|")
for m, name in MODELS[:2]:
    for setting, lab in (("LODO", "other dialects"), ("Standard->dialect", "Standard Bangla (Common Voice)")):
        c = []
        for est in ("features", "features+hidden"):
            x = [r for r in H if r["model"] == m and r["setting"] == setting and r["est"] == est]
            c.append(f"{wavg(x,'ECE'):.3f} / {wavg(x,'AUROC'):.3f}")
        print(f"| {name} | {lab} | " + " | ".join(c) + " |")
