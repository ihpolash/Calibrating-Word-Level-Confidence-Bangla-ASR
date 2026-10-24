# Per-word confidence feature extraction for Bangla ASR models on RegSpeech12 (Kaggle only).
import os, re, glob, time, unicodedata
import numpy as np, pandas as pd, torch, soundfile as sf, librosa

MODELS = ["std", "reg"]          # keys of MODEL_IDS; set per kernel
LIMIT = 0                        # >0: smoke test on first LIMIT items
BS, CHUNK = 8, 200
MODEL_IDS = {
    "std":   ("whisper", "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"),
    "reg":   ("whisper", "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"),
    "large": ("whisper", "openai/whisper-large-v3"),
    "indic": ("ctc", "ai4bharat/indicwav2vec_v1_bengali"),
    "xlsr":  ("ctc", "arijitx/wav2vec2-xls-r-300m-bengali"),
}
ROOT = os.path.dirname(glob.glob("/kaggle/input/**/test.tsv", recursive=True)[0])
OUT = "/kaggle/working"

# ---- items: random sample of Common Voice 13 Bengali test (standard Bangla, read speech)
S = pd.read_csv(f"{ROOT}/test.tsv", sep="\t", quoting=3).rename(columns={"path": "file_name", "sentence": "transcripts"})
S = S.sample(2500, random_state=0).sort_values("file_name").reset_index(drop=True); S["split"] = "test"; S["region"] = "standard"
if LIMIT: S = S.head(LIMIT)
print("items:", len(S), flush=True)

def norm(t):
    t = re.sub(r"<[^>]*>", " ", str(t))
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t).split()

def lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[-1]

def align(ref, hyp):
    """per hyp word: err (0/1), aligned ref word, char-level lev ratio (1.0 for insertions); plus utterance edit distance"""
    R, H = len(ref), len(hyp)
    D = np.zeros((R + 1, H + 1), int); D[:, 0] = range(R + 1); D[0, :] = range(H + 1)
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            D[i, j] = min(D[i-1, j] + 1, D[i, j-1] + 1, D[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    i, j = R, H
    err, rw, ratio = [0] * H, [""] * H, [0.0] * H
    while j > 0:
        if i > 0 and D[i, j] == D[i-1, j-1] + (ref[i-1] != hyp[j-1]):
            rw[j-1] = ref[i-1]
            if ref[i-1] != hyp[j-1]:
                err[j-1] = 1; ratio[j-1] = lev(ref[i-1], hyp[j-1]) / max(len(ref[i-1]), len(hyp[j-1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i-1, j] + 1:
            i -= 1
        else:
            err[j-1] = 1; ratio[j-1] = 1.0; j -= 1
    return err, rw, ratio, int(D[R, H])

def load(r):
    a, sr = sf.read(f"{ROOT}/test/{r.file_name}", dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if sr != 16000: a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a[:30 * 16000]

def feats(p, ent=None, mar=None):
    p = np.asarray(p)
    f = dict(mean_p=p.mean(), min_p=p.min(), first_p=p[0], last_p=p[-1], mean_logp=np.log(np.clip(p, 1e-9, 1)).mean(), n_tok=len(p))
    f["mean_ent"] = float(np.mean(ent)) if ent is not None else np.nan
    f["max_ent"] = float(np.max(ent)) if ent is not None else np.nan
    f["mean_margin"] = float(np.mean(mar)) if mar is not None else np.nan
    return f

# ---- model runners: return, per clip, list of (word, token-feature dict)
def run_whisper(mid, auds):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, GenerationConfig
    proc = WhisperProcessor.from_pretrained(mid)
    model = WhisperForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.float16).cuda().eval()
    if "tugstugi" in mid: model.generation_config = GenerationConfig.from_pretrained("openai/whisper-medium")
    tok = proc.tokenizer; eos = tok.eos_token_id
    pre_ids = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|bn|>", "<|transcribe|>", "<|notimestamps|>"])
    def go(batch):
        x = proc.feature_extractor(batch, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
        with torch.no_grad():
            out = model.generate(x, language="bn", task="transcribe", num_beams=1, do_sample=False, max_new_tokens=200)
            seq = out if torch.is_tensor(out) else out.sequences
            pre = torch.tensor(pre_ids, device=seq.device)[None].expand(len(seq), -1)
            o = model(input_features=x, decoder_input_ids=torch.cat([pre, seq[:, :-1]], 1), output_hidden_states=True)
            lg = o.logits[:, pre.shape[1] - 1:].float(); hs = o.decoder_hidden_states[-1][:, pre.shape[1] - 1:].cpu().numpy()
            lp = lg.log_softmax(-1); pr = lp.exp()
            p = pr.gather(-1, seq[:, :, None])[..., 0].cpu().numpy()
            ent = (-(pr * lp).sum(-1)).cpu().numpy()
            t2 = pr.topk(2, -1).values; mar = (t2[..., 0] - t2[..., 1]).cpu().numpy()
        res = []
        for b in range(len(seq)):
            words, ids, ps, es, ms, last = [], [], [], [], [], [-1]
            def flush():
                if ids:
                    f = feats(ps, es, ms); f["_h"] = hs[b, last[0]]; words.append((tok.decode(ids).strip(), f))
            for ti, (t, pp, ee, mm) in enumerate(zip(seq[b].cpu().numpy(), p[b], ent[b], mar[b])):
                if t >= eos:
                    if t == eos: break
                    continue
                if tok.convert_ids_to_tokens(int(t)).startswith("Ġ"): flush(); ids, ps, es, ms = [], [], [], []
                ids.append(int(t)); ps.append(pp); es.append(ee); ms.append(mm); last[0] = ti
            flush(); res.append(words)
        return res
    return go, lambda: (model, )

def run_ctc(mid, auds):
    from transformers import AutoModelForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(mid)
    model = AutoModelForCTC.from_pretrained(mid).cuda().eval().half()
    tok = proc.tokenizer; blank = tok.pad_token_id; delim = tok.word_delimiter_token
    id2t = {v: k for k, v in tok.get_vocab().items()}
    def go(batch):
        inp = proc.feature_extractor(batch, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
        with torch.no_grad():
            lg = model(inp.input_values.cuda().half(), attention_mask=inp.attention_mask.cuda()).logits.float()
        lens = model._get_feat_extract_output_lengths(inp.attention_mask.sum(-1)).cpu().numpy()
        pr = lg.softmax(-1); lp = lg.log_softmax(-1)
        top2 = pr.topk(2, -1).values; mar = (top2[..., 0] - top2[..., 1]).cpu().numpy()
        ent = (-(pr * lp).sum(-1)).cpu().numpy(); mx = pr.max(-1).values.cpu().numpy(); am = pr.argmax(-1).cpu().numpy()
        res = []
        for b in range(len(am)):
            words, cur, prev = [], None, None
            for t in range(lens[b]):
                k = am[b, t]
                if k == prev: 
                    if cur is not None and k != blank: cur["p"].append(mx[b, t])
                    continue
                prev = k
                if k == blank: continue
                tk = id2t[int(k)]
                if tk == delim:
                    if cur and cur["c"]: words.append(cur)
                    cur = None; continue
                if cur is None: cur = dict(c=[], p=[], e=[], m=[])
                cur["c"].append(tk); cur["p"].append(mx[b, t]); cur["e"].append(ent[b, t]); cur["m"].append(mar[b, t])
            if cur and cur["c"]: words.append(cur)
            res.append([("".join(w["c"]), feats(w["p"], w["e"], w["m"])) for w in words])
        return res
    return go, lambda: (model, )

t0 = time.time()
for key in MODELS:
    kind, mid = MODEL_IDS[key]
    odir = f"{OUT}/feat_{key}"; os.makedirs(odir, exist_ok=True)
    go, _ = (run_whisper if kind == "whisper" else run_ctc)(mid, None)
    for c0 in range(0, len(S), CHUNK):
        fn = f"{odir}/words_{c0:05d}.csv"
        if os.path.exists(fn): continue
        sub = S.iloc[c0:c0 + CHUNK]; auds = [load(r) for r in sub.itertuples()]
        # sort by length inside chunk to reduce padding (CTC); keep mapping
        order = np.argsort([len(a) for a in auds]); rows, utts, H = [], [], []
        for s in range(0, len(order), BS):
            idx = order[s:s + BS]; res = go([auds[i] for i in idx])
            for i, words in zip(idx, res):
                r = sub.iloc[i]; hyp, hf = [], []
                for w, f in words:
                    for ww in norm(w): hyp.append(ww); hf.append(f)
                ref = norm(r.transcripts); err, rw, ratio, ed = align(ref, hyp)
                utts.append(dict(model=key, file=r.file_name, region=r.region, split=r.split, dur=len(auds[i]) / 16000, nref=len(ref), nhyp=len(hyp), ed=ed))
                for k, (w, f, e, rr, ra) in enumerate(zip(hyp, hf, err, rw, ratio)):
                    H.append(f["_h"]); rows.append(dict(model=key, file=r.file_name, region=r.region, split=r.split, pos=k, nhyp=len(hyp), word=w, ref_word=rr, err=e, lev_ratio=ra, **{kk: vv for kk, vv in f.items() if kk != "_h"}))
        pd.DataFrame(utts).to_csv(f"{odir}/utt_{c0:05d}.csv", index=False)
        pd.DataFrame(rows).to_csv(fn, index=False); np.save(f"{odir}/hid_{c0:05d}.npy", np.stack(H).astype(np.float16) if H else np.zeros((0, 1024), np.float16))
        print(key, c0 + len(sub), "/", len(S), round(time.time() - t0), "s", flush=True)
    del go; torch.cuda.empty_cache()
print("DONE", round(time.time() - t0), "s")
