"""Fill draft_template.md with generated tables -> draft.md  (run: python3 -I make_tables.py > tables.md && python3 -I build.py)"""
import re
tabs = {}
for blk in re.split(r"\n(?=### Table )", open("tables.md").read()):
    m = re.match(r"### Table (\w+)\.", blk)
    if m:
        ls = blk.strip().split("\n"); last = max(i for i, l in enumerate(ls) if l.startswith("|"))
        tabs[m.group(1)] = "\n".join(ls[:last + 1])
related = """**Table R. Closest prior work.** "Dialect speech" means the evaluation speech is a regional variety.

| Paper | Confidence or calibration measured | Shift studied | Dialect speech | Bangla |
|---|---|---|---|---|
| Li et al. 2022 | word/token, yes | text and acoustic domain (TED-LIUM, Switchboard) | no | no |
| Aggarwal et al. 2025 | word, yes | eight out-of-domain sets | no | no |
| Huo et al. 2025 | token, yes | additive noise | no | no |
| Jia and Van hamme 2026 | token and word, yes | Common Voice, US accent only | no | no |
| Liang et al. 2025 | sub-token probabilities | language resource level | no | no |
| Ballier et al. 2024 | ECE, yes | three languages | no | no |
| Dipto 2025; Hassan 2025; Hasan 2025 | no | regional dialects | yes | yes |
| This paper | word, yes | Standard Bangla and 12 dialects | yes | yes |"""
s = open("draft_template.md").read()
def sub(m):
    k = m.group(1); return "\n\n" + (related if k == "related" else tabs[k]) + "\n\n"
open("draft.md", "w").write(re.sub(r"<!--TABLE:(\w+)-->", sub, s))
print("tables:", sorted(tabs))
