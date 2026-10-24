"""draft.md -> arxiv/main.tex (limited Markdown subset used by the draft). Run: python3 -I to_tex.py"""
import re
src = open("draft.md", encoding="utf8").read().split("\n")
ESC = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}
def esc(t):
    t = "".join(ESC.get(c, c) for c in t)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = t.replace("~", r"\textasciitilde{}").replace("<", r"\textless{}").replace(">", r"\textgreater{}")
    t = re.sub(r"(?<![\w}])-(\d)", r"$-$\1", t)                       # minus signs
    t = re.sub(r"\[(TK[^\]]*)\]", lambda m: r"\textbf{[" + m.group(1) + "]}", t)
    def cite(m):
        keys = [k.strip() for k in m.group(1).split(";")]
        return r"~\cite{" + ",".join(keys) + "}" if all(re.fullmatch(r"[A-Za-z]+\d{4}", k) for k in keys) else m.group(0)
    t = re.sub(r"\s?\[([A-Za-z]+\d{4}(?:;\s*[A-Za-z]+\d{4})*)\]", cite, t)
    t = re.sub(r"\"([^\"]+)\"", r"``\1''", t)
    return t
out, para, i = [], [], 0
def flush():
    global para
    if para: out.append(esc(" ".join(para)) + "\n"); para = []
def table(rows, caption):
    hdr = [c.strip() for c in rows[0].strip("|").split("|")]; body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
    wide = hdr[0] == "Paper"
    spec = "p{3.4cm}p{2.9cm}p{4.6cm}cc" if wide else "l" + "c" * (len(hdr) - 1)
    tex = [r"\begin{table}[htb]", r"\centering", r"\par\noindent\begin{minipage}{\linewidth}\small " + esc(caption) + r"\end{minipage}\par\smallskip", (r"\footnotesize" if wide else r"\small")]
    tex.append(r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{" if not wide else "")
    tex.append(r"\begin{tabular}{" + spec + "}"); tex.append(r"\hline"); tex.append(" & ".join(r"\textbf{" + esc(h) + "}" for h in hdr) + r" \\ \hline")
    for r in body: tex.append(" & ".join(esc(c).replace(r"\textbf{", r"\textbf{") for c in r) + r" \\")
    tex.append(r"\hline\end{tabular}" + ("}" if not wide else "")); tex.append(r"\end{table}")
    out.append("\n".join(x for x in tex if x != "") + "\n")
caption = None; lastpara = None; in_abs = False; in_enum = False; in_refs = False; refs = []
while i < len(src):
    ln = src[i]
    if ln.startswith("# "): title = ln[2:]; i += 1; continue
    if ln.startswith("## "):
        flush()
        if in_abs: out.append(r"\end{abstract}"); in_abs = False
        if in_enum: out.append(r"\end{enumerate}"); in_enum = False
        h = ln[3:]
        if h == "Abstract": out.append(r"\begin{abstract}"); in_abs = True
        elif h == "References": in_refs = True
        else:
            h2 = re.sub(r"^\d+\.\s*", "", h); star = "*" if h2 == h else ""
            out.append(r"\section" + star + "{" + esc(h2) + "}")
        i += 1; continue
    if ln.startswith("### Table ") or ln.startswith("**Table "):
        flush(); caption = re.sub(r"^### ", "", ln).replace("**", "", 2) if ln.startswith("### ") else ln.replace("**", "", 2); i += 1; continue
    if ln.startswith("### "):
        flush(); out.append(r"\subsection{" + esc(re.sub(r"^\d+\.\d+\s*", "", ln[4:])) + "}"); i += 1; continue
    mfig = re.match(r"!\[(.*)\]\((.*)\)", ln)
    if mfig:
        flush(); out.append(r"\begin{figure}[htb]\centering\includegraphics[width=\linewidth]{" + mfig.group(2) + r"}\par\smallskip\begin{minipage}{\linewidth}\small Figure 1. " + esc(mfig.group(1)) + r"\end{minipage}\end{figure}" + "\n"); i += 1; continue
    if ln.startswith("|"):
        flush(); rows = []
        while i < len(src) and src[i].startswith("|"): rows.append(src[i]); i += 1
        cap = caption or (lastpara.rstrip(":") if lastpara else "")
        if not caption and lastpara and out and out[-1].strip() == esc(lastpara).strip(): out.pop()
        table(rows, cap); caption = None; lastpara = None; continue
    if in_refs:
        m = re.match(r"\[(\w+)\]\s+(.*)", ln)
        if m: refs.append((m.group(1), m.group(2)))
        i += 1; continue
    if ln.startswith("Spearman(gap"): i += 1; continue
    if re.match(r"^\d+\.\s", ln):
        flush()
        if not in_enum: out.append(r"\begin{enumerate}"); in_enum = True
        out.append(r"\item " + esc(re.sub(r"^\d+\.\s", "", ln))); i += 1; continue
    if not ln.strip():
        flush()
        if in_enum: out.append(r"\end{enumerate}"); in_enum = False
        i += 1; continue
    if i < 6 and ln.strip() and "author" not in globals(): author = ln.strip(); i += 1; continue
    para.append(ln.strip()); lastpara = ln.strip(); i += 1
flush()
bib = "\n".join(r"\bibitem{" + k + "} " + esc(v) for k, v in refs)
tex = r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\title{""" + esc(title) + r"""}
\author{""" + r" \\\\ ".join(esc(a.strip()) for a in author.split(";")) + r"""}
\date{}
\begin{document}
\maketitle
""" + "\n\n".join(out) + r"""
\begin{thebibliography}{99}
""" + bib + r"""
\end{thebibliography}
\end{document}
"""
open("arxiv/main.tex", "w", encoding="utf8").write(tex)
print("wrote arxiv/main.tex", len(tex), "chars;", len(refs), "refs")
