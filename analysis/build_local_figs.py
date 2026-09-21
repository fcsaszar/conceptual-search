#!/usr/bin/env python3
"""Local-search deliverables: the ladder table and the trajectory figure.

Table (always rebuilt): revisions.tex --- step and accepted change
only; quality moved to the figure per the measure change.

Figure (rebuilt when reference-set scores exist): local-trajectory --- the
incumbent's win rate against the thirty-venture reference set after each round,
with the originals as a tick rug at the right edge.

Usage: python3 analysis/build_local_figs.py --run-id <id>
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_LOCAL, C_SEED, C_HELD = "#b65f45", "#b8b8b8", "#4a4a4a"
INK, INK2 = "#1a1a1a", "#555555"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9.5,
    "axes.edgecolor": "#3c3c3c", "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": "white", "pdf.fonttype": 42,
})

sq = lambda s: " ".join(str(s).split())


def latex_escape(s):
    s = re.sub(r"(?<=[\s(])'([^']+)'(?=[\s.,;:)])", r"`\1'", " " + str(s))[1:]
    return (s.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
             .replace("_", "\\_").replace("$", "\\$"))



def championship_overrides(BASE, run_id):
    """text-key -> championship-restricted win rate, for plans in both data
    sources (the championship is canonical; bipartite scores fill the rest)."""
    import csv as _csv, json as _json, re as _re
    _csv.field_size_limit(10**9)
    champ = _json.load(open(BASE / "analysis" / "universe-champ-wr.json"))
    sq_ = lambda s: " ".join(str(s).split())
    out = {}
    ent = BASE / "out" / f"{run_id}-polish-entrants.csv"
    if ent.exists():
        by_id = {}
        for r in _csv.DictReader(open(ent)):
            by_id[r["id"]] = sq_(r["plan_text"])
        for cid, v in champ.items():
            m = _re.match(r"(g12-s\d+)-(parent|polished)", cid)
            if m:
                key = f"{m.group(1)}-{m.group(2)}"
                for eid, text in by_id.items():
                    if eid.startswith(key):
                        out[text] = v["universe_wr"]
    lc = BASE / "out" / f"{run_id}-local-candidates.csv"
    if lc.exists():
        sel = [r for r in _csv.DictReader(open(lc)) if r["selected"] == "True"]
        if sel:
            lf = [v for v in champ.values() if v["kind"] == "evolved-local"]
            if lf:
                out[sq_(sel[-1]["plan_text"])] = lf[0]["universe_wr"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--seed-id", default="clockchain")
    ap.add_argument("--firm-set", default="in/firm-set-15-evalrank.txt")
    args = ap.parse_args()
    BASE = Path(args.base_dir)
    OUT = BASE / "out"
    RID = args.run_id

    lc = pd.read_csv(OUT / f"{RID}-local-candidates.csv")
    lpath = lc[lc["selected"] == True].sort_values("step").copy()

    scores_path = OUT / f"{RID}-universe-common-quality.csv"
    if not scores_path.exists():
        print("reference-set scores not found; cannot build table or figure")
        return
    uq = pd.read_csv(scores_path)
    U = dict(zip(uq["text_key"].map(sq), uq["benchmark_quality"]))
    U.update(championship_overrides(BASE, RID))
    champ = json.load(open(BASE / "analysis" / "universe-champ-wr.json"))

    xs, ys = [], []
    for _, r in lpath.iterrows():
        key = sq(r["plan_text"])
        if r["step"] == 0:
            # the seed is an original; use its restricted championship rate
            # (the bipartite score would count its two self-matches as ties)
            ys.append(champ[args.seed_id]["universe_wr"])
        elif key in U:
            ys.append(float(U[key]))
        else:
            print(f"WARNING: step {r['step']} state missing from reference-set scores")
            continue
        xs.append(int(r["step"]))

    # ---- table: round, accepted change, performance (with trend arrow)
    perf_by_step = dict(zip(xs, ys))
    lines, prev = [], perf_by_step.get(0)
    if prev is not None:
        lines.append(f"0 & \\emph{{Seed concept (\\texttt{{{args.seed_id}}}), "
                     f"before any revision.}} & {prev*100:.0f}\\% & \\\\")
    for _, r in lpath.iterrows():
        if r["step"] == 0:
            continue
        change = ("\\emph{No variant beats the incumbent; the concept is retained.}"
                  if not isinstance(r["requested_change"], str)
                  else latex_escape(sq(r["requested_change"])))
        wr = perf_by_step.get(int(r["step"]))
        if wr is None:
            cell = "-- &"
        else:
            arrow = ("$\\uparrow$" if wr > prev + 0.001 else
                     "$\\downarrow$" if wr < prev - 0.001 else "$=$")
            cell = f"{wr*100:.0f}\\% & {arrow}"
            prev = wr
        lines.append(f"{int(r['step'])} & {change} & {cell} \\\\")
    # Performance header right-flush over its own two columns; the pair is
    # sized to the header so nothing widens and the symbol ends at the edge
    (BASE / "figs" / "revisions.tex").write_text(
        "\\begin{tabular}{@{}>{\\centering\\arraybackslash}p{1.2cm}"
        ">{\\raggedright\\arraybackslash}p{\\dimexpr\\textwidth-1.2cm-2.05cm-0.3cm-4\\tabcolsep-2pt\\relax}"
        ">{\\raggedleft\\arraybackslash}p{2.05cm}@{\\,}"
        ">{\\raggedleft\\arraybackslash}p{0.3cm}@{}}\n\\toprule\n"
        "\\textbf{Round} & \\textbf{Accepted change} & "
        "\\multicolumn{2}{r@{}}{\\textbf{Performance}} \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print("wrote figs/revisions.tex")

    originals = {k: v["universe_wr"] for k, v in champ.items()
                 if v["kind"] == "original"}
    archive = {l.split("#")[0].strip()
               for l in open(BASE / args.firm_set) if l.strip()}

    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    ax.plot(xs, ys, color=C_LOCAL, lw=1.6, marker="o", markersize=4.2,
            zorder=5, clip_on=False)
    ax.annotate(args.seed_id, (xs[0], ys[0]), xytext=(3, -8),
                textcoords="offset points", fontsize=8, color=INK2)
    rug_x = 12.75
    for k, wr in originals.items():
        ax.scatter([rug_x], [wr], marker="_", s=52, zorder=3, clip_on=False,
                   color="#9a9a9a" if k in archive else C_HELD)
    top2 = sorted(originals.items(), key=lambda kv: -kv[1])[:2]
    for j, (k, wr) in enumerate(top2):
        ax.annotate(k, (rug_x, wr), xytext=(6, 3 if j == 0 else -5),
                    textcoords="offset points",
                    va="center", fontsize=8, color=INK2, annotation_clip=False)
    ax.set_xlim(0, 12.75)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(0, 13, 2))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Local-search round")
    ax.set_ylabel("Performance (win rate vs. the reference set)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    (BASE / "figs").mkdir(exist_ok=True)
    fig.savefig(BASE / "figs" / "local-trajectory.pdf")
    fig.savefig(BASE / "figs" / "local-trajectory.png", dpi=170)
    print(f"wrote local-trajectory ({len(xs)} states, "
          f"start {ys[0]:.1%} end {ys[-1]:.1%})")


if __name__ == "__main__":
    main()
